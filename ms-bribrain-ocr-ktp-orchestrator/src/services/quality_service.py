"""
Quality checking service module.

Handles both rule-based and deep learning quality checks.
"""

import aiohttp
import asyncio
import heapq
import json
import time
from typing import Dict, Any, List, Optional

from src.core.config import get_settings
from src.core.logging import get_logger
from src.api.models import ServiceResult, QualityData

logger = get_logger(__name__)


async def request_quality_rulebase(
    session: aiohttp.ClientSession,
    file_bytes: bytes,
    filename: str,
    content_type: str,
    ocr_result: Optional[List],
    request_id: str,
) -> ServiceResult[QualityData]:
    """
    Validate image quality using rule-based checks.

    Args:
        session: The aiohttp session.
        file_bytes: The image content in bytes.
        filename: The name of the image file.
        content_type: The MIME type of the image.
        ocr_result: Raw OCR output list associated with the image.
        request_id: The request ID.

    Returns:
        ServiceResult with:
        - success: Quality check passed
        - rejection: Quality issues detected (blur, glare, rotation)
        - error: Service failure (timeout, HTTP error)
    """
    start_time = time.time()
    settings = get_settings()
    quality_config = settings.services.quality

    form = aiohttp.FormData()
    form.add_field("file", file_bytes, filename=filename, content_type=content_type)
    form.add_field("ocr_result", json.dumps(ocr_result))
    headers = {"x-request-id": request_id, "x-api-key": quality_config.api_key or ""}

    try:
        timeout = aiohttp.ClientTimeout(total=quality_config.timeout)
        async with session.post(
            quality_config.url, data=form, timeout=timeout, headers=headers
        ) as res:
            if res.status == 200:
                res_json = await res.json()
                data = res_json.get("data", res_json)
                is_blurry = data.get("is_blurry", False) if isinstance(data, dict) else False
                is_glare = data.get("is_glare", False) if isinstance(data, dict) else False
                is_rotated = data.get("is_rotated", False) if isinstance(data, dict) else False

                elapsed = time.time() - start_time
                logger.info(
                    f"[{request_id}] Rule-based quality check for {filename}: "
                    f"blur={is_blurry}, glare={is_glare}, rotated={is_rotated} "
                    f"({elapsed:.2f}s)"
                )

                # Business logic rejection - quality issues detected
                match (is_blurry, is_glare, is_rotated):
                    case (True, True, True):
                        return ServiceResult.reject(
                            rejection_type="quality_rulebase",
                            message="Poor image quality detected: blur, glare, and rotation/zoom",
                            details={
                                "is_blurry": True,
                                "is_glare": True,
                                "is_rotated": True,
                            },
                        )
                    case (True, False, False):
                        return ServiceResult.reject(
                            rejection_type="quality_rulebase",
                            message="Poor image quality detected: blur",
                            details={
                                "is_blurry": True,
                                "is_glare": False,
                                "is_rotated": False,
                            },
                        )
                    case (False, True, False):
                        return ServiceResult.reject(
                            rejection_type="quality_rulebase",
                            message="Poor image quality detected: glare",
                            details={
                                "is_blurry": False,
                                "is_glare": True,
                                "is_rotated": False,
                            },
                        )
                    case (False, False, True):
                        return ServiceResult.reject(
                            rejection_type="quality_rulebase",
                            message="Poor image quality detected: rotation/zoom",
                            details={
                                "is_blurry": False,
                                "is_glare": False,
                                "is_rotated": True,
                            },
                        )
                    case (False, False, False):
                        # Quality check passed
                        return ServiceResult.success(QualityData(passed=True))
                    case _:
                        # Multiple issues but not all three
                        issues = []
                        if is_blurry:
                            issues.append("blur")
                        if is_glare:
                            issues.append("glare")
                        if is_rotated:
                            issues.append("rotation/zoom")
                        return ServiceResult.reject(
                            rejection_type="quality_rulebase",
                            message=f"Poor image quality detected: {' and '.join(issues)}",
                            details={
                                "is_blurry": is_blurry,
                                "is_glare": is_glare,
                                "is_rotated": is_rotated,
                            },
                        )
            else:
                # Service error - HTTP non-200
                elapsed = time.time() - start_time
                logger.warning(
                    f"[{request_id}] Quality rulebase service responded with status {res.status} for {filename} elapsed time {elapsed:.2f}s"
                )
                return ServiceResult.fail(
                    error_type="http_error",
                    message="Quality rulebase service returned an error",
                    status_code=res.status,
                    details=f"File: {filename}",
                )

    except asyncio.TimeoutError:
        # Service error - timeout
        elapsed = time.time() - start_time
        logger.error(
            f"[{request_id}] Quality rulebase service timeout after {elapsed:.2f}s for {filename}"
        )
        return ServiceResult.fail(
            error_type="timeout",
            message="Quality rulebase service timed out",
            details=f"File: {filename}, Duration: {elapsed:.2f}s",
        )

    except aiohttp.ClientError as e:
        # Service error - connection error
        elapsed = time.time() - start_time
        logger.exception(
            f"[{request_id}] Quality rulebase service connection error after {elapsed:.2f}s for {filename}: {e}"
        )
        return ServiceResult.fail(
            error_type="connection_error",
            message="Quality rulebase service connection error",
            details=f"File: {filename}, Error: {str(e)}",
        )

    except Exception as e:
        # Service error - unknown
        elapsed = time.time() - start_time
        logger.exception(
            f"[{request_id}] Unexpected error during quality rulebase service call after {elapsed:.2f}s for {filename}: {e}"
        )
        return ServiceResult.fail(
            error_type="unknown_error",
            message="Quality rulebase service unexpected error",
            details=f"File: {filename}, Error: {str(e)}",
        )


async def get_lowest_confidence(ocr_raw: Optional[List], k: int = 5) -> List[Dict[str, Any]]:
    """
    Extract OCR lines with the lowest confidence scores.

    Args:
        ocr_raw: Raw OCR output containing bounding boxes, text, and confidence scores.
        k: Number of lowest-confidence OCR lines to return. Defaults to 5.

    Returns:
        A list of OCR entries containing bounding box, text, and confidence score.
    """
    if not ocr_raw:
        return []

    def _well_formed(line: Any) -> bool:
        # Expected shape: [bbox, [text, confidence]]. The raw OCR text comes from
        # an external service and is not validated, so skip malformed entries
        # (dict-shaped, too short, non-sequence) instead of indexing blindly and
        # 500-ing the whole request (BUG-09).
        return (
            isinstance(line, (list, tuple))
            and len(line) >= 2
            and isinstance(line[1], (list, tuple))
            and len(line[1]) >= 2
        )

    candidates = [line for line in ocr_raw if _well_formed(line)]

    lowest = heapq.nsmallest(
        k,
        candidates,
        key=lambda line: line[1][1],  # confidence is line[1][1]
    )

    return [
        {
            "bbox": line[0],
            "text": line[1][0],
            "confidence": line[1][1],
        }
        for line in lowest
    ]


async def request_quality_dl(
    session: aiohttp.ClientSession,
    file_bytes: bytes,
    filename: str,
    content_type: str,
    res_ocr: Optional[List],
    request_id: str,
) -> ServiceResult[QualityData]:
    """
    Validate image quality using a deep learning model.

    Args:
        session: The aiohttp session.
        file_bytes: The image content in bytes.
        filename: The name of the image file.
        content_type: The MIME type of the image.
        res_ocr: Raw OCR result used to extract low-confidence regions (can be None).
        request_id: The request ID for logging.

    Returns:
        ServiceResult with:
        - success: Quality check passed
        - rejection: Poor quality detected by DL model
        - error: Service failure (timeout, HTTP error)
    """
    start_time = time.time()
    settings = get_settings()
    qualitydl_config = settings.services.qualitydl

    crop = await get_lowest_confidence(res_ocr)
    form = aiohttp.FormData()
    form.add_field("file", file_bytes, filename=filename, content_type=content_type)
    form.add_field("crops", json.dumps(crop))
    headers = {"x-request-id": request_id, "x-api-key": qualitydl_config.api_key or ""}

    try:
        timeout = aiohttp.ClientTimeout(total=qualitydl_config.timeout)
        async with session.post(
            qualitydl_config.url, data=form, timeout=timeout, headers=headers
        ) as res:
            if res.status == 200:
                res_json = await res.json()
                data = res_json.get("data", res_json)
                quality = data.get("label") if isinstance(data, dict) else res_json.get("label")

                elapsed = time.time() - start_time
                logger.info(
                    f"[{request_id}] Quality DL check for {filename}: {quality} ({elapsed:.2f}s)"
                )

                if quality == "bad":
                    # Business logic rejection - bad quality
                    return ServiceResult.reject(
                        rejection_type="quality_dl",
                        message="Poor image quality detected by DL model",
                        details={"quality": quality},
                    )
                else:
                    # Quality check passed
                    return ServiceResult.success(QualityData(passed=True))
            else:
                # Service error - HTTP non-200
                elapsed = time.time() - start_time
                logger.warning(
                    f"[{request_id}] Quality DL service responded with status {res.status} for {filename} elapsed time {elapsed:.2f}s"
                )
                return ServiceResult.fail(
                    error_type="http_error",
                    message="Quality DL service returned an error",
                    status_code=res.status,
                    details=f"File: {filename}",
                )

    except asyncio.TimeoutError:
        # Service error - timeout
        elapsed = time.time() - start_time
        logger.error(f"[{request_id}] Quality DL service timeout after {elapsed:.2f}s for {filename}")
        return ServiceResult.fail(
            error_type="timeout",
            message="Quality DL service timed out",
            details=f"File: {filename}, Duration: {elapsed:.2f}s",
        )

    except aiohttp.ClientError as e:
        # Service error - connection error
        elapsed = time.time() - start_time
        logger.exception(
            f"[{request_id}] Quality DL service connection error after {elapsed:.2f}s for {filename}: {e}"
        )
        return ServiceResult.fail(
            error_type="connection_error",
            message="Quality DL service connection error",
            details=f"File: {filename}, Error: {str(e)}, Duration: {elapsed:.2f}s",
        )

    except Exception as e:
        # Service error - unknown
        elapsed = time.time() - start_time
        logger.exception(
            f"[{request_id}] Unexpected error during quality DL service call after {elapsed:.2f}s for {filename}: {e}"
        )
        return ServiceResult.fail(
            error_type="unknown_error",
            message="Quality DL service unexpected error",
            details=f"File: {filename}, Error: {str(e)}, Duration: {elapsed:.2f}s",
        )
