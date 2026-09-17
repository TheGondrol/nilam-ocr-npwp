"""
Postprocess service module.

Handles postprocessing of OCR results.
"""

import aiohttp
import asyncio
import time
import json
from src.core.config import get_settings
from src.core.logging import get_logger
from src.api.models import ServiceResult, PostprocessData

logger = get_logger(__name__)


async def request_postprocess(
    session: aiohttp.ClientSession, ocr_raw: list, request_id: str
) -> ServiceResult[PostprocessData]:
    """
    Postprocess the OCR result.

    Args:
        session: The aiohttp session.
        ocr_raw: Raw OCR data (list with bounding boxes and text) to be postprocessed.
        request_id: The request ID.

    Returns:
        ServiceResult containing PostprocessData on success, or error details on failure.
    """
    start_time = time.time()
    settings = get_settings()
    postprocess_config = settings.services.postprocess

    # Convert list to string representation as expected by API
    body = {"ocr_text": json.dumps(ocr_raw)}
    headers = {"x-request-id": request_id, "x-api-key": postprocess_config.api_key or ""}

    try:
        timeout = aiohttp.ClientTimeout(total=postprocess_config.timeout)
        async with session.post(
            postprocess_config.url, json=body, timeout=timeout, headers=headers
        ) as res:
            if res.status == 200:
                elapsed = time.time() - start_time
                logger.info(f"[{request_id}] Postprocess completed in {elapsed:.2f}s")
                result_json = await res.json()
                data = result_json.get("data", result_json)
                return ServiceResult.success(PostprocessData(results=data))
            else:
                elapsed = time.time() - start_time
                logger.warning(
                    f"[{request_id}] Postprocess service responded with status {res.status} elapsed time {elapsed:.2f}s"
                )
                return ServiceResult.fail(
                    error_type="http_error",
                    message=f"Postprocess service returned status {res.status}",
                    status_code=res.status,
                )

    except asyncio.TimeoutError:
        elapsed = time.time() - start_time
        logger.error(f"[{request_id}] Postprocess service timeout after {elapsed:.2f}s")
        return ServiceResult.fail(
            error_type="timeout",
            message=f"Postprocess request timed out after {elapsed:.2f}s",
        )

    except aiohttp.ClientError as e:
        elapsed = time.time() - start_time
        logger.exception(f"[{request_id}] Postprocess service connection error after {elapsed:.2f}s: {e}")
        return ServiceResult.fail(
            error_type="connection_error",
            message=f"Postprocess connection error: {str(e)}",
            details=f"Duration: {elapsed:.2f}s",
        )

    except Exception as e:
        elapsed = time.time() - start_time
        logger.exception(
            f"[{request_id}] Unexpected error during postprocess service call after {elapsed:.2f}s: {e}"
        )
        return ServiceResult.fail(
            error_type="unknown_error",
            message=f"Unexpected postprocess error: {str(e)}",
            details=f"Duration: {elapsed:.2f}s",
        )
