"""
OCR service module.

Handles OCR extraction requests to external OCR service.
"""

import aiohttp
import asyncio
import time

from src.core.config import get_settings
from src.core.logging import get_logger
from src.api.models import ServiceResult, OCRData

logger = get_logger(__name__)


async def request_ocr(
    session: aiohttp.ClientSession,
    file_bytes: bytes,
    filename: str,
    content_type: str,
    request_id: str,
) -> ServiceResult[OCRData]:
    """
    Perform OCR on the given file.

    Args:
        session: The aiohttp session.
        file_bytes: The file content in bytes.
        filename: The name of the file.
        content_type: The MIME type of the file.
        request_id: The request ID.

    Returns:
        ServiceResult containing OCRData on success, or error details on failure.
    """
    start_time = time.time()
    settings = get_settings()
    ocr_config = settings.services.ocr

    form = aiohttp.FormData()
    form.add_field("file", file_bytes, filename=filename, content_type=content_type)
    headers = {"x-request-id": request_id, "x-api-key": ocr_config.api_key or ""}

    try:
        timeout = aiohttp.ClientTimeout(total=ocr_config.timeout)
        async with session.post(
            ocr_config.url, data=form, timeout=timeout, headers=headers
        ) as res:
            if res.status != 200:
                elapsed = time.time() - start_time
                logger.warning(
                    f"[{request_id}] OCR service responded with status {res.status} for file {filename} elapsed time {elapsed:.2f}s"
                )
                return ServiceResult.fail(
                    error_type="http_error",
                    message=f"OCR service returned status {res.status}",
                    status_code=res.status,
                    details=f"File: {filename}",
                )

            res_json = await res.json()
            data = res_json.get("data", {})

            # OCR API returns raw data in "ocr_result" field (list with bboxes)
            ocr_result_field = data.get("ocr_result") if isinstance(data, dict) else None

            elapsed = time.time() - start_time
            logger.info(f"[{request_id}] OCR completed for {filename} in {elapsed:.2f}s")
            return ServiceResult.success(OCRData(text=ocr_result_field))

    except asyncio.TimeoutError:
        elapsed = time.time() - start_time
        logger.error(f"[{request_id}] OCR service timeout after {elapsed:.2f}s for file {filename}")
        return ServiceResult.fail(
            error_type="timeout",
            message=f"OCR request timed out after {elapsed:.2f}s",
            details=f"File: {filename}",
        )

    except aiohttp.ClientError as e:
        elapsed = time.time() - start_time
        logger.exception(
            f"[{request_id}] OCR service connection error after {elapsed:.2f}s for file {filename}: {e}"
        )
        return ServiceResult.fail(
            error_type="connection_error",
            message=f"OCR connection error: {str(e)}",
            details=f"File: {filename}, Duration: {elapsed:.2f}s",
        )

    except Exception as e:
        elapsed = time.time() - start_time
        logger.exception(
            f"[{request_id}] Unexpected error during OCR service call after {elapsed:.2f}s for file {filename}: {e}"
        )
        return ServiceResult.fail(
            error_type="unknown_error",
            message=f"Unexpected OCR error: {str(e)}",
            details=f"File: {filename}, Duration: {elapsed:.2f}s",
        )
