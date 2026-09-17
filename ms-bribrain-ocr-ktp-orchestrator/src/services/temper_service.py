"""
Temper detection service module.

Handles tampering detection requests to check for document manipulation.
"""

import aiohttp
import asyncio
import time

from src.core.config import get_settings
from src.core.logging import get_logger
from src.api.models import ServiceResult, TemperData

logger = get_logger(__name__)


async def request_temper(
    session: aiohttp.ClientSession,
    file_bytes: bytes,
    filename: str,
    content_type: str,
    request_id: str,
) -> ServiceResult[TemperData]:
    """
    Perform temper on the given file.

    Args:
        session: The aiohttp session.
        file_bytes: The file content in bytes.
        filename: The name of the file.
        content_type: The MIME type of the file.
        request_id: The request ID.

    Returns:
        ServiceResult with:
        - success: No tampering detected
        - rejection: Tampering detected
        - error: Service failure (timeout, HTTP error)
    """
    start_time = time.time()
    settings = get_settings()
    temper_config = settings.services.temper

    form = aiohttp.FormData()
    form.add_field("file", file_bytes, filename=filename, content_type=content_type)
    headers = {"x-request-id": request_id, "x-api-key": temper_config.api_key or ""}

    try:
        timeout = aiohttp.ClientTimeout(total=temper_config.timeout)
        async with session.post(
            temper_config.url, data=form, timeout=timeout, headers=headers
        ) as res:
            if res.status != 200:
                logger.warning(
                    f"[{request_id}] Temper service responded with status {res.status} for file {filename}"
                )
                return ServiceResult.fail(
                    error_type="http_error",
                    message=f"Temper service returned status {res.status}",
                    status_code=res.status,
                    details=f"File: {filename}",
                )

            res_json = await res.json()
            data = res_json.get("data", res_json)
            prediction = data.get("prediction") if isinstance(data, dict) else res_json.get("prediction")
            elapsed = time.time() - start_time
            logger.info(f"[{request_id}] Temper check completed for {filename} in {elapsed:.2f}s")

            if prediction == "good":
                # No tampering detected
                return ServiceResult.success(TemperData(passed=True))
            else:
                # Tampering detected - business logic rejection
                return ServiceResult.reject(
                    rejection_type="temper",
                    message="Image tampering detected",
                    details={"prediction": prediction},
                )

    except asyncio.TimeoutError:
        elapsed = time.time() - start_time
        logger.error(f"[{request_id}] Temper service timeout after {elapsed:.2f}s for file {filename}")
        return ServiceResult.fail(
            error_type="timeout",
            message=f"Temper request timed out after {elapsed:.2f}s",
            details=f"File: {filename}",
        )

    except aiohttp.ClientError as e:
        elapsed = time.time() - start_time
        logger.exception(
            f"[{request_id}] Temper service connection error after {elapsed:.2f}s for file {filename}: {e}"
        )
        return ServiceResult.fail(
            error_type="connection_error",
            message=f"Temper connection error: {str(e)}",
            details=f"File: {filename}, Duration: {elapsed:.2f}s",
        )

    except Exception as e:
        elapsed = time.time() - start_time
        logger.exception(
            f"[{request_id}] Unexpected error during temper service call after {elapsed:.2f}s for file {filename}: {e}"
        )
        return ServiceResult.fail(
            error_type="unknown_error",
            message=f"Unexpected temper error: {str(e)}",
            details=f"File: {filename}, Duration: {elapsed:.2f}s",
        )
