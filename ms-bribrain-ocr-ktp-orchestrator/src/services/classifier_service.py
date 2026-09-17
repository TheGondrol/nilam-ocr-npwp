"""
Classifier service module.

Handles KTP classification requests to detect if image is a valid KTP.
"""

import aiohttp
import asyncio
import time

from src.core.config import get_settings
from src.core.logging import get_logger
from src.api.models import ServiceResult, ClassifierData

logger = get_logger(__name__)


async def request_classifier(
    session: aiohttp.ClientSession,
    file_bytes: bytes,
    filename: str,
    content_type: str,
    request_id: str,
) -> ServiceResult[ClassifierData]:
    """
    Perform classifier on the given file.

    Args:
        session: The aiohttp session.
        file_bytes: The file content in bytes.
        filename: The name of the file.
        content_type: The MIME type of the file.
        request_id: The request ID.

    Returns:
        ServiceResult containing ClassifierData on success, or error details on failure.
    """
    start = time.time()
    settings = get_settings()
    classifier_config = settings.services.classifier

    form = aiohttp.FormData()
    form.add_field("file", file_bytes, filename=filename, content_type=content_type)
    headers = {"x-request-id": request_id, "x-api-key": classifier_config.api_key or ""}

    try:
        timeout = aiohttp.ClientTimeout(total=classifier_config.timeout)
        async with session.post(
            classifier_config.url, data=form, timeout=timeout, headers=headers
        ) as res:
            if res.status != 200:
                logger.warning(
                    f"[{request_id}] Classifier service responded with status {res.status} for file {filename}"
                )
                return ServiceResult.fail(
                    error_type="http_error",
                    message=f"Classifier service returned status {res.status}",
                    status_code=res.status,
                    details=f"File: {filename}",
                )

            res_json = await res.json()
            data = res_json.get("data", res_json)
            elapsed = time.time() - start
            logger.info(f"[{request_id}] Classifier check completed for {filename} in {elapsed:.2f}s")
            return ServiceResult.success(ClassifierData(raw_response=data))

    except asyncio.TimeoutError:
        elapsed = time.time() - start
        logger.error(f"[{request_id}] Classifier service timeout after {elapsed:.2f}s for file {filename}")
        return ServiceResult.fail(
            error_type="timeout",
            message=f"Classifier request timed out after {elapsed:.2f}s",
            details=f"File: {filename}",
        )

    except aiohttp.ClientError as e:
        elapsed = time.time() - start
        logger.exception(
            f"[{request_id}] Classifier service connection error after {elapsed:.2f}s for file {filename}: {e}"
        )
        return ServiceResult.fail(
            error_type="connection_error",
            message=f"Classifier connection error: {str(e)}",
            details=f"File: {filename}, Duration: {elapsed:.2f}s",
        )

    except Exception as e:
        elapsed = time.time() - start
        logger.exception(
            f"[{request_id}] Unexpected error during classifier service call after {elapsed:.2f}s for file {filename}: {e}"
        )
        return ServiceResult.fail(
            error_type="unknown_error",
            message=f"Unexpected classifier error: {str(e)}",
            details=f"File: {filename}, Duration: {elapsed:.2f}s",
        )
