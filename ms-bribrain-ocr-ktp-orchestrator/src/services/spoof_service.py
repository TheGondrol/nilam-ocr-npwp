"""
Spoof detection service module.

Handles lamination and recapture detection.
"""

import aiohttp
import asyncio
import time

from src.core.config import get_settings
from src.core.logging import get_logger
from src.api.models import ServiceResult, SpoofData

logger = get_logger(__name__)


async def request_lamination(
    session: aiohttp.ClientSession,
    file_bytes: bytes,
    filename: str,
    content_type: str,
    request_id: str,
) -> ServiceResult[SpoofData]:
    """
    Perform lamination detection on the given image.

    Args:
        session: The aiohttp session.
        file_bytes: The image content in bytes.
        filename: The name of the image file.
        content_type: The MIME type of the image.
        request_id: The request ID.

    Returns:
        ServiceResult with:
        - success: No lamination detected
        - rejection: Lamination detected (spoof)
        - error: Service failure (timeout, HTTP error)
    """
    start_time = time.time()
    settings = get_settings()
    lamination_config = settings.services.lamination

    form = aiohttp.FormData()
    form.add_field("file", file_bytes, filename=filename, content_type=content_type)
    headers = {"x-request-id": request_id, "x-api-key": lamination_config.api_key or ""}

    try:
        timeout = aiohttp.ClientTimeout(total=lamination_config.timeout)
        async with session.post(
            lamination_config.url, data=form, timeout=timeout, headers=headers
        ) as res:
            if res.status == 200:
                res_json = await res.json()
                data = res_json.get("data", res_json)
                elapsed = time.time() - start_time
                prediction = data.get("prediction") if isinstance(data, dict) else None
                logger.info(
                    f"[{request_id}] Lamination check for {filename}: {prediction} "
                    f"({elapsed:.2f}s)"
                )

                # Check if lamination was detected (spoof)
                if prediction == "UNLAMINATED":
                    return ServiceResult.reject(
                        rejection_type="spoof_lamination",
                        message="Lamination detected on document",
                        details=data,
                    )
                else:
                    # No lamination detected
                    return ServiceResult.success(SpoofData(passed=True))
            else:
                elapsed = time.time() - start_time
                logger.warning(
                    f"[{request_id}] Lamination service responded with status {res.status} for {filename} elapsed time {elapsed:.2f}s"
                )
                return ServiceResult.fail(
                    error_type="http_error",
                    message=f"Lamination service returned status {res.status}",
                    status_code=res.status,
                    details=f"File: {filename}",
                )

    except asyncio.TimeoutError:
        elapsed = time.time() - start_time
        logger.error(f"[{request_id}] Lamination service timeout after {elapsed:.2f}s for {filename}")
        return ServiceResult.fail(
            error_type="timeout",
            message=f"Lamination service timed out after {elapsed:.2f}s",
            details=f"File: {filename}",
        )

    except aiohttp.ClientError as e:
        elapsed = time.time() - start_time
        logger.exception(
            f"[{request_id}] Lamination service connection error after {elapsed:.2f}s for {filename}: {e}"
        )
        return ServiceResult.fail(
            error_type="connection_error",
            message=f"Lamination service connection error: {str(e)}",
            details=f"File: {filename}, Duration: {elapsed:.2f}s",
        )

    except Exception as e:
        elapsed = time.time() - start_time
        logger.exception(
            f"[{request_id}] Lamination service error after {elapsed:.2f}s for {filename}: {e}"
        )
        return ServiceResult.fail(
            error_type="unknown_error",
            message=f"Lamination service unexpected error: {str(e)}",
            details=f"File: {filename}, Duration: {elapsed:.2f}s",
        )


async def request_recapture(
    session: aiohttp.ClientSession,
    file_bytes: bytes,
    filename: str,
    content_type: str,
    request_id: str,
) -> ServiceResult[SpoofData]:
    """
    Perform recapture detection on the given image.

    Args:
        session: The aiohttp session.
        file_bytes: The image content in bytes.
        filename: The name of the image file.
        content_type: The MIME type of the image.
        request_id: The request ID.

    Returns:
        ServiceResult with:
        - success: No recapture detected
        - rejection: Recapture detected (spoof)
        - error: Service failure (timeout, HTTP error)
    """
    start_time = time.time()
    settings = get_settings()
    recapture_config = settings.services.recapture

    form = aiohttp.FormData()
    form.add_field("file", file_bytes, filename=filename, content_type=content_type)
    headers = {"x-request-id": request_id, "x-api-key": recapture_config.api_key or ""}

    try:
        timeout = aiohttp.ClientTimeout(total=recapture_config.timeout)
        async with session.post(
            recapture_config.url, data=form, timeout=timeout, headers=headers
        ) as res:
            if res.status == 200:
                res_json = await res.json()
                data = res_json.get("data", res_json)
                elapsed = time.time() - start_time
                prediction = data.get("prediction") if isinstance(data, dict) else None
                logger.info(
                    f"[{request_id}] Recapture check for {filename}: {prediction} "
                    f"({elapsed:.2f}s)"
                )

                # Check if recapture was detected (spoof)
                if prediction == "RECAPTURED":
                    return ServiceResult.reject(
                        rejection_type="spoof_recapture",
                        message="Recapture detected on document",
                        details=data,
                    )
                else:
                    # No recapture detected
                    return ServiceResult.success(SpoofData(passed=True))
            else:
                elapsed = time.time() - start_time
                logger.warning(
                    f"[{request_id}] Recapture service responded with status {res.status} for {filename} elapsed time {elapsed:.2f}s"
                )
                return ServiceResult.fail(
                    error_type="http_error",
                    message=f"Recapture service returned status {res.status}",
                    status_code=res.status,
                    details=f"File: {filename}",
                )

    except asyncio.TimeoutError:
        elapsed = time.time() - start_time
        logger.error(f"[{request_id}] Recapture service timeout after {elapsed:.2f}s for {filename}")
        return ServiceResult.fail(
            error_type="timeout",
            message=f"Recapture service timed out after {elapsed:.2f}s",
            details=f"File: {filename}",
        )

    except aiohttp.ClientError as e:
        elapsed = time.time() - start_time
        logger.exception(
            f"[{request_id}] Recapture service connection error after {elapsed:.2f}s for {filename}: {e}"
        )
        return ServiceResult.fail(
            error_type="connection_error",
            message=f"Recapture service connection error: {str(e)}",
            details=f"File: {filename}, Duration: {elapsed:.2f}s",
        )

    except Exception as e:
        elapsed = time.time() - start_time
        logger.exception(
            f"[{request_id}] Recapture service error after {elapsed:.2f}s for {filename}: {e}"
        )
        return ServiceResult.fail(
            error_type="unknown_error",
            message=f"Recapture service unexpected error: {str(e)}",
            details=f"File: {filename}, Duration: {elapsed:.2f}s",
        )


async def request_graycopy(
    session: aiohttp.ClientSession,
    file_bytes: bytes,
    filename: str,
    content_type: str,
    request_id: str,
) -> ServiceResult[SpoofData]:
    """
    Perform graycopy detection on the given image.

    Args:
        session: The aiohttp session.
        file_bytes: The image content in bytes.
        filename: The name of the image file.
        content_type: The MIME type of the image.
        request_id: The request ID.

    Returns:
        ServiceResult with:
        - success: No graycopy detected
        - rejection: Graycopy detected (spoof)
        - error: Service failure (timeout, HTTP error)
    """
    start_time = time.time()
    settings = get_settings()
    graycopy_config = settings.services.graycopy

    form = aiohttp.FormData()
    form.add_field("file", file_bytes, filename=filename, content_type=content_type)
    headers = {"x-request-id": request_id, "x-api-key": graycopy_config.api_key or ""}

    try:
        timeout = aiohttp.ClientTimeout(total=graycopy_config.timeout)
        async with session.post(
            graycopy_config.url, data=form, timeout=timeout, headers=headers
        ) as res:
            if res.status == 200:
                res_json = await res.json()
                data = res_json.get("data", res_json)
                elapsed = time.time() - start_time
                prediction = data.get("prediction") if isinstance(data, dict) else None
                logger.info(
                    f"[{request_id}] Graycopy check for {filename}: {prediction} "
                    f"({elapsed:.2f}s)"
                )

                # Check if graycopy was detected (spoof)
                if prediction == "GRAYCOPY":
                    return ServiceResult.reject(
                        rejection_type="spoof_graycopy",
                        message="Graycopy detected on document",
                        details=data,
                    )
                else:
                    # No graycopy detected
                    return ServiceResult.success(SpoofData(passed=True))
            else:
                elapsed = time.time() - start_time
                logger.warning(
                    f"[{request_id}] Graycopy service responded with status {res.status} for {filename} elapsed time {elapsed:.2f}s"
                )
                return ServiceResult.fail(
                    error_type="http_error",
                    message=f"Graycopy service returned status {res.status}",
                    status_code=res.status,
                    details=f"File: {filename}",
                )

    except asyncio.TimeoutError:
        elapsed = time.time() - start_time
        logger.error(f"[{request_id}] Graycopy service timeout after {elapsed:.2f}s for {filename}")
        return ServiceResult.fail(
            error_type="timeout",
            message=f"Graycopy service timed out after {elapsed:.2f}s",
            details=f"File: {filename}",
        )

    except aiohttp.ClientError as e:
        elapsed = time.time() - start_time
        logger.exception(
            f"[{request_id}] Graycopy service connection error after {elapsed:.2f}s for {filename}: {e}"
        )
        return ServiceResult.fail(
            error_type="connection_error",
            message=f"Graycopy service connection error: {str(e)}",
            details=f"File: {filename}, Duration: {elapsed:.2f}s",
        )

    except Exception as e:
        elapsed = time.time() - start_time
        logger.exception(
            f"[{request_id}] Graycopy service error after {elapsed:.2f}s for {filename}: {e}"
        )
        return ServiceResult.fail(
            error_type="unknown_error",
            message=f"Graycopy service unexpected error: {str(e)}",
            details=f"File: {filename}, Duration: {elapsed:.2f}s",
        )
