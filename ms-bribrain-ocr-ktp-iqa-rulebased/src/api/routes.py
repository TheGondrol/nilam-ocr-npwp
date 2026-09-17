"""
OCR Quality API endpoint.
Handles image quality checking requests.
"""
import asyncio
from fastapi import APIRouter, Depends, File, UploadFile, Form
from fastapi.responses import JSONResponse
from typing import Dict, Any, Optional

from src.api.dependencies import verify_api_key
import json
import time
from src.core.logging import get_logger, request_id_ctx
from src.core.config import get_config
from src.services.image_quality import image_quality, ImageDecodeError
from src.services.database_service import insert_log

logger = get_logger(__name__)
config = get_config()

router = APIRouter()

STATUS_DESCS = {
    200: "OK",
    400: "Bad Request",
    401: "Unauthorized",
    403: "Forbidden",
    404: "Not Found",
    405: "Method Not Allowed",
    413: "Payload Too Large",
    422: "Unprocessable Entity",
    500: "Internal Server Error",
    503: "Service Unavailable",
}


def _envelope(
    status_code: int,
    request_id: str,
    message: str = "Success",
    data=None,
    error_code: Optional[str] = None,
    errors=None,
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "status_code": status_code,
            "status_desc": STATUS_DESCS.get(status_code, "Unknown"),
            "message": message,
            "data": data if data is not None else "",
            "error_code": error_code,
            "errors": errors,
            "request_id": request_id,
        },
    )


@router.post("/ocr_quality")
async def check_ocr_quality(
    file: UploadFile = File(...),
    ocr_result: str = Form(...),
    _: str = Depends(verify_api_key),
) -> JSONResponse:
    """
    Check image quality based on blur, glare, and rotation.
    
    Args:
        file: Uploaded image file
        ocr_result: OCR results as string representation of list
        
    Returns:
        Dict containing quality check results
    """

    logger.info(f"Received OCR quality check request - filename: {file.filename}")
    request_id = request_id_ctx.get()
    start_time = time.time()
    req_payload = {"file_name": file.filename, "ocr_result": ocr_result}
    image_bytes = None  # Initialize for cleanup in finally block

    # Initialize state for finally block logging
    response_code = 500
    error_message = ""
    result: Dict[str, Any] = {}

    try:
        # Parse OCR result
        try:
            ocr_result_parsed = json.loads(ocr_result)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse OCR result: {e}")
            response_code = 400
            error_message = "Invalid OCR result format"
            return _envelope(400, request_id, message=error_message, error_code="INVALID_INPUT")

        # Enforce maximum upload size — reject early using the declared size,
        # before reading the whole file into memory.
        max_mb = config.server.max_file_size_mb
        max_bytes = max_mb * 1024 * 1024
        declared_size = getattr(file, "size", None)
        if declared_size is not None and declared_size > max_bytes:
            logger.warning(f"File too large (declared): {declared_size} bytes")
            response_code = 413
            error_message = f"File too large. Maximum size is {max_mb} MB."
            return _envelope(413, request_id, message=f"File too large. Maximum size is {max_mb} MB.", error_code="FILE_TOO_LARGE")

        # Read file bytes
        image_bytes = await file.read()
        logger.debug(f"Image file read - size: {len(image_bytes)} bytes")

        # Authoritative check, if the declared size was absent/wrong.
        if len(image_bytes) > max_bytes:
            logger.warning(f"File too large: {len(image_bytes)} bytes")
            response_code = 413
            error_message = f"File too large. Maximum size is {max_mb} MB."
            return _envelope(413, request_id, message=f"File too large. Maximum size is {max_mb} MB.", error_code="FILE_TOO_LARGE")

        # Process image quality (run in thread pool to avoid blocking event loop)
        result = await asyncio.to_thread(image_quality, image_bytes, ocr_result_parsed)
        response_code = 200

        logger.info(f"OCR quality check completed successfully for {file.filename}")
        return _envelope(200, request_id, data=result)

    except ImageDecodeError as e:
        logger.warning(f"Undecodable image upload for {file.filename}: {e}")
        response_code = 400
        error_message = f"Invalid image: {str(e)}"
        return _envelope(400, request_id, message=error_message, error_code="INVALID_INPUT")
    except Exception as e:
        logger.error(f"Error processing OCR quality request: {e}", exc_info=True)
        error_message = "Internal server error"
        return _envelope(500, request_id, message="Internal server error", error_code="INTERNAL_ERROR")
    finally:
        # Single logging call - always executed
        await insert_log(
            request_id=request_id,
            response_code=response_code,
            payload=req_payload,
            error_message=error_message,
            result=result if response_code == 200 else {},
            processing_time=time.time() - start_time
        )
        # CRITICAL: Always cleanup resources to prevent memory leaks
        await file.close()
        if image_bytes is not None:
            del image_bytes
