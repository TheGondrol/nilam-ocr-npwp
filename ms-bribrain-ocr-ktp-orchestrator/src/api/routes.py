"""
API routes module.

Defines all API endpoints for the OCR KTP orchestrator.
"""

import asyncio
import time
import uuid
from typing import Optional
from fastapi import APIRouter, Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.api.dependencies import verify_api_key, validate_request_id, reject_request_body
from src.core.logging import get_logger
from src.core.config import get_settings
from src.core.http_client import get_http_client
from src.api.models import ErrorResponse, OCRSuccessResponse, ErrorCode
from src.services.database_service import (
    insert_log,
    create_ocr_result,
    claim_ocr_result,
    update_ocr_result,
    get_ocr_result,
)
from src.schemas.database_schema import OcrStatus
from src.core.crypto import decrypt
from src.services.manage_service import ocr_process

logger = get_logger(__name__)
router = APIRouter()

STATUS_DESCS = {
    200: "OK",
    400: "Bad Request",
    401: "Unauthorized",
    403: "Forbidden",
    404: "Not Found",
    405: "Method Not Allowed",
    409: "Conflict",
    413: "Payload Too Large",
    422: "Unprocessable Entity",
    429: "Too Many Requests",
    500: "Internal Server Error",
}

# Error codes for HTTPExceptions whose detail is a plain string (e.g. FastAPI's
# default 404/405). Envelope-dict details (raised by our dependencies) already
# carry their own error_code and are passed through unchanged.
_ERROR_CODES = {
    401: ErrorCode.UNAUTHORIZED,
    403: ErrorCode.FORBIDDEN,
    404: ErrorCode.NOT_FOUND,
    405: ErrorCode.METHOD_NOT_ALLOWED,
}


def _envelope(
    status_code: int,
    request_id: str,
    message: str = "Success",
    data=None,
    error_code: Optional[str] = None,
    errors=None,
) -> JSONResponse:
    """Build a standard response envelope."""
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


@router.post(
    "/v1/generate-request-id",
    dependencies=[Depends(verify_api_key)],
    responses={
        200: {"description": "Request ID generated successfully"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
    summary="Generate a unique request ID",
    description="Generates a unique OCR request ID and creates a placeholder record for tracking",
)
async def generate_request_id():
    """Generate a unique request ID and create a placeholder record."""
    start_time = time.time()
    request_id = f"OCR_{uuid.uuid4()}"

    response_code = 200
    error_message = ""

    try:
        await create_ocr_result(request_id)
        return _envelope(200, request_id, data={"request_id": request_id})
    except Exception as e:
        logger.exception(f"[{request_id}] Failed to generate request ID: {e}")
        response_code = 500
        error_message = str(e)
        return _envelope(
            500, request_id,
            message="An unexpected error occurred. Please try again later.",
            error_code=ErrorCode.INTERNAL_ERROR,
        )
    finally:
        await insert_log(
            request_id=request_id,
            response_code=response_code,
            payload=None,
            error_message=error_message,
            result=None,
            processing_time=time.time() - start_time,
        )


@router.get(
    "/v1/get-ocr-result/{request_id}",
    dependencies=[Depends(verify_api_key), Depends(reject_request_body)],
    responses={
        200: {"description": "OCR result retrieved successfully"},
        404: {"model": ErrorResponse, "description": "Request ID not found"},
        400: {"model": ErrorResponse, "description": "Invalid request ID format"},
    },
    summary="Retrieve OCR result by request ID",
    description="Retrieves the OCR processing result and status for a given request ID",
)
async def get_result(request_id: str):
    """Retrieve OCR result by request ID."""
    start_time = time.time()
    response_code = 200
    error_message = ""

    try:
        validate_request_id(request_id)

        record = await get_ocr_result(request_id)

        if record is None:
            response_code = 404
            error_message = "No record found for the provided request ID."
            return _envelope(
                404, request_id,
                message=error_message,
                error_code=ErrorCode.REQUEST_NOT_FOUND,
            )

        return _envelope(
            200,
            request_id,
            data={
                "request_id": record.request_id,
                "status": record.status,
                "result": decrypt(record.result),
                "error_message": record.error_message,
                "created_at": record.created_at.isoformat(),
                "updated_at": record.updated_at.isoformat(),
            },
        )

    except HTTPException as e:
        # Validation failures (e.g. invalid request_id format) -- log then re-raise
        response_code = e.status_code
        error_message = (
            e.detail.get("message") if isinstance(e.detail, dict) else str(e.detail)
        )
        raise
    except Exception as e:
        logger.exception(f"[{request_id}] Failed to retrieve result: {e}")
        response_code = 500
        error_message = str(e)
        return _envelope(
            500, request_id,
            message="An unexpected error occurred. Please try again later.",
            error_code=ErrorCode.INTERNAL_ERROR,
        )

    finally:
        await insert_log(
            request_id=request_id,
            response_code=response_code,
            payload=None,
            error_message=error_message,
            result=None,
            processing_time=time.time() - start_time,
        )


@router.post(
    "/v1/extract-ocr",
    response_model=OCRSuccessResponse,
    dependencies=[Depends(verify_api_key)],
    responses={
        200: {"description": "OCR extraction successful"},
        400: {
            "model": ErrorResponse,
            "description": "Invalid input or quality check failed",
        },
        409: {"model": ErrorResponse, "description": "Request ID not found or already processed"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
    summary="Extract KTP data from image",
    description="Performs OCR extraction on KTP (Indonesian ID card) images with quality and spoof detection. Requires a pre-generated request_id from /v1/generate-id.",
)
async def extract_text_lines(
    request_id: str = Form(..., description="Pre-generated request ID from /v1/generate-id"),
    file: UploadFile = File(...),
):
    """
    Endpoint for extracting KTP data from images using OCR.

    Args:
        request_id: Pre-generated request ID from /v1/generate-id.
        file: Uploaded image file (JPEG/PNG).

    Returns:
        JSONResponse:
            - 200: Success, returns extracted KTP fields.
            - 400: Invalid file, low confidence, or poor image quality.
            - 409: request_id not found or already processed.
            - 500: Internal server error during processing.
    """
    start_time = time.time()

    # Tracking variables for the finally block -- initialized up front so the
    # audit log (and result update) always have defined values, even when an
    # early validation failure returns before the OCR pipeline runs.
    filename: Optional[str] = None
    file_bytes: Optional[bytes] = None
    claimed = False
    response_code = 500
    error_message = ""
    result_data: Optional[dict] = None
    final_status = OcrStatus.FAILED  # Default to failed; overwrite on success

    try:
        # Validate request_id format
        validate_request_id(request_id)
        logger.info(
            f"[{request_id}] Received OCR extraction request for file: {file.filename}"
        )

        filename = file.filename
        content_type = file.content_type
        settings = get_settings()
        max_file_size = settings.minio.max_size

        # Validate content_type BEFORE reading file bytes (fail fast)
        if content_type not in ["image/jpeg", "image/png", "image/jpg"]:
            logger.warning(
                f"[{request_id}] Invalid file type uploaded: {content_type} for file {filename}"
            )
            await file.close()
            response_code = 400
            error_message = "Invalid file type. Only JPEG, JPG, and PNG images are accepted."
            return _envelope(
                400, request_id,
                message=error_message,
                error_code=ErrorCode.INVALID_FILE_TYPE,
            )

        # Early size check via Content-Length header if available (fail fast before read)
        content_length = file.size if hasattr(file, "size") and file.size is not None else None
        if content_length is not None and content_length > max_file_size:
            logger.warning(
                f"[{request_id}] File too large (Content-Length): {content_length} bytes (limit: {max_file_size})"
            )
            await file.close()
            response_code = 413
            error_message = f"File size exceeds the maximum limit of {max_file_size // (1024 * 1024)} MB. Please upload a smaller image."
            return _envelope(
                413, request_id,
                message=error_message,
                error_code=ErrorCode.FILE_TOO_LARGE,
            )

        # Read file and immediately close to free resources
        file_bytes = await file.read()

        # Close the upload file to release the underlying SpooledTemporaryFile
        await file.close()

        # Authoritative size check on actual bytes read
        if len(file_bytes) > max_file_size:
            logger.warning(
                f"[{request_id}] File too large: {len(file_bytes)} bytes (limit: {max_file_size})"
            )
            response_code = 413
            error_message = f"File size exceeds the maximum limit of {max_file_size // (1024 * 1024)} MB. Please upload a smaller image."
            return _envelope(
                413, request_id,
                message=error_message,
                error_code=ErrorCode.FILE_TOO_LARGE,
            )

        # Atomic claim -- prevents concurrent duplicate processing
        try:
            claimed = await claim_ocr_result(request_id)
        except Exception as e:
            logger.exception(f"[{request_id}] Failed to claim request: {e}")
            response_code = 500
            error_message = str(e)
            return _envelope(
                500, request_id,
                message="An unexpected error occurred. Please try again later.",
                error_code=ErrorCode.INTERNAL_ERROR,
            )

        if not claimed:
            response_code = 409
            error_message = "This request ID was not found or has already been processed."
            return _envelope(
                409, request_id,
                message=error_message,
                error_code=ErrorCode.REQUEST_ALREADY_PROCESSED,
            )

        # Process OCR with overall timeout
        overall_timeout = settings.services.orchestrator.overall_timeout or 60

        # Get shared HTTP client session
        session = await get_http_client()

        try:
            result, pipeline_error = await asyncio.wait_for(
                ocr_process(
                    session, filename or "unknown", file_bytes, content_type or "image/jpeg", request_id
                ),
                timeout=overall_timeout
            )
        except asyncio.TimeoutError:
            elapsed = time.time() - start_time
            logger.error(
                f"[{request_id}] OCR pipeline timeout after {elapsed:.2f}s (limit: {overall_timeout}s)"
            )
            response_code = 500
            error_message = f"OCR pipeline timeout after {elapsed:.2f}s"
            return _envelope(
                500, request_id,
                message="Processing took too long. Please try again with a clearer image.",
                error_code=ErrorCode.PIPELINE_TIMEOUT,
            )

        if pipeline_error is not None:
            response_code = pipeline_error.http_status
            error_message = pipeline_error.message
            # Distinguish business rejections (4xx) from server errors (5xx) so
            # they are not both persisted as FAILED (BUG-15).
            final_status = (
                OcrStatus.FAILED
                if pipeline_error.http_status >= 500
                else OcrStatus.REJECTED
            )

            logger.info(
                f"[{request_id}] Image {filename} had "
                f"{'service error' if pipeline_error.http_status == 500 else 'rejection'} "
                f"during processing: [{pipeline_error.error_code}] {pipeline_error.message}"
            )

            return _envelope(
                pipeline_error.http_status,
                request_id,
                message=pipeline_error.message,
                error_code=pipeline_error.error_code,
                errors=pipeline_error.details,
            )

        # Success case
        logger.info(
            f"[{request_id}] OCR processing completed for {filename} in {time.time() - start_time:.2f}s"
        )
        response_code = 200
        error_message = ""
        final_status = OcrStatus.COMPLETED
        result_data = result

        return _envelope(200, request_id, data=result)

    except HTTPException as e:
        # Validation failures (e.g. invalid request_id format) -- log then re-raise
        response_code = e.status_code
        error_message = (
            e.detail.get("message") if isinstance(e.detail, dict) else str(e.detail)
        )
        raise
    except Exception as e:
        logger.exception(
            f"[{request_id}] Unexpected error processing {filename} after {time.time() - start_time:.2f}s: {e}"
        )
        response_code = 500
        error_message = str(e)

        return _envelope(
            500, request_id,
            message="An unexpected error occurred. Please try again later.",
            error_code=ErrorCode.INTERNAL_ERROR,
        )

    finally:
        # Single point of mutation -- only touch result state if we claimed it
        if claimed:
            await update_ocr_result(
                request_id,
                status=final_status,
                result=result_data if final_status == OcrStatus.COMPLETED else None,
                error_message=error_message if final_status != OcrStatus.COMPLETED else None,
            )

        # Single logging point - guaranteed to run for every path
        await insert_log(
            request_id=request_id,
            response_code=response_code,
            payload=filename,
            error_message=error_message,
            result=result_data,
            processing_time=time.time() - start_time,
        )

        if file_bytes is not None:
            del file_bytes


# ---------------------------------------------------------------------------
# Exception handlers
#
# The dependencies (verify_api_key, validate_request_id, reject_request_body)
# and the rate limiter already build a full envelope. But without these
# handlers, an HTTPException whose detail is that envelope dict is rendered by
# FastAPI as `{"detail": {envelope}}` (nested), and request-validation errors
# fall back to FastAPI's default `{"detail": [...]}`. These handlers ensure
# every error response is the same top-level envelope the route endpoints and
# the rate-limit middleware already return.
# ---------------------------------------------------------------------------


def register_exception_handlers(app: FastAPI) -> None:
    """Register handlers that render all errors as the top-level envelope."""

    @app.exception_handler(StarletteHTTPException)
    async def _http_exception_handler(request: Request, exc: StarletteHTTPException):
        # Dependencies raise HTTPException with a full envelope dict as detail —
        # return it directly as the top-level body (un-nest from `detail`).
        if isinstance(exc.detail, dict) and "status_code" in exc.detail:
            return JSONResponse(status_code=exc.status_code, content=exc.detail)
        # Otherwise (e.g. FastAPI's default 404/405 with a string detail) wrap it.
        message = exc.detail if isinstance(exc.detail, str) else "Request rejected"
        return _envelope(
            exc.status_code,
            None,
            message=message,
            error_code=_ERROR_CODES.get(exc.status_code, ErrorCode.INTERNAL_ERROR),
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_exception_handler(request: Request, exc: RequestValidationError):
        return _envelope(
            422,
            None,
            message="Request validation failed",
            error_code=ErrorCode.VALIDATION_ERROR,
            errors=jsonable_encoder(exc.errors()),
        )
