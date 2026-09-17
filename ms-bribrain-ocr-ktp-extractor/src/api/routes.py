"""API routes for OCR extraction"""

import logging
import time
from typing import Optional
from fastapi import APIRouter, Depends, FastAPI, File, UploadFile, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.api.dependencies import verify_api_key

from src.services.ocr_service import perform_ocr_async
from src.core.config import settings
from src.core.exceptions import ImageValidationError, OCRProcessingError, OCRInitializationError
from src.services.database_service import insert_log
from src.core.logging import request_id_ctx
from src.models.schemas import OCRSuccessResponse, ErrorResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["OCR"])

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

_ERROR_CODES = {
    400: "INVALID_INPUT",
    401: "UNAUTHORIZED",
    403: "FORBIDDEN",
    404: "NOT_FOUND",
    405: "METHOD_NOT_ALLOWED",
    422: "VALIDATION_ERROR",
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


async def _log_and_respond_error(
    request_id: str,
    status_code: int,
    error_message: str,
    filename: str,
    processing_time: float,
    error_code: str = "INTERNAL_ERROR",
) -> JSONResponse:
    """
    Helper to log error and return consistent error response.

    Args:
        request_id: Request tracking ID
        status_code: HTTP status code
        error_message: Error message to return
        filename: Original filename for logging
        processing_time: Time elapsed since request start
        error_code: Machine-readable error code

    Returns:
        JSONResponse with error details
    """
    await insert_log(
        request_id=request_id,
        response_code=status_code,
        payload=filename or "",
        error_message=error_message,
        result="",
        processing_time=processing_time
    )
    return _envelope(status_code, request_id, message=error_message, error_code=error_code)


@router.post(
    "/v1/ocr_extract",
    response_model=OCRSuccessResponse,
    responses={
        200: {"model": OCRSuccessResponse, "description": "Successful OCR extraction"},
        400: {"model": ErrorResponse, "description": "Invalid request (bad file type, size, or image)"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
    summary="Extract text from image",
    description="Perform OCR text extraction on uploaded image using PaddleOCR."
)
async def extract_text_lines(request: Request, file: UploadFile = File(...), _: str = Depends(verify_api_key)):
    """
    Endpoint for extracting text from images using OCR.

    Args:
        request: FastAPI request object
        file: Uploaded image file (JPEG/PNG)

    Returns:
        JSONResponse:
            - 200: Successful extraction with OCR results
            - 400: Invalid file (type, size, or corrupted)
            - 500: Internal server error
    """
    request_id = request_id_ctx.get()
    start_time = time.time()

    logger.info(f"Received OCR extraction request: {file.filename}")

    try:
        # Validate file type early
        if file.content_type not in settings.ocr_allowed_types:
            logger.warning(f"Invalid file type: {file.content_type}")
            allowed = ", ".join(settings.ocr_allowed_types)
            return await _log_and_respond_error(
                request_id=request_id,
                status_code=400,
                error_message=f"Invalid file type. Only {allowed} are supported.",
                filename=file.filename,
                processing_time=time.time() - start_time,
                error_code="INVALID_FILE_TYPE",
            )

        # Reject oversized uploads early, using the declared size, before
        # reading the whole file into memory.
        declared_size = getattr(file, "size", None)
        if declared_size is not None and declared_size > settings.ocr_max_size_mb * 1024 * 1024:
            logger.warning(f"File too large (declared): {declared_size} bytes")
            return await _log_and_respond_error(
                request_id=request_id,
                status_code=413,
                error_message=f"File too large. Maximum size is {settings.ocr_max_size_mb} MB.",
                filename=file.filename,
                processing_time=time.time() - start_time,
                error_code="FILE_TOO_LARGE",
            )

        # Read image bytes
        try:
            image_bytes = await file.read()
            file_size_mb = len(image_bytes) / (1024 * 1024)
            logger.info(f"File size: {file_size_mb:.2f} MB")

            # Check file size
            if file_size_mb > settings.ocr_max_size_mb:
                logger.warning(f"File too large: {file_size_mb:.2f} MB")
                return await _log_and_respond_error(
                    request_id=request_id,
                    status_code=413,
                    error_message=f"File too large. Maximum size is {settings.ocr_max_size_mb} MB.",
                    filename=file.filename,
                    processing_time=time.time() - start_time,
                    error_code="FILE_TOO_LARGE",
                )

        except Exception as e:
            logger.error(f"Failed to read file: {str(e)}", exc_info=True)
            return await _log_and_respond_error(
                request_id=request_id,
                status_code=400,
                error_message="Failed to read uploaded file.",
                filename=file.filename,
                processing_time=time.time() - start_time,
                error_code="IMAGE_PROCESSING_FAILED",
            )

        # Perform OCR on the image
        try:
            result = await perform_ocr_async(image_bytes)
            processing_time = time.time() - start_time

            logger.info(f"OCR completed successfully in {processing_time:.2f}s")

            # Log successful request
            await insert_log(
                request_id=request_id,
                response_code=200,
                payload=file.filename,
                error_message="",
                result=result,
                processing_time=processing_time
            )

            return _envelope(200, request_id, data={
                "ocr_result": result,
                "processing_time": round(processing_time, 2),
                "text_regions_count": len(result)
            })

        except ImageValidationError as e:
            logger.error(f"Image validation error: {e.message}")
            return await _log_and_respond_error(
                request_id=request_id,
                status_code=400,
                error_message=f"Image processing failed: {e.message}",
                filename=file.filename,
                processing_time=time.time() - start_time,
                error_code="IMAGE_PROCESSING_FAILED",
            )

        except (OCRProcessingError, OCRInitializationError) as e:
            logger.error(f"OCR error: {e.message}")
            return await _log_and_respond_error(
                request_id=request_id,
                status_code=500,
                error_message="OCR processing failed. Please try again.",
                filename=file.filename,
                processing_time=time.time() - start_time,
                error_code="OCR_PROCESSING_FAILED",
            )

        except ValueError as e:
            # Legacy support for ValueError from image processing
            logger.error(f"Image processing error: {str(e)}")
            return await _log_and_respond_error(
                request_id=request_id,
                status_code=400,
                error_message=f"Image processing failed: {str(e)}",
                filename=file.filename,
                processing_time=time.time() - start_time,
                error_code="IMAGE_PROCESSING_FAILED",
            )

        except RuntimeError as e:
            # Legacy support for RuntimeError from OCR
            logger.error(f"OCR execution error: {str(e)}")
            return await _log_and_respond_error(
                request_id=request_id,
                status_code=500,
                error_message="OCR processing failed. Please try again.",
                filename=file.filename,
                processing_time=time.time() - start_time,
                error_code="OCR_PROCESSING_FAILED",
            )

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        return await _log_and_respond_error(
            request_id=request_id,
            status_code=500,
            error_message="An unexpected error occurred while processing the image.",
            filename=file.filename,
            processing_time=time.time() - start_time
        )


# ---------------------------------------------------------------------------
# Exception handlers
#
# Early rejections (auth → 401/403, request validation → 422, unknown path →
# 404, etc.) are raised before the route handler runs, so without these
# handlers FastAPI renders them as `{"detail": ...}` and they never reach
# OcrKtpLog. These re-shape such responses into the same envelope the route
# uses and log every rejected request — full parity with the sibling service,
# which logs all paths including early rejections.
# ---------------------------------------------------------------------------


async def _log_early_rejection(
    request: Request, status_code: int, error_message: str
) -> None:
    """Best-effort log of a request rejected before the route body ran.

    insert_log self-gates on log_to_database and on the DB being initialised,
    so this is a no-op when logging is disabled.
    """
    try:
        await insert_log(
            request_id=request_id_ctx.get(),
            response_code=status_code,
            payload={"path": request.url.path, "method": request.method},
            error_message=error_message,
            result="",
            processing_time=0.0,
        )
    except Exception as exc:  # never let logging mask the real response
        logger.error(f"Failed to log early rejection: {exc}")


def register_exception_handlers(app: FastAPI) -> None:
    """Register envelope-shaped handlers for auth/validation/HTTP errors.

    Called from main.py for the production app. Tests can call this on an
    ad-hoc app to exercise the unified error contract.
    """

    @app.exception_handler(StarletteHTTPException)
    async def _http_exception_handler(request: Request, exc: StarletteHTTPException):
        message = exc.detail if isinstance(exc.detail, str) else "Request rejected"
        await _log_early_rejection(request, exc.status_code, message)
        return _envelope(
            exc.status_code,
            request_id_ctx.get(),
            message=message,
            error_code=_ERROR_CODES.get(exc.status_code, "ERROR"),
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_exception_handler(request: Request, exc: RequestValidationError):
        await _log_early_rejection(request, 422, "Request validation failed")
        return _envelope(
            422,
            request_id_ctx.get(),
            message="Request validation failed",
            error_code="VALIDATION_ERROR",
            errors=jsonable_encoder(exc.errors()),
        )
