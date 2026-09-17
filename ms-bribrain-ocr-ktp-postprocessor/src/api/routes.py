"""API routes for OCR postprocessing."""

import asyncio
import json
import logging
import time
from typing import Optional

from fastapi import APIRouter, Depends, FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.api.dependencies import verify_api_key, require_json_content_type
from src.schemas.api_schema import OCRExtract, LogEntry
from src.services.ocr_processor import mappingnext
from src.services.database_service import insert_log
from src.core.logging import request_id_ctx

logger = logging.getLogger(__name__)

router = APIRouter()

STATUS_DESCS = {
    200: "OK",
    400: "Bad Request",
    401: "Unauthorized",
    403: "Forbidden",
    404: "Not Found",
    405: "Method Not Allowed",
    415: "Unsupported Media Type",
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
    415: "UNSUPPORTED_MEDIA_TYPE",
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


@router.post("/v1/ocr_postprocess", dependencies=[Depends(require_json_content_type)])
async def extract_text_lines(request: Request, payload: OCRExtract, _: str = Depends(verify_api_key)):
    """
    Endpoint untuk ekstraksi data KTP dari gambar menggunakan OCR.

    Args:
        payload (OCRExtract): Payload berisi OCR text data.

    Returns:
        JSONResponse:
            - 200: Jika berhasil, mengembalikan hasil ekstraksi field KTP.
            - 400: Jika file tidak valid, confidence rendah, atau kualitas gambar buruk.
            - 500: Jika terjadi error internal saat proses.
    """
    request_id = request_id_ctx.get()
    start_time = time.time()
    raw_text = None
    parsed_ocr = None
    response_code = 500
    error_message = ""
    result = None
    nik_image_box = None

    try:
        raw_text = payload.ocr_text
        # Log only non-PII metadata: raw_text is KTP OCR JSON (NIK/name/DOB/address) (BUG-26).
        logger.info(f"Received postprocess request ({len(raw_text) if raw_text else 0} chars)")

        # Convert JSON string → actual Python object (list)
        parsed_ocr = json.loads(raw_text)
        logger.debug(f"Parsed OCR data with {len(parsed_ocr)} items")

        # Process the OCR data (run in thread pool to avoid blocking event loop)
        result, nik_image_box = await asyncio.to_thread(mappingnext, parsed_ocr)

        logger.info(f"Extracted {len(result)} fields: {list(result.keys())}")
        logger.info(f"Total processing time: {time.time() - start_time:.3f}s")

        response_code = 200
        return _envelope(200, request_id, data={"ocr_result": result, "nik_image_box": nik_image_box})

    except json.JSONDecodeError as e:
        response_code = 400
        # Log full detail for ops; persist a generic message — str(e) can echo raw
        # OCR field values (NIK, name, DOB, ...) which must not be stored as PII.
        logger.error(f"Failed to parse OCR text: {e}")
        error_message = "Failed to parse OCR text: invalid JSON format"
        return _envelope(400, request_id, message="Invalid OCR text format. Expected a valid JSON list representation.", error_code="INVALID_INPUT")

    except Exception as e:
        response_code = 500
        # Log full detail for ops; persist a generic message — str(e) can echo raw
        # OCR field values (NIK, name, DOB, ...) which must not be stored as PII.
        logger.error(f"Error processing OCR data: {e}", exc_info=True)
        error_message = "Error processing OCR data"
        return _envelope(500, request_id, message="An error occurred while processing the image.", error_code="INTERNAL_ERROR")

    finally:
        await insert_log(LogEntry(
            request_id=request_id,
            response_code=response_code,
            payload=parsed_ocr if parsed_ocr else raw_text,
            error_message=error_message,
            result=result,
            processing_time=time.time() - start_time
        ))


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

    insert_log self-gates on insert_to_database and on the DB being
    initialised, so this is a no-op when logging is disabled.
    """
    try:
        await insert_log(LogEntry(
            request_id=request_id_ctx.get(),
            response_code=status_code,
            payload={"path": request.url.path, "method": request.method},
            error_message=error_message,
            result=None,
            processing_time=0.0,
        ))
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
