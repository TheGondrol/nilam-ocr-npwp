"""API routes for quality classification."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from fastapi import APIRouter, Depends, FastAPI, File, Form, Request, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.api.dependencies import verify_api_key
from src.core.config import config
from src.core.logging import request_id_ctx
from src.schemas.api_schema import ClassificationResponse, Crop, HealthResponse, LivenessResponse, ReadinessResponse
from src.services.database_services import insert_log
from src.services.quality_service import (
    ImageDecodeError,
    get_device,
    get_model,
    process_and_classify_sync,
)

logger = logging.getLogger(__name__)

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

# Thread pool for CPU-bound ML operations
_executor: Optional[ThreadPoolExecutor] = None

# API Router
router = APIRouter(tags=["KTP Quality Classification"])


def set_executor(executor: ThreadPoolExecutor) -> None:
    """Set global thread pool executor."""
    global _executor
    _executor = executor


@router.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": config.get("api.title", "KTP Image Quality Classifier API"),
        "description": config.get(
            "api.description",
            "Classifies KTP image quality by analyzing OCR text crop quality",
        ),
        "version": config.get("api.version", "1.0.0"),
        "endpoints": {
            "health": "/health",
            "liveness": "/health/live",
            "readiness": "/health/ready",
            "filter": "/filter (POST)",
        },
    }


# ---------------------------------------------------------------------------
# Health check helpers
# ---------------------------------------------------------------------------

async def _check_database() -> dict:
    """Check if the async database engine is initialised and can execute a query."""
    try:
        from src.services.database_services import _async_session_factory
        from sqlalchemy import text

        if _async_session_factory is None:
            return {"status": "down", "reason": "not initialised"}
        async with _async_session_factory() as session:
            await session.execute(text("SELECT 1"))
        return {"status": "up"}
    except Exception as exc:
        return {"status": "down", "reason": str(exc)}


def _check_model() -> dict:
    """Check if the ML model is loaded and ready for inference."""
    try:
        model = get_model()
        if model is not None:
            device = get_device()
            return {"status": "up", "device": str(device) if device else "unknown"}
        return {"status": "down", "reason": "model not loaded"}
    except Exception as exc:
        return {"status": "down", "reason": str(exc)}


# ---------------------------------------------------------------------------
# Health endpoints
# ---------------------------------------------------------------------------

@router.get(
    "/health/live",
    response_model=LivenessResponse,
    summary="Liveness probe",
    description="Returns 200 if the process is alive. Use for K8s livenessProbe.",
)
async def liveness():
    """Liveness probe — lightweight, always returns 200 if the process can respond."""
    return LivenessResponse(
        status="alive",
        version=config.get("api.version", "1.0.0"),
    )


@router.get(
    "/health/ready",
    response_model=ReadinessResponse,
    summary="Readiness probe",
    description="Returns 200 if model loaded + DB up. Use for K8s readinessProbe.",
)
async def readiness_check():
    """
    Readiness probe — checks core infrastructure.

    Returns 200 if model is loaded and database is up.
    Returns 503 if either is down.
    """
    model_check = _check_model()
    db_check = await _check_database()

    model_ok = model_check["status"] == "up"
    db_ok = db_check["status"] == "up"
    ready = model_ok and db_ok

    return JSONResponse(
        status_code=200 if ready else 503,
        content={
            "status": "ready" if ready else "not_ready",
            "checks": {
                "model": model_check,
                "database": db_check,
            },
        },
    )


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Full health check",
    description="Returns detailed health status including model, database, and device info.",
)
async def health_check():
    """
    Full health check — model, database, and device info.

    Status logic:
    - healthy:   model loaded and DB up
    - degraded:  model loaded but DB down
    - unhealthy: model not loaded
    """
    model_check = _check_model()
    db_check = await _check_database()

    model = get_model()
    device = get_device()

    model_ok = model_check["status"] == "up"
    db_ok = db_check["status"] == "up"

    if model_ok and db_ok:
        status = "healthy"
    elif model_ok:
        status = "degraded"
    else:
        status = "unhealthy"

    return HealthResponse(
        status=status,
        model_loaded=model is not None,
        device=str(device) if device else "unknown",
        version=config.get("api.version", "1.0.0"),
        checks={
            "model": model_check,
            "database": db_check,
        },
    )


@router.post("/v1/ocr_qualitydl", response_model=ClassificationResponse)
async def filter_quality(
    file: UploadFile = File(..., description="Image file to classify"),
    crops: str = Form(..., description="JSON string of crops list"),
    _: str = Depends(verify_api_key),
):
    """
    Classify image quality based on OCR text crop quality.

    This endpoint:
    1. Receives an image and list of crops (bbox, text, confidence from OCR)
    2. Filters crops to focus on text (width > height)
    3. Extracts and classifies each crop as good/bad
    4. Returns overall image quality (bad if >= threshold bad crops)

    Args:
        file: Uploaded image file
        crops: JSON string containing list of crop objects with bbox, text, confidence

    Returns:
        ClassificationResponse with predictions and overall label

    Raises:
        HTTPException: If processing fails
    """
    request_id = request_id_ctx.get()
    start_time = time.time()
    response_code = 500
    error_message = ""
    result = None
    payload = None

    try:
        # Validate model loaded
        model = get_model()
        if model is None:
            response_code = 503
            error_message = "Model not loaded"
            return _envelope(503, request_id, message="Model not loaded", error_code="MODEL_NOT_LOADED")

        # Validate file type
        if not file.content_type or not file.content_type.startswith("image/"):
            response_code = 400
            error_message = "File must be an image"
            return _envelope(400, request_id, message="File must be an image", error_code="INVALID_FILE_TYPE")

        # Parse crops from JSON string
        try:
            crops_data = json.loads(crops)
            if not isinstance(crops_data, list):
                raise ValueError("Crops must be a list")

            # Validate and convert to Crop objects
            crop_objects = [Crop(**crop_dict) for crop_dict in crops_data]

        except json.JSONDecodeError as e:
            response_code = 400
            logger.error(f"Invalid JSON in crops: {e}")
            error_message = "Invalid JSON in crops"
            return _envelope(400, request_id, message="Invalid JSON in crops. Expected a valid JSON list.", error_code="INVALID_INPUT")
        except Exception as e:
            response_code = 400
            # Log detail for ops; store/return generic — crop data can carry OCR field text (PII).
            logger.error(f"Invalid crop data: {e}")
            error_message = "Invalid crop data"
            return _envelope(400, request_id, message="Invalid crop data.", error_code="INVALID_INPUT")

        logger.info(f"Processing request with {len(crop_objects)} crops")

        # Store payload for logging
        payload = {
            "filename": file.filename,
            "content_type": file.content_type,
            "num_crops": len(crop_objects),
        }

        # Validate file size — reject early using the declared size, before
        # reading the whole file into memory.
        max_mb = config.max_image_size_mb
        max_bytes = max_mb * 1024 * 1024
        declared_size = getattr(file, "size", None)
        if declared_size is not None and declared_size > max_bytes:
            response_code = 413
            error_message = f"File too large: {declared_size} bytes (declared, max {max_mb} MB)"
            logger.warning(error_message)
            return _envelope(
                413,
                request_id,
                message=f"File too large. Maximum size is {max_mb} MB.",
                error_code="FILE_TOO_LARGE",
            )

        # Read file bytes
        contents = await file.read()

        # Authoritative check, if the declared size was absent/wrong.
        if len(contents) > max_bytes:
            response_code = 413
            error_message = f"File too large: {len(contents)} bytes (max {max_mb} MB)"
            logger.warning(error_message)
            return _envelope(
                413,
                request_id,
                message=f"File too large. Maximum size is {max_mb} MB.",
                error_code="FILE_TOO_LARGE",
            )

        # Run classification in thread pool (CPU-bound operation)
        loop = asyncio.get_running_loop()
        classification_result = await loop.run_in_executor(
            _executor,
            process_and_classify_sync,
            contents,
            crop_objects,
        )

        response_code = 200
        result = classification_result.model_dump()

        logger.info(
            f"Classification successful: {classification_result.label} "
            f"({classification_result.num_bad}/{classification_result.num_filtered} bad crops)"
        )

        return _envelope(200, request_id, data=classification_result.model_dump())

    except ImageDecodeError as e:
        response_code = 400
        error_message = f"Invalid image: {e}"
        logger.warning(f"Undecodable image upload for {file.filename}: {e}")
        return _envelope(400, request_id, message=f"Invalid image: {e}", error_code="INVALID_INPUT")
    except Exception as e:
        response_code = 500
        # Detail is logged below; store/return a generic message (str(e) may carry field text).
        error_message = "Classification failed"
        logger.error(f"Classification failed: {e}", exc_info=True)
        return _envelope(500, request_id, message="Classification failed.", error_code="PREDICTION_FAILED")

    finally:
        # Log to database (always executes)
        processing_time = time.time() - start_time

        if config.log_to_database:
            try:
                await insert_log(
                    request_id=request_id,
                    response_code=response_code,
                    payload=payload,
                    error_message=error_message,
                    result=result,
                    processing_time=processing_time,
                )
            except Exception as e:
                logger.error(f"Failed to log to database: {e}")

        logger.info(f"Request completed in {processing_time:.3f}s with code {response_code}")


# ---------------------------------------------------------------------------
# Exception handlers
#
# Early rejections (auth → 401/403, request validation → 422, unknown path →
# 404, etc.) are raised before filter_quality runs, so without these handlers
# FastAPI renders them as `{"detail": ...}` and they never reach
# OcrKtpQualityLog. These re-shape such responses into the same envelope the
# route uses and log every rejected request — full parity with the sibling
# service, which logs all paths including early rejections.
# ---------------------------------------------------------------------------


async def _log_early_rejection(
    request: Request, status_code: int, error_message: str
) -> None:
    """Best-effort log of a request rejected before the route body ran."""
    if not config.log_to_database:
        return
    try:
        await insert_log(
            request_id=request_id_ctx.get(),
            response_code=status_code,
            payload={"path": request.url.path, "method": request.method},
            error_message=error_message,
            result=None,
            processing_time=0.0,
        )
    except Exception as exc:  # never let logging mask the real response
        logger.error(f"Failed to log early rejection: {exc}")


def register_exception_handlers(app: FastAPI) -> None:
    """Register envelope-shaped handlers for auth/validation/HTTP errors.

    Called from main.py for the production app. Tests that build an ad-hoc app
    can call this too to exercise the unified error contract.
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
