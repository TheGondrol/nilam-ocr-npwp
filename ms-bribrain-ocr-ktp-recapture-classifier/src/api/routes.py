"""
API Routes for Screen Recapture Detection
"""

from typing import Optional

from fastapi import APIRouter, Depends, FastAPI, File, Request, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from datetime import datetime

from ..api.dependencies import verify_api_key
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor

from ..schemas.api_schema import PredictionResponse, HealthResponse, LivenessResponse, ReadinessResponse
from ..core.config import config
from ..core.logging import logger, request_id_ctx
from ..services.database_services import insert_log
from ..services.recapture_service import process_and_predict_sync, get_model, get_device as get_service_device
from ..services.threshold_provider import get_provider

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

# Create router without prefix - using explicit paths
router = APIRouter(tags=["OCR Recapture Detection"])

# Cached config values (avoid lookups per request)
_thread_pool_workers = config.get('server.thread_pool_workers', 4)

_executor = ThreadPoolExecutor(max_workers=_thread_pool_workers)

@router.get("/", response_model=dict)
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Screen Recapture Detection API",
        "description": "Detects recaptured screen photos vs original captures",
        "version": config.get('api.version', '1.0.0'),
        "endpoints": {
            "health": "/health",
            "liveness": "/health/live",
            "readiness": "/health/ready",
            "predict": "/v1/ocr_recapture",
            "docs": "/docs",
        },
    }


# ---------------------------------------------------------------------------
# Health helpers
# ---------------------------------------------------------------------------


def _check_model() -> dict:
    """Check model status."""
    try:
        model = get_model()
        if model is not None:
            device = get_service_device()
            return {"status": "up", "device": str(device) if device else "cpu"}
        return {"status": "down", "reason": "model not loaded"}
    except Exception as exc:
        return {"status": "down", "reason": str(exc)}


async def _check_database() -> dict:
    """Check database connectivity."""
    try:
        from ..services.database_services import _async_session_factory
        from sqlalchemy import text

        if _async_session_factory is None:
            return {"status": "down", "reason": "not initialised"}
        async with _async_session_factory() as session:
            await session.execute(text("SELECT 1"))
        return {"status": "up"}
    except Exception as exc:
        return {"status": "down", "reason": str(exc)}


# ---------------------------------------------------------------------------
# Health endpoints
# ---------------------------------------------------------------------------


@router.get("/health/live", response_model=LivenessResponse)
async def liveness():
    """Liveness probe — always returns 200 if the process is alive."""
    return LivenessResponse(
        status="alive",
        version=config.get('api.version', '1.0.0'),
    )


@router.get("/health/ready")
async def readiness():
    """Readiness probe — 200 if model loaded and DB reachable, 503 otherwise."""
    model = _check_model()
    db = await _check_database()
    checks = {"model": model, "database": db}

    if model["status"] == "up" and db["status"] == "up":
        return {"status": "ready", "checks": checks}

    return JSONResponse(
        status_code=503,
        content={"status": "not_ready", "checks": checks},
    )


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Full health check with component details."""
    model = _check_model()
    db = await _check_database()
    checks = {"model": model, "database": db}

    model_up = model["status"] == "up"
    db_up = db["status"] == "up"

    if model_up and db_up:
        status_val = "healthy"
    elif model_up and not db_up:
        status_val = "degraded"
    else:
        status_val = "unhealthy"

    code = 200 if status_val in ("healthy", "degraded") else 503

    body = HealthResponse(
        status=status_val,
        model_loaded=model_up,
        device=model.get("device", "unknown"),
        version=config.get('api.version', '1.0.0'),
        checks=checks,
    )

    if code == 200:
        return body
    return JSONResponse(status_code=code, content=body.model_dump())


@router.post("/v1/ocr_recapture", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...), _: str = Depends(verify_api_key)):
    """
    Predict whether an image is a recaptured screen photo or original capture.
    
    Args:
        file: Image file (JPEG, PNG, etc.)
    
    Returns:
        PredictionResponse with prediction result and confidence
        - ORIGINAL: Original capture (class 0 / `good` folder - normal/legitimate)
        - RECAPTURED: Screen recaptured photo (class 1 / `recaptured` folder - suspicious)
    """
    request_id = request_id_ctx.get()
    start_time = time.time()
    
    # Initialize tracking variables for finally block
    response_code = 500
    error_message = ""
    result: PredictionResponse | None = None
    
    try:
        # Validate model is loaded
        model = get_model()
        if model is None:
            response_code = 503
            error_message = "Model not loaded"
            logger.error("Model not loaded")
            return _envelope(503, request_id, message="Model not loaded", error_code="MODEL_NOT_LOADED")
        
        # Validate file type
        if not file.content_type or not file.content_type.startswith("image/"):
            response_code = 400
            error_message = f"Invalid file type: {file.content_type}"
            logger.warning(f"Invalid file type: {file.content_type}")
            return _envelope(400, request_id, message=f"Invalid file type: {file.content_type}. Please upload an image.", error_code="INVALID_FILE_TYPE")
        
        # Validate file size — reject early using the declared size, before
        # reading the whole file into memory.
        max_mb = config.max_image_size_mb
        max_bytes = max_mb * 1024 * 1024
        declared_size = getattr(file, "size", None)
        if declared_size is not None and declared_size > max_bytes:
            response_code = 413
            error_message = f"File too large: {declared_size} bytes (declared)"
            logger.warning(f"File too large (declared): {declared_size} bytes")
            return _envelope(413, request_id, message=f"File too large. Maximum size is {max_mb} MB.", error_code="FILE_TOO_LARGE")

        # Read image bytes and process in single executor call
        contents = await file.read()

        # Authoritative check, if the declared size was absent/wrong.
        if len(contents) > max_bytes:
            response_code = 413
            error_message = f"File too large: {len(contents)} bytes"
            logger.warning(f"File too large: {len(contents)} bytes")
            return _envelope(413, request_id, message=f"File too large. Maximum size is {max_mb} MB.", error_code="FILE_TOO_LARGE")

        loop = asyncio.get_running_loop()
        
        # Combined processing: load image, resize, and predict in one executor call
        # This reduces overhead from 3 separate executor calls (~30-50ms savings)
        inference_start = time.time()
        try:
            predicted_class, prob_recaptured, confidence, original_size = await loop.run_in_executor(
                _executor, process_and_predict_sync, contents
            )
            inference_time = time.time() - inference_start
            logger.info(f"Received image: {file.filename}, original size: {original_size}")
            logger.info(f"Inference completed in {inference_time:.2f} seconds")
        except Exception as e:
            response_code = 400
            error_message = f"Failed to process image: {str(e)}"
            logger.error(f"Failed to process image {file.filename}: {str(e)}")
            return _envelope(400, request_id, message=f"Failed to process image: {str(e)}", error_code="IMAGE_PROCESSING_FAILED")
        
        # Threshold-gated decision: flag as RECAPTURED when the recapture
        # probability meets or exceeds the configured threshold. The
        # checkpoint's class-to-label mapping is 0=ORIGINAL, 1=RECAPTURED;
        # `prob_recaptured` already reflects that. The threshold is
        # refreshed live from the management DB.
        threshold = get_provider().get("threshold")
        if prob_recaptured >= threshold:
            prediction = "RECAPTURED"
            confidence = prob_recaptured
        else:
            prediction = "ORIGINAL"
            confidence = 1.0 - prob_recaptured

        logger.info(
            f"Prediction: {prediction}, "
            f"Confidence: {confidence:.4f}, "
            f"Prob(recaptured): {prob_recaptured:.4f}, "
            f"Threshold: {threshold:.4f}"
        )
        logger.info(f"Total processing time: {time.time() - start_time:.2f} seconds")

        # Success case
        response_code = 200
        error_message = ""

        result = PredictionResponse(
            filename=file.filename or "",
            prediction=prediction,
            confidence=round(confidence, 4),
            probability_recaptured=round(prob_recaptured, 4),
            threshold=threshold,
            timestamp=datetime.now().isoformat()
        )

        return _envelope(200, request_id, data=result.model_dump())

    except Exception as e:
        response_code = 500
        error_message = f"Prediction error: {str(e)}"
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        return _envelope(500, request_id, message=f"Prediction failed: {str(e)}", error_code="INTERNAL_ERROR")
    
    finally:
        # Single logging point - guaranteed to run
        await insert_log(
            request_id=request_id,
            response_code=response_code,
            payload={"file_name": file.filename},
            error_message=error_message,
            result=result.model_dump() if result else None,
            processing_time=time.time() - start_time
        )


# ---------------------------------------------------------------------------
# Exception handlers
#
# Early rejections (auth → 401/403, request validation → 422, unknown path →
# 404, etc.) are raised before the predict handler runs, so without these
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
            result=None,
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