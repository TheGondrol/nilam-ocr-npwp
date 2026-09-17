"""
API Routes for OCR Graycopy Detection Service
"""

import asyncio
import io
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, FastAPI, File, Request, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from PIL import Image

from src.api.dependencies import verify_api_key
from src.core.config import get_config
from src.core.logging import get_logger, request_id_ctx
from src.schemas.api_schema import HealthResponse, LivenessResponse, PredictionResponse, ReadinessResponse
from src.services.database_service import insert_log
from src.services.graycopy_detection import GraycopyDetectionService
from src.services.threshold_provider import get_provider

logger = get_logger()

# Get config at module level
config = get_config()



# Global service instance (will be initialized in main.py)
_detection_service: Optional[GraycopyDetectionService] = None

# Thread pool executor for CPU-bound operations
_executor: Optional[ThreadPoolExecutor] = None


def get_executor() -> ThreadPoolExecutor:
    """Get or create thread pool executor"""
    global _executor
    if _executor is None:
        _executor = ThreadPoolExecutor(max_workers=config.server.thread_pool_workers)
    return _executor


def shutdown_executor() -> None:
    """Shutdown thread pool executor gracefully"""
    global _executor
    if _executor is not None:
        _executor.shutdown(wait=True)
        _executor = None
        logger.info("Executor shutdown complete")


def set_detection_service(service: Optional[GraycopyDetectionService]) -> None:
    """Set the detection service instance"""
    global _detection_service
    _detection_service = service


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
    request_id: Optional[str],
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


# Create router
router = APIRouter()


def _process_image_sync(image_bytes: bytes) -> Image.Image:
    """
    Synchronous image processing - runs in thread pool.
    
    Args:
        image_bytes: Image data in bytes
    
    Returns:
        PIL Image in RGB mode
    
    Raises:
        ValueError: If image cannot be opened or processed
    """
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return image
    except Exception as e:
        raise ValueError(f"Invalid image data: {str(e)}")


@router.get("/", response_model=dict)
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Graycopy Detection API",
        "description": "Detects photocopied/graycopy ID documents (positive) vs original documents (negative)",
        "version": "1.0.0",
        "preprocessing": "Smart preprocessing - auto-crops if needed (target: 224x224)",
        "endpoints": {
            "health": "/health",
            "liveness": "/health/live",
            "readiness": "/health/ready",
            "predict": "/predict",
            "docs": "/docs"
        }
    }


# ---------------------------------------------------------------------------
# Health check helpers
# ---------------------------------------------------------------------------

async def _check_database() -> dict:
    """Check if the async database engine is initialised and can execute a query."""
    try:
        from src.services.database_service import _async_session_factory
        from sqlalchemy import text

        if _async_session_factory is None:
            return {"status": "down", "reason": "not initialised"}
        async with _async_session_factory() as session:
            await session.execute(text("SELECT 1"))
        return {"status": "up"}
    except Exception as exc:
        return {"status": "down", "reason": str(exc)}


def _check_model() -> dict:
    """Check if the detection model is loaded and ready for inference."""
    try:
        if _detection_service is not None and _detection_service.is_ready():
            return {"status": "up", "device": _detection_service.get_device_info()}
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
    return LivenessResponse(status="alive", version="1.0.0")


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

    model_ok = model_check["status"] == "up"
    db_ok = db_check["status"] == "up"

    if model_ok and db_ok:
        status = "healthy"
    elif model_ok:
        status = "degraded"
    else:
        status = "unhealthy"

    model_loaded = _detection_service is not None and _detection_service.is_ready()
    device = _detection_service.get_device_info() if _detection_service else "unknown"

    return HealthResponse(
        status=status,
        model_loaded=model_loaded,
        device=device,
        timestamp=datetime.now().isoformat(),
        checks={
            "model": model_check,
            "database": db_check,
        },
    )


@router.post("/v1/ocr_graycopy", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(...),
    _: str = Depends(verify_api_key)
):
    """
    Predict whether an ID document image is a photocopy/graycopy or original.
    
    Smart preprocessing: Automatically detects if image needs cropping
    - If image is already 224x224: just resize and normalize
    - If image is not 224x224: apply crop preprocessing (Resize(512) + CenterCrop(224))
    
    Args:
        file: Image file (JPEG, PNG, etc.)
    
    Returns:
        PredictionResponse with prediction result and confidence
        - GRAYCOPY: Photocopied or scanned document (class 0) - suspicious
        - ORIGINAL: Original legitimate document (class 1)
    
    Example:
        curl -X POST "http://localhost:8020/v1/ocr_graycopy" \\
             -H "accept: application/json" \\
             -H "Content-Type: multipart/form-data" \\
             -F "file=@/path/to/image.jpg"
    """
    request_id = request_id_ctx.get()
    start_time = time.time()
    filename = file.filename or "unknown"

    logger.info(f"Processing image {filename}")

    contents: Optional[bytes] = None
    image: Optional[Image.Image] = None

    # Logging variables - tracked throughout execution
    response_code = 500
    error_message = ""
    log_result: Optional[dict] = None
    envelope_response = None

    try:
        # Check model readiness
        if _detection_service is None or not _detection_service.is_ready():
            response_code = 503
            error_message = "Model not loaded"
            envelope_response = _envelope(503, request_id, message="Model not loaded", error_code="MODEL_NOT_LOADED")
            return envelope_response
        service = _detection_service

        # Reject oversized uploads early, using the declared size, before
        # reading the whole file into memory.
        max_mb = config.server.max_file_size_mb
        max_bytes = max_mb * 1024 * 1024
        declared_size = getattr(file, "size", None)
        if declared_size is not None and declared_size > max_bytes:
            response_code = 413
            error_message = f"File too large. Maximum size is {max_mb} MB."
            logger.warning(f"File too large (declared): {declared_size} bytes")
            envelope_response = _envelope(413, request_id, message=f"File too large. Maximum size is {max_mb} MB.", error_code="FILE_TOO_LARGE")
            return envelope_response

        # Read image bytes
        contents = await file.read()

        # Authoritative check, if the declared size was absent/wrong.
        if len(contents) > max_bytes:
            response_code = 413
            error_message = f"File too large. Maximum size is {max_mb} MB."
            logger.warning(f"File too large: {len(contents)} bytes")
            envelope_response = _envelope(413, request_id, message=f"File too large. Maximum size is {max_mb} MB.", error_code="FILE_TOO_LARGE")
            return envelope_response

        # Process image in thread pool
        loop = asyncio.get_running_loop()
        executor = get_executor()

        try:
            image = await loop.run_in_executor(executor, _process_image_sync, contents)
        except ValueError as e:
            response_code = 400
            error_message = str(e)
            logger.error(f"Failed to process image {filename}: {str(e)}")
            envelope_response = _envelope(400, request_id, message=str(e), error_code="IMAGE_PROCESSING_FAILED")
            return envelope_response

        # Resolve threshold from the provider (refreshed periodically from DB)
        threshold = get_provider().get("threshold")

        # Run prediction in thread pool
        try:
            result = await loop.run_in_executor(
                executor, service.predict, image, filename, threshold
            )
        except Exception as e:
            response_code = 500
            error_message = f"Prediction failed: {str(e)}"
            logger.error(f"Prediction failed for {filename}: {str(e)}")
            envelope_response = _envelope(500, request_id, message=f"Prediction failed: {str(e)}", error_code="PREDICTION_FAILED")
            return envelope_response

        # Success - build response and use Pydantic serialization for logging
        response_code = 200
        response = PredictionResponse(
            filename=filename,
            prediction=result["prediction"],
            confidence=result["confidence"],
            probability_original=result["probability_original"],
            probability_graycopy=result["probability_graycopy"],
            threshold=threshold,
            timestamp=datetime.now().isoformat()
        )
        log_result = response.model_dump()

        logger.info(f"Success processing image {filename} in {time.time() - start_time:.3f}s")

        envelope_response = _envelope(200, request_id, data=response.model_dump())
        return envelope_response

    except Exception as e:
        response_code = 500
        error_message = f"Unexpected error: {str(e)}"
        logger.error(f"Unexpected error processing {filename}: {str(e)}", exc_info=True)
        envelope_response = _envelope(500, request_id, message="Internal server error", error_code="INTERNAL_ERROR")
        return envelope_response
    finally:
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Single insert_log call for all cases
        await insert_log(
            request_id=request_id or "unknown",
            response_code=response_code,
            payload={"filename": filename},
            error_message=error_message,
            result=log_result,
            processing_time=processing_time
        )

        # Explicit memory cleanup
        if image is not None:
            del image
        if contents is not None:
            del contents


# ---------------------------------------------------------------------------
# Exception handlers
#
# Early rejections (auth → 401/403, request validation → 422, unknown path →
# 404, etc.) are raised before the predict handler runs, so without these
# handlers FastAPI renders them as `{"detail": ...}` and they never reach
# OcrKtpLog. These re-shape such responses into the same envelope the route
# uses and log every rejected request — full parity with the sibling service,
# which logs all paths including early rejections. The generic Exception
# handler also returns the envelope so every error response shares one shape.
# ---------------------------------------------------------------------------


async def _log_early_rejection(
    request: Request, status_code: int, error_message: str
) -> None:
    """Best-effort log of a request rejected before the route body ran.

    insert_log self-gates on config.logging.log_to_database and on the DB being
    initialised, so this is a no-op when logging is disabled.
    """
    try:
        await insert_log(
            request_id=request_id_ctx.get() or "unknown",
            response_code=status_code,
            payload={"path": request.url.path, "method": request.method},
            error_message=error_message,
            result=None,
            processing_time=0.0,
        )
    except Exception as exc:  # never let logging mask the real response
        logger.error(f"Failed to log early rejection: {exc}")


def register_exception_handlers(app: FastAPI) -> None:
    """Register envelope-shaped handlers for auth/validation/HTTP/unhandled errors.

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

    @app.exception_handler(Exception)
    async def _unhandled_exception_handler(request: Request, exc: Exception):
        # Catch-all for truly unhandled errors (the predict route handles its
        # own 500s). Log full detail server-side, return a sanitized envelope.
        logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
        return _envelope(
            500,
            request_id_ctx.get(),
            message="Internal server error",
            error_code="INTERNAL_ERROR",
        )