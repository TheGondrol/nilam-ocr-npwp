"""
API Routes
Defines all API endpoints for the lamination detection service
"""

from fastapi import APIRouter, Depends, FastAPI, File, UploadFile, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from PIL import Image

from ..api.dependencies import verify_api_key
import time
import io
import asyncio
from datetime import datetime
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor
from ..services.database_service import insert_log

from ..core.logging import logger, request_id_ctx
from ..core.config import config
from ..services.predictor import predictor
from .schemas import PredictionResponse, HealthResponse, LivenessResponse, ReadinessResponse, RootResponse, BatchPredictionResponse


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


router = APIRouter(tags=["Lamination Detection"])
_executor = ThreadPoolExecutor(max_workers=config.thread_pool_workers)

# Maximum file size from config (in bytes)
MAX_FILE_SIZE = config.max_file_size_mb * 1024 * 1024


def _process_image_sync(image_bytes: bytes) -> Image.Image:
    """
    Synchronous image processing - runs in thread pool.

    Args:
        image_bytes: Image data in bytes

    Returns:
        Image
    """
    image = Image.open(io.BytesIO(image_bytes))
    # Only convert if not already in RGB or grayscale mode
    if image.mode not in ('RGB', 'L'):
        image = image.convert("RGB")
    return image


@router.get("/", response_model=RootResponse)
async def root():
    """Root endpoint with API information."""
    return RootResponse(
        message="Document Lamination Detection API",
        description="Detects unlaminated (suspicious) vs laminated (normal) ID documents",
        version="1.0.0",
        endpoints={
            "health": "/health",
            "liveness": "/health/live",
            "readiness": "/health/ready",
            "predict": "/v1/ocr_laminate",
            "docs": "/docs"
        }
    )


# ---------------------------------------------------------------------------
# Health check helpers
# ---------------------------------------------------------------------------

async def _check_database() -> dict:
    """Check if the async database engine is initialised and can execute a query."""
    try:
        from ..services.database_service import _async_session_factory
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
        if predictor.is_loaded():
            return {"status": "up", "device": predictor.get_device_string()}
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
        version="1.0.0",
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
        model_loaded=predictor.is_loaded(),
        device=predictor.get_device_string(),
        version="1.0.0",
        checks={
            "model": model_check,
            "database": db_check,
        },
    )


@router.post("/v1/ocr_laminate", response_model=PredictionResponse)
async def predict(request: Request, file: UploadFile = File(...), _: str = Depends(verify_api_key)):
    """
    Predict whether an ID document is laminated (normal) or unlaminated (suspicious).
    
    Args:
        file: Image file (JPEG, PNG, etc.)
    
    Returns:
        PredictionResponse with prediction result
        - LAMINATED: Normal/legitimate document
        - UNLAMINATED: Suspicious/potentially fraudulent document
    """
    request_id = request_id_ctx.get()
    start_time = time.time()
    logger.info(f"Received image: {file.filename}, original size: {file.size}")
    
    # Initialize tracking variables for logging
    response_code = 500
    error_message = ""
    result: PredictionResponse | None = None
    
    try:
        # Validate model is loaded
        if not predictor.is_loaded():
            logger.error("Prediction attempted but model not loaded")
            response_code = 503
            error_message = "Model not loaded"
            return _envelope(503, request_id, message="Model not loaded", error_code="MODEL_NOT_LOADED")

        # Validate file type
        if not file.content_type or not file.content_type.startswith("image/"):
            logger.warning(f"Invalid file type: {file.content_type}")
            response_code = 400
            error_message = f"Invalid file type: {file.content_type}. Please upload an image."
            return _envelope(400, request_id, message=error_message, error_code="INVALID_FILE_TYPE")

        # Read and process image
        loop = asyncio.get_running_loop()

        # Reject oversized uploads early, using the declared size, before
        # reading the whole file into memory.
        declared_size = getattr(file, "size", None)
        if declared_size is not None and declared_size > MAX_FILE_SIZE:
            logger.warning(f"File too large (declared): {declared_size} bytes")
            response_code = 413
            error_message = f"File too large. Maximum size is {MAX_FILE_SIZE // (1024*1024)}MB"
            return _envelope(413, request_id, message=error_message, error_code="FILE_TOO_LARGE")

        contents = await file.read()

        # Validate file size (authoritative, if the declared size was absent/wrong)
        if len(contents) > MAX_FILE_SIZE:
            logger.warning(f"File too large: {len(contents)} bytes")
            response_code = 413
            error_message = f"File too large. Maximum size is {MAX_FILE_SIZE // (1024*1024)}MB"
            return _envelope(413, request_id, message=error_message, error_code="FILE_TOO_LARGE")

        try:
            image = await loop.run_in_executor(
                _executor, _process_image_sync, contents
            )
        except Exception as e:
            logger.error(f"Failed to process image {file.filename}: {str(e)}")
            response_code = 400
            error_message = f"Failed to process image: {str(e)}"
            return _envelope(400, request_id, message=error_message, error_code="IMAGE_PROCESSING_FAILED")

        # Make prediction. Offload to the thread pool: predictor.predict runs a
        # full torch forward pass (+ cuda.synchronize), so calling it inline
        # would block the asyncio event loop for every concurrent request
        # (BUG-02). Mirror the decode step above and the sibling services.
        prediction, prob = await loop.run_in_executor(
            _executor, predictor.predict, image, file.filename or "unknown"
        )

        # Success
        response_code = 200
        result = PredictionResponse(
            filename=file.filename or "unknown",
            prediction=prediction,
            prob=round(prob, 4),
            threshold=predictor.threshold,
            timestamp=datetime.now().isoformat()
        )

        logger.info(f"Success prediction: {prediction}, Prob(unlaminated): {prob}")
        logger.info(f"Total processing time: {time.time() - start_time}")
        return _envelope(200, request_id, data=result.model_dump())

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        response_code = 500
        error_message = f"Prediction failed: {str(e)}"
        return _envelope(500, request_id, message=error_message, error_code="PREDICTION_FAILED")
    finally:
        await insert_log(
            request_id=request_id,
            response_code=response_code,
            payload={"filename": file.filename} if file.filename else None,
            error_message=error_message,
            result=result.model_dump() if result is not None else None,
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

    insert_log self-gates on config.log_to_database and on the DB being
    initialised, so this is a no-op when logging is disabled.
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