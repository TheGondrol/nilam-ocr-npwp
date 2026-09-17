"""
API Routes
Defines all API endpoints for the YOLO detection service
"""

from fastapi import APIRouter, Depends, FastAPI, File, Request, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from PIL import Image
import io
from datetime import datetime
from typing import Optional
import asyncio
from concurrent.futures import ThreadPoolExecutor
from src.services.database_service import insert_log
from .dependencies import verify_api_key
import time
from ..core.config import config as app_config
from ..core.logging import logger, request_id_ctx
from ..core.device import get_optimal_worker_count
from ..services.predictor import predictor
from .schemas import (
    KTPDetectionResponse,
    HealthResponse,
    LivenessResponse,
    ReadinessResponse,
    RootResponse,
    Detection,
    BoundingBox,
)


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


router = APIRouter(tags=["YOLO Detection"])

# Dynamically set worker count based on available CPUs (K8s-aware)
# Use multiplier for I/O-bound tasks (image decoding) to overlap with CPU work

_thread_multiplier = app_config.get("performance.thread_pool_multiplier", 1)
_worker_count = get_optimal_worker_count(for_inference=True) * _thread_multiplier
_executor = ThreadPoolExecutor(max_workers=_worker_count)
logger.info(
    f"ThreadPoolExecutor initialized with {_worker_count} workers (multiplier: {_thread_multiplier})"
)


def _process_image_sync(image_bytes: bytes) -> Image.Image:
    """
    Synchronous image processing - runs in thread pool.

    Args:
        image_bytes: Image data in bytes

    Returns:
        Image
    """
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return image


@router.get("/", response_model=RootResponse)
async def root():
    """Root endpoint"""
    return RootResponse(
        message="KTP Detection API",
        description="Detects ktp and non-ktp objects using YOLO model",
        version="1.0.0",
        endpoints={
            "health": "/health",
            "liveness": "/health/live",
            "readiness": "/health/ready",
            "predict": "/v1/ocr_ktp_detection",
            "docs": "/docs",
        },
    )


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
    """Check if the YOLO model is loaded and ready for inference."""
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
    return LivenessResponse(status="alive", version="1.0.0")


@router.get(
    "/health/ready",
    response_model=ReadinessResponse,
    summary="Readiness probe",
    description="Returns 200 if the service can accept traffic (model loaded + DB up). Use for K8s readinessProbe.",
)
async def readiness():
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
        device_info=predictor.get_device_info(),
        timestamp=datetime.now().isoformat(),
        checks={
            "model": model_check,
            "database": db_check,
        },
    )


@router.post("/v1/ocr_ktp_detection", response_model=KTPDetectionResponse)
async def predict(file: UploadFile = File(...), _: str = Depends(verify_api_key)):
    """
    Detect objects in an image using YOLO model.

    Args:
        file: Image file (JPEG, PNG, etc.)

    Returns:
        PredictionResponse with detection results
    """
    request_id = request_id_ctx.get()
    start_time = time.time()

    # Initialize tracking variables for finally block
    response_code = 500
    error_message = ""
    result: Optional[KTPDetectionResponse] = None
    filename = file.filename or "unknown"

    try:
        # Validate model is loaded
        if not predictor.is_loaded():
            response_code = 503
            error_message = "Model not loaded"
            logger.error("Prediction attempted but model not loaded")
            return _envelope(
                503, request_id,
                message="Model not loaded",
                error_code="MODEL_NOT_LOADED",
            )

        # Validate file type
        if not file.content_type or not file.content_type.startswith("image/"):
            response_code = 400
            error_message = (
                f"Invalid file type: {file.content_type}. Please upload an image."
            )
            logger.warning(f"Invalid file type: {file.content_type}")
            return _envelope(
                400, request_id,
                message=f"Invalid file type: {file.content_type}. Please upload an image.",
                error_code="INVALID_FILE_TYPE",
            )

        # Read and process image
        loop = asyncio.get_running_loop()

        # Reject oversized uploads early, using the declared size, before
        # reading the whole file into memory.
        max_mb = app_config.max_image_size_mb
        max_bytes = max_mb * 1024 * 1024
        declared_size = getattr(file, "size", None)
        if declared_size is not None and declared_size > max_bytes:
            response_code = 413
            error_message = f"File too large: {declared_size} bytes (declared, max {max_mb} MB)"
            logger.warning(error_message)
            return _envelope(
                413, request_id,
                message=f"File too large. Maximum size is {max_mb} MB.",
                error_code="FILE_TOO_LARGE",
            )

        contents = await file.read()

        # Authoritative check, if the declared size was absent/wrong.
        if len(contents) > max_bytes:
            response_code = 413
            error_message = f"File too large: {len(contents)} bytes (max {max_mb} MB)"
            logger.warning(error_message)
            return _envelope(
                413, request_id,
                message=f"File too large. Maximum size is {max_mb} MB.",
                error_code="FILE_TOO_LARGE",
            )

        try:
            image = await loop.run_in_executor(_executor, _process_image_sync, contents)
        except Exception as e:
            response_code = 400
            error_message = f"Failed to process image: {str(e)}"
            logger.error(f"Failed to process image {file.filename}: {str(e)}")
            return _envelope(
                400, request_id,
                message=f"Failed to process image: {str(e)}",
                error_code="IMAGE_PROCESSING_FAILED",
            )

        # Make prediction
        # We run all inference in the thread pool to prevent blocking the asyncio event loop.
        # This ensures the health check endpoint remains responsive even under load.
        try:
            # Use ThreadPoolExecutor for both PyTorch and OpenVINO models
            detection = await loop.run_in_executor(
                _executor, predictor.predict, image, filename
            )
        except Exception as e:
            response_code = 500
            error_message = f"Failed to make prediction: {str(e)}"
            logger.error(
                f"Failed to make prediction for image {file.filename}: {str(e)}"
            )
            return _envelope(
                500, request_id,
                message=f"Failed to make prediction: {str(e)}",
                error_code="PREDICTION_FAILED",
            )

        # Process detections
        detection_list = []
        ktp_count = 0
        non_ktp_count = 0
        for det in detection:
            detection_list.append(
                Detection(
                    class_id=det["class_id"],
                    class_name=det["class_name"],
                    confidence=round(det["confidence"], 4),
                    bbox=BoundingBox(
                        x1=round(det["bbox"]["x1"], 2),
                        y1=round(det["bbox"]["y1"], 2),
                        x2=round(det["bbox"]["x2"], 2),
                        y2=round(det["bbox"]["y2"], 2),
                    ),
                )
            )
            if det["class_name"] == "ktp":
                ktp_count += 1
            elif det["class_name"] == "non-ktp":
                non_ktp_count += 1

        # Decision logic
        if ktp_count == 0 and non_ktp_count == 0:
            detected = False
            status = "no KTP detected"
            reason = "No KTP or non-KTP found in the image"
        elif ktp_count == 0 and non_ktp_count > 0:
            detected = False
            status = "only non-KTP detected"
            reason = "No KTP found, only non-KTP objects detected"
        elif ktp_count == 1 and non_ktp_count == 0:
            detected = True
            status = "OK"
            reason = None
        elif ktp_count > 1 and non_ktp_count == 0:
            detected = True
            status = "multiple KTPs detected"
            reason = "More than one KTP found in the image"
        else:  # ktp_count >= 1 and non_ktp_count >= 1
            detected = True
            status = "KTP and non-KTP detected"
            reason = "Image contains both KTP and non-KTP objects"

        # Success case
        response_code = 200
        error_message = ""

        result = KTPDetectionResponse(
            filename=filename,
            detected=detected,
            num_detected=ktp_count,
            detections=detection_list,
            status=status,
            reason=reason,
            timestamp=datetime.now().isoformat(),
        )

        logger.info(f"Total prediction time: {time.time() - start_time}")

        return _envelope(200, request_id, data=result.model_dump())

    except Exception as e:
        response_code = 500
        error_message = f"Prediction error: {str(e)}"
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        return _envelope(
            500, request_id,
            message=f"Prediction failed: {str(e)}",
            error_code="INTERNAL_ERROR",
        )

    finally:
        # Single logging point - guaranteed to run
        await insert_log(
            request_id=request_id,
            response_code=response_code,
            payload={"filename": filename},
            error_message=error_message,
            result=result.model_dump() if result else None,
            processing_time=time.time() - start_time,
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