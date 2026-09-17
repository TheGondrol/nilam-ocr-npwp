"""
FastAPI routes for Tamper Detection API
All endpoints with async handlers
"""

import time
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, FastAPI, File, Request, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from pydantic import BaseModel

from src.api.dependencies import verify_api_key
from src.core.config import config
from src.core.logging import get_logger, request_id_ctx
from src.services.database_service import insert_log
from src.services.threshold_provider import get_provider
from src.services.tamper_detection import get_tamper_service

logger = get_logger(__name__)

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


# Create router
router = APIRouter()


# Pydantic models
class PredictionResponse(BaseModel):
    """Response model for prediction endpoint"""
    filename: str
    prediction: str
    confidence: float
    probabilities: dict
    threshold: float
    timestamp: str


class BatchPredictionResponse(BaseModel):
    """Response model for batch prediction endpoint"""
    results: List[PredictionResponse]
    total_processed: int
    timestamp: str


class HealthResponse(BaseModel):
    """Response model for full health check"""
    status: str
    model_loaded: bool
    device: str
    version: str
    checks: Optional[dict] = None


class LivenessResponse(BaseModel):
    """Response model for liveness probe"""
    status: str
    version: str


class ReadinessResponse(BaseModel):
    """Response model for readiness probe"""
    status: str
    checks: Optional[dict] = None


# ---------------------------------------------------------------------------
# Health helpers
# ---------------------------------------------------------------------------


def _check_model() -> dict:
    """Check model status."""
    try:
        service = get_tamper_service()
        if service.is_ready():
            return {"status": "up", "device": str(service.device) if service.device else "cpu"}
        return {"status": "down", "reason": "model not loaded"}
    except Exception as exc:
        return {"status": "down", "reason": str(exc)}


async def _check_database() -> dict:
    """Check database connectivity."""
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


# Routes
@router.get("/", response_model=dict)
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Document Tamper Detection API",
        "description": "Detects tampered documents (text modifications) vs authentic documents",
        "version": config.app.version,
        "endpoints": {
            "health": "/health",
            "liveness": "/health/live",
            "readiness": "/health/ready",
            "predict": "/v1/ocr_tamper",
            "docs": "/docs",
        },
    }


@router.get("/health/live", response_model=LivenessResponse)
async def liveness():
    """Liveness probe — always returns 200 if the process is alive."""
    return LivenessResponse(
        status="alive",
        version=config.app.version,
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
        version=config.app.version,
        checks=checks,
    )

    if code == 200:
        return body
    return JSONResponse(status_code=code, content=body.model_dump())


@router.post("/v1/ocr_tamper")
async def predict(file: UploadFile = File(...), _: str = Depends(verify_api_key)):
    """
    Predict whether a document image is tampered or authentic
    
    Args:
        file: Image file (JPEG, PNG, etc.)
    
    Returns:
        PredictionResponse with prediction result and confidence
        - authentic: Original, unmodified document
        - tampered: Document with text modifications
    """
    request_id = request_id_ctx.get() or "unknown"
    start_time = time.time()
    filename = file.filename or "unknown"
    response_code = 500
    error_message = ""
    result_data = None

    try:
        service = get_tamper_service()
        max_mb = config.image.max_size_mb
        max_bytes = max_mb * 1024 * 1024
        # Reject oversized uploads early, using the declared size, before
        # reading the whole file into memory.
        declared_size = getattr(file, "size", None)
        if declared_size is not None and declared_size > max_bytes:
            response_code = 413
            error_message = f"File too large: {declared_size} bytes (declared, max {max_mb} MB)"
            logger.warning(f"File too large (declared): {declared_size} bytes")
            return _envelope(413, request_id, message=f"File too large. Maximum size is {max_mb} MB.", error_code="FILE_TOO_LARGE")
        image_bytes = await file.read()
        if len(image_bytes) > max_bytes:
            response_code = 413
            error_message = f"File too large: {len(image_bytes)} bytes (max {max_mb} MB)"
            logger.warning(f"File too large: {len(image_bytes)} bytes")
            return _envelope(413, request_id, message=f"File too large. Maximum size is {max_mb} MB.", error_code="FILE_TOO_LARGE")
        result = await service.predict_from_bytes(image_bytes)

        # Threshold-gated decision. The checkpoint's `id2label` may use
        # either {"good", "tamper"} (current saved model) or
        # {"authentic", "tampered"} (older docs/tests). Identify the
        # suspicious-class label by name match so the gate works for both.
        threshold = get_provider().get("threshold")
        probabilities = result["probabilities"]
        SUSPICIOUS = {"tamper", "tampered"}
        SAFE = {"good", "authentic"}
        suspicious_label = next(
            (k for k in probabilities if k.lower() in SUSPICIOUS), None
        )
        safe_label = next(
            (k for k in probabilities if k.lower() in SAFE), None
        )

        if suspicious_label is None or safe_label is None:
            # Unknown label scheme — fall back to model's argmax decision.
            logger.warning(
                f"Unknown id2label keys {list(probabilities.keys())}; "
                "falling back to argmax. Threshold gating disabled for this request."
            )
            prediction = result["predicted_class"]
            confidence = float(result["confidence"])
            prob_suspicious = float("nan")
        else:
            prob_suspicious = float(probabilities[suspicious_label])
            if prob_suspicious >= threshold:
                prediction = suspicious_label
                confidence = prob_suspicious
            else:
                prediction = safe_label
                confidence = float(probabilities[safe_label])

        logger.info(
            f"Prediction: {prediction}, "
            f"Confidence: {confidence:.4f}, "
            f"Prob(suspicious): {prob_suspicious:.4f}, "
            f"Threshold: {threshold:.4f}"
        )

        response_code = 200
        response_dict = {
            "filename": filename,
            "prediction": prediction,
            "confidence": confidence,
            "probabilities": probabilities,
            "threshold": threshold,
            "timestamp": datetime.now().isoformat()
        }
        result_data = response_dict
        return _envelope(200, request_id, data=response_dict)

    except ValueError as e:
        # Invalid/unreadable file content (e.g. an Excel file uploaded instead
        # of an image) — this is a client error, not a server fault.
        response_code = 400
        error_message = f"Invalid file for {filename}: {str(e)}"
        logger.warning(f"Invalid file for {filename}: {str(e)} with request id: {request_id}")
        return _envelope(400, request_id, message=f"Invalid file: {str(e)}", error_code="INVALID_FILE")

    except Exception as e:
        response_code = 500
        error_message = f"Prediction error for {filename}: {str(e)}"
        logger.error(f"Prediction error for {filename}: {str(e)} with request id: {request_id}")
        return _envelope(500, request_id, message=f"Prediction failed: {str(e)}", error_code="INTERNAL_ERROR")

    finally:
        await insert_log(
            request_id=request_id,
            response_code=response_code,
            payload={"file_name": filename},
            error_message=error_message,
            result=result_data,
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