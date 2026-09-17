"""
FastAPI application entry point.
OCR Quality Service.
"""
from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from contextlib import asynccontextmanager

from src.core.config import get_config
from src.core.logging import setup_logging, get_logger, request_id_ctx

# Setup logging early, before any other module-level loggers emit records
setup_logging()

from src.core.device import get_device_info
from src.api.routes import router as api_router, _envelope
from src.middleware.add_requestid import RequestIdMiddleware
from src.schemas.api_schema import HealthResponse, LivenessResponse, ReadinessResponse
from src.services.database_service import init_engine, dispose_engine, insert_log
from src.services.threshold_provider import init_provider, get_provider


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    Handles startup and shutdown events.
    """
    # Startup
    config = get_config()
    logger = get_logger(__name__)
    logger.info("=" * 60)
    logger.info(f"Starting {config.app.name} v{config.app.version}")
    logger.info("=" * 60)

    # Validate configuration before startup
    from src.core.validation import validate_startup_configuration
    validate_startup_configuration()

    # Log device information
    device_info = get_device_info()
    logger.info(f"Device configuration: {device_info}")

    logger.info(f"Server will run on {config.server.host}:{config.server.port}")
    logger.info("Application startup complete")

    # Initialize async database engine
    await init_engine()

    # Initialize threshold provider (refreshes from management DB hourly)
    await init_provider(
        service_name="dgc_irl",
        defaults={
            "blur_threshold": float(config.quality.blur.threshold),
            "confidence_threshold_median": float(config.quality.confidence.threshold_median),
            "glare_min_text_confidence": float(config.quality.glare.min_text_confidence),
            "glare_affected_percentage_threshold": float(
                config.quality.glare.affected_percentage_threshold
            ),
            "rotation_min_face_proportion": float(config.quality.rotation.min_face_proportion),
            "rotation_face_detection_confidence": float(
                config.quality.rotation.face_detection_confidence
            ),
        },
    ).initialize()

    yield

    # Shutdown
    await get_provider().shutdown()
    await dispose_engine()
    logger.info("Shutting down application")
    logger.info("=" * 60)


# Create FastAPI application
config = get_config()
app = FastAPI(
    title=config.app.name,
    version=config.app.version,
    debug=config.app.debug,
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,  # type: ignore[arg-type]
    allow_origins=config.cors.allow_origins,
    allow_credentials=config.cors.allow_credentials and "*" not in config.cors.allow_origins,  # BUG-21: no wildcard + credentials
    allow_methods=config.cors.allow_methods,
    allow_headers=config.cors.allow_headers,
)

app.add_middleware(RequestIdMiddleware)  # type: ignore[arg-type]

# Add Elastic APM middleware (if enabled)
_apm_logger = get_logger(__name__)
if config.elastic_apm.enabled and config.elastic_apm.server_url:
    from elasticapm.contrib.starlette import ElasticAPM, make_apm_client
    _apm_cfg = {
        "SERVICE_NAME": config.elastic_apm.service_name,
        "SERVER_URL": config.elastic_apm.server_url,
        "ENVIRONMENT": config.elastic_apm.environment,
        "SECRET_TOKEN": config.elastic_apm.secret_token,
        "VERIFY_SERVER_CERT": config.elastic_apm.verify_server_cert,
    }
    apm = make_apm_client(_apm_cfg)
    app.add_middleware(ElasticAPM, client=apm)  # type: ignore[arg-type]
    _apm_logger.info(
        f"Elastic APM: ENABLED (Server: {_apm_cfg['SERVER_URL']}, "
        f"Service: {_apm_cfg['SERVICE_NAME']}, Env: {_apm_cfg['ENVIRONMENT']})"
    )
else:
    _apm_logger.info("Elastic APM: DISABLED")

# Include routers
app.include_router(
    api_router,
    prefix="/v1",
    tags=["OCR Quality"]
)


# ---------------------------------------------------------------------------
# Exception handlers
#
# Early rejections (auth → 401/403, request validation → 422, unknown path →
# 404, etc.) are raised before the route handler runs, so without these handlers
# FastAPI renders them as `{"detail": ...}` and they never reach OcrKtpLog.
# These handlers re-shape such responses into the same envelope the route uses
# and log every rejected request — full parity with the sibling service, which
# logs all paths including early rejections (commit 2a1bd55).
# ---------------------------------------------------------------------------

_ERROR_CODES = {
    400: "INVALID_INPUT",
    401: "UNAUTHORIZED",
    403: "FORBIDDEN",
    404: "NOT_FOUND",
    405: "METHOD_NOT_ALLOWED",
    422: "VALIDATION_ERROR",
}


async def _log_early_rejection(
    request: Request, status_code: int, error_message: str
) -> None:
    """Best-effort log of a request rejected before the route body ran.

    Every rejected request is logged (parity with the sibling service), so
    404/405 from scanners/typos also land in OcrKtpLog.
    """
    try:
        await insert_log(
            request_id=request_id_ctx.get(),
            response_code=status_code,
            payload={"path": request.url.path, "method": request.method},
            error_message=error_message,
            result={},
            processing_time=0.0,
        )
    except Exception as exc:  # never let logging mask the real response
        get_logger(__name__).error(f"Failed to log early rejection: {exc}")


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Render HTTPExceptions (e.g. 401/403 from auth) in the standard envelope."""
    message = exc.detail if isinstance(exc.detail, str) else "Request rejected"
    await _log_early_rejection(request, exc.status_code, message)
    return _envelope(
        exc.status_code,
        request_id_ctx.get(),
        message=message,
        error_code=_ERROR_CODES.get(exc.status_code, "ERROR"),
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Render request-validation errors (422) in the standard envelope."""
    errors = exc.errors()
    await _log_early_rejection(request, 422, "Request validation failed")
    return _envelope(
        422,
        request_id_ctx.get(),
        message="Request validation failed",
        error_code="VALIDATION_ERROR",
        errors=jsonable_encoder(errors),
    )


@app.get("/")
async def root():
    """Root endpoint with service information."""
    return {
        "service": config.app.name,
        "version": config.app.version,
        "status": "running",
        "endpoints": {
            "health": "/health",
            "liveness": "/health/live",
            "readiness": "/health/ready",
            "ocr_quality": "/v1/ocr_quality (POST)",
        },
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


# ---------------------------------------------------------------------------
# Health endpoints
# ---------------------------------------------------------------------------

@app.get(
    "/health/live",
    response_model=LivenessResponse,
    summary="Liveness probe",
    description="Returns 200 if the process is alive. Use for K8s livenessProbe.",
)
async def liveness():
    """Liveness probe — lightweight, always returns 200 if the process can respond."""
    return LivenessResponse(
        status="alive",
        version=config.app.version,
    )


@app.get(
    "/health/ready",
    response_model=ReadinessResponse,
    summary="Readiness probe",
    description="Returns 200 if DB is up. Use for K8s readinessProbe.",
)
async def readiness_check():
    """
    Readiness probe — checks core infrastructure.

    Returns 200 if database is up.
    Returns 503 if database is down.
    """
    db_check = await _check_database()

    db_ok = db_check["status"] == "up"

    return JSONResponse(
        status_code=200 if db_ok else 503,
        content={
            "status": "ready" if db_ok else "not_ready",
            "checks": {
                "database": db_check,
            },
        },
    )


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Full health check",
    description="Returns detailed health status including database info.",
)
async def health_check():
    """
    Full health check — database info.

    Status logic:
    - healthy:   DB up
    - unhealthy: DB down
    """
    db_check = await _check_database()

    db_ok = db_check["status"] == "up"
    status = "healthy" if db_ok else "unhealthy"

    return HealthResponse(
        status=status,
        version=config.app.version,
        checks={
            "database": db_check,
        },
    )


if __name__ == "__main__":
    import uvicorn

    logger = get_logger(__name__)
    logger.info(f"Starting server on {config.server.host}:{config.server.port}")

    uvicorn.run(
        "src.main:app",
        host=config.server.host,
        port=config.server.port,
        reload=config.app.debug,
        workers=config.server.workers
    )
