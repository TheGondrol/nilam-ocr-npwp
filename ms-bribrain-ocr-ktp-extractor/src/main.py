"""Main FastAPI application"""

import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.core.logging import setup_logging
from src.api.routes import router, register_exception_handlers
from src.core.device import detect_device
from src.middleware.add_requestid import RequestIdMiddleware
from src.services.ocr_service import cleanup_ocr, is_ocr_ready, warmup_ocr
from src.models.schemas import HealthResponse, LivenessResponse, ReadinessResponse, RootResponse
from src.services.gcs_service import download_model_gcs
from src.services.database_service import init_engine, dispose_engine
from src.services.threshold_provider import init_provider, get_provider
from src.core.config import settings as _ext_settings

# Application version
VERSION = "1.0.0"

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    Handles startup and shutdown events.
    """
    # Startup
    logger.info("=" * 60)
    logger.info("Starting OCR Extract Application")
    logger.info("=" * 60)

    # Validate configuration before startup
    from src.core.validation import validate_startup_configuration
    validate_startup_configuration()

    # Initialize async database engine
    await init_engine()

    # Initialize threshold provider (refreshes from management DB hourly)
    await init_provider(
        service_name="dgc_ext",
        defaults={
            "width_threshold_ratio": float(_ext_settings.ocr_width_threshold_ratio),
        },
    ).initialize()

    # Detect and log device
    try:
        device = detect_device()
        logger.info(f"Device configuration: {device.upper()}")
        app_enviro = os.getenv("APP_ENVIRO", "onprem")
        if app_enviro != "onprem":
            download_model_gcs(device)
        else:
            logger.info("On-prem environment: skipping GCS model download")
        # Pre-warm TensorRT engine to avoid slow first request
        warmup_ocr()

        logger.info("Application startup complete")
    except Exception as e:
        logger.error(f"Error during device detection: {str(e)}", exc_info=True)

    yield

    # Shutdown
    logger.info("Initiating graceful shutdown...")

    # Stop threshold refresh task
    try:
        await get_provider().shutdown()
    except Exception as e:
        logger.error(f"Error during threshold provider shutdown: {str(e)}", exc_info=True)

    # Dispose async database engine
    try:
        await dispose_engine()
    except Exception as e:
        logger.error(f"Error during database cleanup: {str(e)}", exc_info=True)

    # Clean up OCR resources (thread pool, GPU memory)
    try:
        cleanup_ocr()
    except Exception as e:
        logger.error(f"Error during OCR cleanup: {str(e)}", exc_info=True)

    logger.info("OCR Extract Application shutdown complete")


# Initialize FastAPI app
app = FastAPI(
    title="OCR Extract API",
    description="API for extracting text from images using PaddleOCR",
    version=VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=[
        {
            "name": "OCR",
            "description": "OCR text extraction endpoints"
        },
        {
            "name": "Health",
            "description": "Health check endpoints"
        }
    ]
)

# Setup logging
setup_logging()

# Add CORS middleware (origins configured in config.yaml)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_ext_settings.cors_allow_origins,
    allow_credentials=_ext_settings.cors_allow_credentials and "*" not in _ext_settings.cors_allow_origins,  # BUG-21: no wildcard + credentials
    allow_methods=_ext_settings.cors_allow_methods,
    allow_headers=_ext_settings.cors_allow_headers,
)

# Add RequestIdMiddleware
app.add_middleware(RequestIdMiddleware)

# Add Elastic APM middleware (if enabled)
if _ext_settings.elastic_apm_enabled and _ext_settings.elastic_apm_server_url:
    from elasticapm.contrib.starlette import ElasticAPM, make_apm_client
    _apm_cfg = {
        "SERVICE_NAME": _ext_settings.elastic_apm_service_name,
        "SERVER_URL": _ext_settings.elastic_apm_server_url,
        "ENVIRONMENT": _ext_settings.elastic_apm_environment,
        "SECRET_TOKEN": _ext_settings.elastic_apm_secret_token,
        "VERIFY_SERVER_CERT": _ext_settings.elastic_apm_verify_server_cert,
    }
    apm = make_apm_client(_apm_cfg)
    app.add_middleware(ElasticAPM, client=apm)
    logger.info(
        f"Elastic APM: ENABLED (Server: {_apm_cfg['SERVER_URL']}, "
        f"Service: {_apm_cfg['SERVICE_NAME']}, Env: {_apm_cfg['ENVIRONMENT']})"
    )
else:
    logger.info("Elastic APM: DISABLED")

# Include routers
app.include_router(router)

# Register unified error-envelope handlers (auth/validation/HTTP errors)
register_exception_handlers(app)


@app.get(
    "/",
    response_model=RootResponse,
    tags=["Health"],
    summary="Root endpoint"
)
async def root():
    """Root endpoint with API information."""
    return {
        "message": "OCR Extract API",
        "version": VERSION,
        "endpoints": {
            "ocr_extract": "/v1/ocr_extract",
            "health": "/health",
            "liveness": "/health/live",
            "readiness": "/health/ready",
            "docs": "/docs",
        }
    }


# ---------------------------------------------------------------------------
# Health helpers
# ---------------------------------------------------------------------------


def _check_ocr() -> dict:
    """Check OCR engine status."""
    try:
        if is_ocr_ready():
            return {"status": "up"}
        return {"status": "down", "reason": "OCR engine not loaded"}
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


# ---------------------------------------------------------------------------
# Health endpoints
# ---------------------------------------------------------------------------


@app.get(
    "/health/live",
    response_model=LivenessResponse,
    tags=["Health"],
    summary="Liveness probe"
)
async def liveness():
    """Liveness probe - always returns 200 if the process is alive."""
    return LivenessResponse(status="alive", version=VERSION)


@app.get(
    "/health/ready",
    response_model=ReadinessResponse,
    tags=["Health"],
    summary="Readiness probe"
)
async def readiness():
    """Readiness probe - 200 if OCR loaded and DB reachable, 503 otherwise."""
    ocr = _check_ocr()
    db = await _check_database()
    checks = {"ocr": ocr, "database": db}

    if ocr["status"] == "up" and db["status"] == "up":
        return {"status": "ready", "checks": checks}

    from fastapi.responses import JSONResponse
    return JSONResponse(
        status_code=503,
        content={"status": "not_ready", "checks": checks},
    )


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"],
    summary="Full health check"
)
async def health_check():
    """Full health check with component details."""
    ocr = _check_ocr()
    db = await _check_database()
    checks = {"ocr": ocr, "database": db}

    ocr_up = ocr["status"] == "up"
    db_up = db["status"] == "up"

    if ocr_up and db_up:
        status_val = "healthy"
    elif ocr_up and not db_up:
        status_val = "degraded"
    else:
        status_val = "unhealthy"

    code = 200 if status_val in ("healthy", "degraded") else 503

    body = HealthResponse(
        status=status_val,
        ocr_loaded=ocr_up,
        version=VERSION,
        checks=checks,
    )

    if code == 200:
        return body

    from fastapi.responses import JSONResponse
    return JSONResponse(status_code=code, content=body.model_dump())


if __name__ == "__main__":
    import uvicorn
    from src.core.config import settings

    uvicorn.run(
        "src.main:app",
        host=settings.server_host,
        port=settings.server_port,
        reload=settings.server_reload
    )
