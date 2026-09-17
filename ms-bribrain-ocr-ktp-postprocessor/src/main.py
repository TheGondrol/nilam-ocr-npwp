"""Main FastAPI application."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import router, register_exception_handlers
from src.core.config import get_config
from src.core.logging import setup_logging
from src.middleware.add_requestid import RequestIdMiddleware
from src.services.database_service import init_engine, dispose_engine
from src.services.threshold_provider import init_provider, get_provider

# Initialize logging first
setup_logging()
logger = logging.getLogger(__name__)

# Get configuration
config = get_config()
api_config = config.get_api_config()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager for startup and shutdown events."""
    # Startup
    logger.info("=" * 60)
    logger.info("Starting OCR Postprocess API")
    logger.info("=" * 60)
    
    # Validate configuration before startup
    from src.core.validation import validate_startup_configuration
    validate_startup_configuration()
    
    # Log configuration info
    logger.info(f"API Version: {api_config.get('version', '1.0.0')}")
    logger.info(f"Host: {api_config.get('host', '0.0.0.0')}")
    logger.info(f"Port: {api_config.get('port', 8000)}")
    
    # Log threshold configuration
    thresholds = config.get_thresholds()
    logger.info(f"Thresholds: {thresholds}")
    
    # Initialize async database engine
    await init_engine()

    # Initialize threshold provider (refreshes from management DB hourly)
    thresholds = config.get_thresholds()
    await init_provider(
        service_name="dgc_pps",
        defaults={
            "partial": float(thresholds.get("partial", 75)),
            "ratio": float(thresholds.get("ratio", 80)),
            "confidence": float(thresholds.get("confidence", 0.8)),
        },
    ).initialize()

    logger.info("=" * 60)
    logger.info("Application startup complete")
    logger.info("=" * 60)

    yield  # Application runs here

    # Shutdown
    logger.info("Shutting down OCR Postprocess API")
    await get_provider().shutdown()
    await dispose_engine()  # Clean up database connections
    logger.info("Database connections closed")


# Create FastAPI app with lifespan
app = FastAPI(
    title=api_config.get("title", "OCR Postprocess API"),
    description=api_config.get("description", "API for post-processing OCR text extraction from KTP"),
    version=api_config.get("version", "1.0.0"),
    lifespan=lifespan,
)

# Add CORS middleware
cors_config = config.get_cors_config()
app.add_middleware(
    CORSMiddleware,  # type: ignore[arg-type]
    allow_origins=cors_config["allow_origins"],
    allow_credentials=cors_config["allow_credentials"] and "*" not in cors_config["allow_origins"],  # BUG-21: no wildcard + credentials
    allow_methods=cors_config["allow_methods"],
    allow_headers=cors_config["allow_headers"],
)

# Add RequestId middleware
app.add_middleware(RequestIdMiddleware)  # type: ignore[arg-type]

# Add Elastic APM middleware (if enabled)
if config.elastic_apm.enabled and config.elastic_apm.server_url:
    from elasticapm.contrib.starlette import ElasticAPM, make_apm_client  # type: ignore[unresolved-import]
    _apm_cfg = {
        "SERVICE_NAME": config.elastic_apm.service_name,
        "SERVER_URL": config.elastic_apm.server_url,
        "ENVIRONMENT": config.elastic_apm.environment,
        "SECRET_TOKEN": config.elastic_apm.secret_token,
        "VERIFY_SERVER_CERT": config.elastic_apm.verify_server_cert,
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


@app.get("/")
async def root():
    """Root endpoint with service info and endpoint listing."""
    return {
        "message": "OCR Postprocess API",
        "description": "API for post-processing OCR text extraction from KTP",
        "version": api_config.get("version", "1.0.0"),
        "endpoints": {
            "health": "/health",
            "liveness": "/health/live",
            "readiness": "/health/ready",
            "predict": "/v1/ocr_postprocess",
            "docs": "/docs",
        },
    }


# ---------------------------------------------------------------------------
# Health helpers
# ---------------------------------------------------------------------------


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


@app.get("/health/live")
async def liveness():
    """Liveness probe — always returns 200 if the process is alive."""
    return {
        "status": "alive",
        "version": api_config.get("version", "1.0.0"),
    }


@app.get("/health/ready")
async def readiness():
    """Readiness probe — 200 if database is reachable, 503 otherwise."""
    from fastapi.responses import JSONResponse

    db = await _check_database()
    checks = {"database": db}

    if db["status"] == "up":
        return {"status": "ready", "checks": checks}

    return JSONResponse(
        status_code=503,
        content={"status": "not_ready", "checks": checks},
    )


@app.get("/health")
async def health_check():
    """Full health check with component details."""
    from fastapi.responses import JSONResponse

    db = await _check_database()
    checks = {"database": db}

    status_val = "healthy" if db["status"] == "up" else "unhealthy"
    code = 200 if status_val == "healthy" else 503

    body = {
        "status": status_val,
        "version": api_config.get("version", "1.0.0"),
        "checks": checks,
    }

    if code == 200:
        return body
    return JSONResponse(status_code=code, content=body)
