"""
OCR KTP Orchestrator - Main Application

FastAPI application for orchestrating OCR extraction on Indonesian ID cards (KTP)
with quality and spoof detection.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from elasticapm.contrib.starlette import ElasticAPM, make_apm_client

from src.core.config import get_settings
from src.core.logging import setup_logging, get_logger
from src.core.device import log_device_info
from src.api.routes import router, register_exception_handlers
from src.services.minio_service import close_minio_service
from src.services.gcs_service import close_gcs_service
from src.api.models import HealthResponse
from src.middleware.rate_limiter import RateLimitMiddleware, SlidingWindowRateLimiter
from src.services.database_service import init_engine, dispose_engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    
    Handles startup and shutdown events.
    """
    # Startup
    logger = setup_logging()
    logger.info("=" * 60)
    logger.info("Starting OCR KTP Orchestrator")
    logger.info("=" * 60)
    
    # Validate configuration before startup
    from src.core.validation import validate_startup_configuration
    validate_startup_configuration()
    
    # Load configuration
    settings = get_settings()
    logger.info(f"Application: {settings.app.name} v{settings.app.version}")
    logger.info(f"Host: {settings.app.host}:{settings.app.port}")
    logger.info(f"Log file: {settings.logging.file}")
    
    # Log device information
    log_device_info()

    # Log running services
    logger.info("Running services:")
    logger.info(f"  OCR: {settings.run_services.ocr}")
    logger.info(f"  Quality Rulebase: {settings.run_services.quality}")
    logger.info(f"  Postprocess: {settings.run_services.postprocess}")
    logger.info(f"  Lamination: {settings.run_services.lamination}")
    logger.info(f"  Recapture: {settings.run_services.recapture}")
    logger.info(f"  Classifier: {settings.run_services.classifier}")
    logger.info(f"  Graycopy: {settings.run_services.graycopy}")
    logger.info(f"  QualityDL: {settings.run_services.qualitydl}")
    
    # Log service endpoints
    logger.info("Service endpoints:")
    logger.info(f"  OCR: {settings.services.ocr.url}")
    logger.info(f"  Quality Rulebase: {settings.services.quality.url}")
    logger.info(f"  Postprocess: {settings.services.postprocess.url}")
    logger.info(f"  Lamination: {settings.services.lamination.url}")
    logger.info(f"  Recapture: {settings.services.recapture.url}")
    logger.info(f"  QualityDL: {settings.services.qualitydl.url}")
    
    # Log rate limiting configuration
    if settings.rate_limit.enabled:
        logger.info(f"Rate limiting: ENABLED ({settings.rate_limit.requests_per_minute} req/min, {settings.rate_limit.requests_per_second} req/sec)")
    else:
        logger.info("Rate limiting: DISABLED")
    
    # Log Elastic APM configuration
    if settings.elastic_apm.enabled:
        logger.info(f"Elastic APM: ENABLED (Server: {settings.elastic_apm.server_url}, Service: {settings.elastic_apm.service_name}, Env: {settings.elastic_apm.environment})")
    else:
        logger.info("Elastic APM: DISABLED")

    logger.info("=" * 60)
    
    # Initialize async database engine
    await init_engine()
    
    # Initialize shared HTTP client for connection pooling
    from src.core.http_client import init_http_client
    await init_http_client()
    logger.info("Initialized shared HTTP client with connection pooling")
    
    yield
    
    # Shutdown
    from src.core.http_client import close_http_client
    await close_http_client()
    logger.info("Closed shared HTTP client")
    await dispose_engine()
    await close_minio_service()
    await close_gcs_service()
    logger.info("Shutting down OCR KTP Orchestrator")


# Create FastAPI app
settings = get_settings()
app = FastAPI(
    title=settings.app.name,
    version=settings.app.version,
    description="OCR extraction service for Indonesian ID cards (KTP) with quality and spoof detection",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,  # ty: ignore[invalid-argument-type]
    allow_origins=settings.cors.allow_origins,
    allow_credentials=settings.cors.allow_credentials and "*" not in settings.cors.allow_origins,  # BUG-21: no wildcard + credentials
    allow_methods=settings.cors.allow_methods,
    allow_headers=settings.cors.allow_headers,
)

# Add Rate Limiting middleware (if enabled)
if settings.rate_limit.enabled:
    rate_limiter = SlidingWindowRateLimiter(
        requests_per_minute=settings.rate_limit.requests_per_minute,
        requests_per_second=settings.rate_limit.requests_per_second,
        burst_size=settings.rate_limit.burst_size,
        cleanup_interval=settings.rate_limit.cleanup_interval,
    )
    app.add_middleware(
        RateLimitMiddleware,  # ty: ignore[invalid-argument-type]
        limiter=rate_limiter,
        exclude_paths=settings.rate_limit.exclude_paths
    )

# Add Elastic APM middleware (if enabled)
if settings.elastic_apm.enabled and settings.elastic_apm.server_url:
    apm_config = {
        'SERVICE_NAME': settings.elastic_apm.service_name,
        'SERVER_URL': settings.elastic_apm.server_url,
        'ENVIRONMENT': settings.elastic_apm.environment,
        'SECRET_TOKEN': settings.elastic_apm.secret_token,
        'VERIFY_SERVER_CERT': settings.elastic_apm.verify_server_cert,
    }
    apm = make_apm_client(apm_config)
    app.add_middleware(ElasticAPM, client=apm)  # ty: ignore[invalid-argument-type]

# Include routers
app.include_router(router)

# Register unified error-envelope handlers (auth/validation/HTTP errors)
register_exception_handlers(app)

# Get logger for this module
logger = get_logger(__name__)


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


def _check_http_client() -> dict:
    """Check if the shared aiohttp session is initialised and open."""
    try:
        from src.core.http_client import _http_client
        if _http_client is None or _http_client.closed:
            return {"status": "down", "reason": "not initialised"}
        return {"status": "up"}
    except Exception as exc:
        return {"status": "down", "reason": str(exc)}


async def _probe_service(name: str, url: str, api_key: str | None) -> tuple[str, dict]:
    """GET /health on a downstream service with a short timeout."""
    import aiohttp
    try:
        from src.core.http_client import _http_client as client
        if client is None or client.closed:
            return name, {"status": "unknown", "reason": "http_client down"}

        headers = {}
        if api_key:
            headers["x-api-key"] = api_key

        from urllib.parse import urlparse, urlunparse
        parsed = urlparse(url.rstrip("/"))
        health_url = urlunparse((parsed.scheme, parsed.netloc, "/health", "", "", ""))

        timeout = aiohttp.ClientTimeout(total=settings.app.health_check_timeout)
        async with client.get(health_url, timeout=timeout, headers=headers) as resp:
            if resp.status == 200:
                return name, {"status": "up", "url": health_url}
            return name, {"status": "down", "url": health_url, "http_status": resp.status}
    except Exception as exc:
        return name, {"status": "down", "reason": str(exc)}


async def _check_downstream_services() -> dict:
    """Probe all enabled downstream services in parallel."""
    import asyncio
    run = settings.run_services
    service_map = {
        "ocr": (run.ocr, settings.services.ocr),
        "classifier": (run.classifier, settings.services.classifier),
        "quality": (run.quality, settings.services.quality),
        "lamination": (run.lamination, settings.services.lamination),
        "recapture": (run.recapture, settings.services.recapture),
        "graycopy": (run.graycopy, settings.services.graycopy),
        "postprocess": (run.postprocess, settings.services.postprocess),
        "temper": (run.temper, settings.services.temper),
    }
    tasks = [
        _probe_service(name, cfg.url, cfg.api_key)
        for name, (enabled, cfg) in service_map.items()
        if enabled
    ]
    if not tasks:
        return {}
    results = await asyncio.gather(*tasks)
    return dict(results)


# ---------------------------------------------------------------------------
# Health endpoints
# ---------------------------------------------------------------------------

@app.get(
    "/health/live",
    summary="Liveness probe",
    description="Returns 200 if the process is alive. Use for K8s livenessProbe.",
)
async def liveness():
    """Liveness probe — lightweight, always returns 200 if the process can respond."""
    return JSONResponse(
        status_code=200,
        content={"status": "alive", "version": settings.app.version},
    )


@app.get(
    "/health/ready",
    summary="Readiness probe",
    description="Returns 200 if the service can accept traffic (DB + HTTP client up). Use for K8s readinessProbe.",
)
async def readiness():
    """
    Readiness probe — checks core infrastructure only.

    Returns 200 if database and HTTP client are up (ready to serve requests).
    Returns 503 if either is down (should not receive traffic).
    """
    db_check = await _check_database()
    http_check = _check_http_client()

    db_ok = db_check["status"] == "up"
    http_ok = http_check["status"] == "up"
    ready = db_ok and http_ok

    return JSONResponse(
        status_code=200 if ready else 503,
        content={
            "status": "ready" if ready else "not_ready",
            "checks": {
                "database": db_check,
                "http_client": http_check,
            },
        },
    )


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Full health check",
    description="Returns detailed health status including all downstream services.",
)
async def health_check():
    """
    Full health check — database, HTTP client, and all downstream services.

    Status logic:
    - healthy:   all checks pass
    - degraded:  DB + HTTP client OK, but one or more downstream services are down
    - unhealthy: DB or HTTP client is down
    """
    db_check = await _check_database()
    http_check = _check_http_client()
    services_health = await _check_downstream_services()

    try:
        from src.core.device import get_device
        device = get_device()
    except Exception:
        device = None

    db_ok = db_check["status"] == "up"
    http_ok = http_check["status"] == "up"
    all_services_up = all(v["status"] == "up" for v in services_health.values())

    if db_ok and http_ok and all_services_up:
        status = "healthy"
        http_status = 200
    elif db_ok and http_ok:
        status = "degraded"
        http_status = 200
    else:
        status = "unhealthy"
        http_status = 503

    return JSONResponse(
        status_code=http_status,
        content={
            "status": status,
            "version": settings.app.version,
            "device": device,
            "checks": {
                "database": db_check,
                "http_client": http_check,
                "services": services_health,
            },
        },
    )


@app.get("/")
async def root():
    """Root endpoint with service information."""
    return JSONResponse(
        status_code=200,
        content={
            "service": settings.app.name,
            "version": settings.app.version,
            "docs": "/docs",
            "health": "/health",
            "liveness": "/health/live",
            "readiness": "/health/ready"
        }
    )
