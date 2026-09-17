"""
FastAPI Service for ID Document Lamination Detection
Detects unlaminated (suspicious) vs laminated (normal) documents
"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .core.logging import logger
from .core.config import config
from .services.predictor import predictor
from .api.routes import router, _executor, register_exception_handlers
from .services.minio_service import download_model_minio
from .services.gcs_service import download_model_gcs
from .services.database_service import init_engine, dispose_engine
from .services.threshold_provider import init_provider, get_provider
from .middleware.add_requestid import RequestIdMiddleware

APP_ENVIRO = os.getenv("APP_ENVIRO", "onprem")


def _configure_cpu_threading():
    """
    Configure CPU threading for optimal performance.
    Sets thread count based on available CPUs or config.
    """
    num_threads = config.get('performance.num_threads', None)
    if num_threads is None:
        # Use half of available CPUs, minimum 1
        num_threads = max(1, (os.cpu_count() or 4) // 2)
    
    # Set threading environment variables
    os.environ.setdefault("OMP_NUM_THREADS", str(num_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(num_threads))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(num_threads))
    
    logger.info(f"CPU threading configured: OMP_NUM_THREADS={num_threads}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown"""
    # Configure CPU threading before model loading
    _configure_cpu_threading()
    
    # Startup
    logger.info("Starting up the application...")
    
    # Validate configuration before startup
    from src.core.validation import validate_startup_configuration
    validate_startup_configuration()
    
    logger.info(f"Configuration loaded: {config.get_all()}")
    if APP_ENVIRO == "onprem":
        download_model_minio()
    else:
        download_model_gcs()

    # Initialize async database engine
    await init_engine()

    # Initialize threshold provider (refreshes from management DB hourly)
    await init_provider(
        service_name="dgc_lmt",
        defaults={"threshold": float(config.threshold)},
    ).initialize()

    predictor.load_model()
    logger.info("Application startup complete")

    yield

    # Shutdown
    logger.info("Shutting down the application...")
    await get_provider().shutdown()
    _executor.shutdown(wait=True)
    logger.info("ThreadPoolExecutor shutdown complete")
    await dispose_engine()
    predictor.cleanup()
    logger.info("Application shutdown complete")


# Initialize FastAPI app with lifespan
app = FastAPI(
    lifespan=lifespan,
    title="Document Lamination Detection API",
    description="API for detecting unlaminated (suspicious) vs laminated (normal) ID documents using CNN",
    version="1.0.0"
)

# Add CORS middleware
cors_settings = config.cors_config
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_settings['allow_origins'],
    allow_credentials=cors_settings['allow_credentials'] and "*" not in cors_settings['allow_origins'],  # BUG-21: no wildcard + credentials
    allow_methods=cors_settings['allow_methods'],
    allow_headers=cors_settings['allow_headers'],
)

# Add RequestIdMiddleware
app.add_middleware(RequestIdMiddleware)

# Add Elastic APM middleware (if enabled)
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
    app.add_middleware(ElasticAPM, client=apm)
    logger.info(
        f"Elastic APM: ENABLED (Server: {_apm_cfg['SERVER_URL']}, "
        f"Service: {_apm_cfg['SERVICE_NAME']}, Env: {_apm_cfg['ENVIRONMENT']})"
    )
else:
    logger.info("Elastic APM: DISABLED")

# Include API routes
app.include_router(router)

# Register unified error-envelope handlers (auth/validation/HTTP errors)
register_exception_handlers(app)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.main:app",
        host=config.server_host,
        port=config.server_port,
        reload=False,
        log_level="info"
    )
