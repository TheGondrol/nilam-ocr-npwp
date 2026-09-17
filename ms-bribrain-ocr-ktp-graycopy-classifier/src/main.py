"""
FastAPI Application Entry Point for Graycopy Detection Service
Uses modern lifespan context manager for startup/shutdown
"""

import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import register_exception_handlers, router, set_detection_service, shutdown_executor
from src.core.config import get_config
from src.core.logging import get_logger, setup_logging
from src.middleware.add_requestid import RequestIdMiddleware
from src.services.graycopy_detection import GraycopyDetectionService
from src.services.minio_service import download_model_minio
from src.services.gcs_service import download_model_gcs
from src.services.database_service import init_engine, dispose_engine
from src.services.threshold_provider import init_provider, get_provider

# Load configuration
config = get_config()
APP_ENVIRO = os.getenv("APP_ENVIRO", "onprem")

# Setup logging
setup_logging(
    log_file=config.logging.file,
    log_level=config.logging.level,
    max_bytes=config.logging.max_bytes,
    backup_count=config.logging.backup_count,
)
logger = get_logger()

# Global detection service reference for cleanup
_detection_service: Optional[GraycopyDetectionService] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events.
    Modern replacement for deprecated @app.on_event decorators.
    """
    global _detection_service
    
    # ===== STARTUP =====
    logger.info("=" * 60)
    logger.info("Starting Graycopy Detection API...")
    logger.info("=" * 60)
    
    # Validate configuration before startup
    from src.core.validation import validate_startup_configuration
    validate_startup_configuration()
    
    try:
        logger.info("Downloading model...")
        if APP_ENVIRO == "onprem":
            download_model_minio()
        else:
            download_model_gcs()
        logger.info("Model downloaded successfully")
        
        # Initialize detection service
        logger.info("Initializing graycopy detection service...")
        _detection_service = GraycopyDetectionService(config)
        set_detection_service(_detection_service)
        logger.info("Service initialized successfully")
        
        # Warmup inference to load model into memory/cache
        logger.info("Performing warmup inference...")
        _detection_service.warmup()
        logger.info("Warmup completed")
        
        logger.info("=" * 60)
        logger.info(f"API ready on http://{config.server.host}:{config.server.port}")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Failed to initialize service: {str(e)}", exc_info=True)
        raise
    
    # Initialize async database engine
    await init_engine()

    # Initialize threshold provider (refreshes from management DB hourly)
    await init_provider(
        service_name="dgc_gry",
        defaults={"threshold": float(config.prediction.threshold)},
    ).initialize()

    yield  # Application runs here

    # ===== SHUTDOWN =====
    logger.info("Shutting down Graycopy Detection API...")

    # Stop threshold refresh task
    await get_provider().shutdown()

    # Dispose async database engine
    await dispose_engine()
    
    # Cleanup thread pool executor
    shutdown_executor()
    logger.info("Thread pool executor shut down")
    
    # Cleanup detection service resources
    if _detection_service is not None:
        _detection_service.cleanup()
        logger.info("Detection service resources cleaned up")
    
    logger.info("Shutdown complete")


# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Graycopy Detection API",
    description="API for detecting photocopied/graycopy ID documents vs original using ResNet-50",
    version="1.0.0",
    lifespan=lifespan
)

# Add Request ID middleware (must be first)
app.add_middleware(RequestIdMiddleware)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.cors.allow_origins,
    allow_credentials=config.cors.allow_credentials and "*" not in config.cors.allow_origins,  # BUG-21: no wildcard + credentials
    allow_methods=config.cors.allow_methods,
    allow_headers=config.cors.allow_headers,
)

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

# Include router
app.include_router(router)

# Register unified error-envelope handlers (auth/validation/HTTP/unhandled errors)
register_exception_handlers(app)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=config.server.host,
        port=config.server.port,
        log_level=config.logging.level.lower()
    )
