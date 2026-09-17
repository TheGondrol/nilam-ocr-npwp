"""
FastAPI Application Entry Point
Document Tamper Detection Service
"""

import logging
import os
from contextlib import asynccontextmanager
from typing import Any, cast

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import router, register_exception_handlers
from src.core import config, setup_logging
from src.middleware.add_requestid import RequestIdMiddleware
from src.services.tamper_detection import initialize_tamper_service
from src.services.minio_service import download_model_minio
from src.services.gcs_service import download_model_gcs
from src.services.database_service import init_engine, dispose_engine
from src.services.threshold_provider import init_provider, get_provider

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

APP_ENVIRO = os.getenv("APP_ENVIRO", "onprem")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager for startup and shutdown events."""
    # Startup
    logger.info("Starting Document Tamper Detection API...")
    
    # Validate configuration before startup
    from src.core.validation import validate_startup_configuration
    validate_startup_configuration()
    
    logger.info(f"Configuration loaded: model_path={config.model.path}, image_size={config.model.image_size}")
    logger.info(f"Device configuration: force_cpu={config.device.force_cpu}")
    
    try:
        logger.info("Downloading model...")
        if APP_ENVIRO == "onprem":
            download_model_minio()
        else:
            download_model_gcs()
        logger.info("Model downloaded successfully")
        
        # Initialize tamper detection service
        initialize_tamper_service(
            model_path=config.model.path,
            image_size=config.model.image_size,
            force_cpu=config.device.force_cpu
        )
        logger.info("Tamper detection service initialized successfully")
        
        # Initialize async database engine
        await init_engine()

        # Initialize threshold provider (refreshes from management DB hourly)
        await init_provider(
            service_name="dgc_tpr",
            defaults={"threshold": float(config.prediction.threshold)},
        ).initialize()

        logger.info(f"Server will run on {config.server.host}:{config.server.port}")

    except Exception as e:
        logger.error(f"Failed to initialize services: {str(e)}")
        raise

    yield  # Application runs here

    # Shutdown
    logger.info("Shutting down Document Tamper Detection API...")
    try:
        await get_provider().shutdown()
    except Exception as e:
        logger.error(f"Error stopping threshold provider: {e}")
    await dispose_engine()
    logger.info("Database connections closed")


# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Document Tamper Detection API",
    description="API for detecting tampered documents (text modifications) using fine-tuned font-identifier model",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    cast(Any, CORSMiddleware),
    allow_origins=config.cors.allow_origins,
    allow_credentials=config.cors.allow_credentials and "*" not in config.cors.allow_origins,  # BUG-21: no wildcard + credentials
    allow_methods=config.cors.allow_methods,
    allow_headers=config.cors.allow_headers,
)

# Add RequestId middleware
app.add_middleware(cast(Any, RequestIdMiddleware))

# Add Elastic APM middleware (if enabled)
if config.elastic_apm.enabled and config.elastic_apm.server_url:
    from elasticapm.contrib.starlette import ElasticAPM, make_apm_client  # ty: ignore[unresolved-import]
    _apm_cfg = {
        "SERVICE_NAME": config.elastic_apm.service_name,
        "SERVER_URL": config.elastic_apm.server_url,
        "ENVIRONMENT": config.elastic_apm.environment,
        "SECRET_TOKEN": config.elastic_apm.secret_token,
        "VERIFY_SERVER_CERT": config.elastic_apm.verify_server_cert,
    }
    apm = make_apm_client(_apm_cfg)
    app.add_middleware(cast(Any, ElasticAPM), client=apm)
    logger.info(
        f"Elastic APM: ENABLED (Server: {_apm_cfg['SERVER_URL']}, "
        f"Service: {_apm_cfg['SERVICE_NAME']}, Env: {_apm_cfg['ENVIRONMENT']})"
    )
else:
    logger.info("Elastic APM: DISABLED")

# Include router
app.include_router(router)

# Register unified error-envelope handlers (auth/validation/HTTP errors)
register_exception_handlers(app)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=config.server.host,
        port=config.server.port
    )
