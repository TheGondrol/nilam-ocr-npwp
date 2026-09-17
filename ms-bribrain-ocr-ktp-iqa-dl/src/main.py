"""Main FastAPI application with lifespan management."""

from __future__ import annotations

import logging
import os

from dotenv import load_dotenv

load_dotenv()
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import register_exception_handlers, router, set_executor
from src.core.config import config
from src.core.device import get_device
from src.core.logging import setup_logging
from src.middleware.add_requestid import RequestIdMiddleware
from src.models.ml_model import get_transform, load_mobilenet_model, warmup_model
from src.services.database_services import dispose_engine, init_engine
from src.services.threshold_provider import init_provider, get_provider
from src.services.minio_service import download_model_minio
from src.services.gcs_service import download_model_gcs
from src.services.quality_service import set_model_globals

# Setup logging
logger = setup_logging()

APP_ENVIRO = os.getenv("APP_ENVIRO", "onprem")

print("APP_ENVIRO", APP_ENVIRO)

def _configure_cpu_threading() -> None:
    """Configure CPU threading for optimal performance."""
    num_threads = str(config.thread_pool_workers)
    os.environ["OMP_NUM_THREADS"] = num_threads
    os.environ["MKL_NUM_THREADS"] = num_threads
    logger.info(f"CPU threading configured: OMP_NUM_THREADS={num_threads}, MKL_NUM_THREADS={num_threads}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events.

    Startup:
    - Validate configuration
    - Configure CPU threading
    - Detect GPU/CPU device
    - Download model from MinIO if needed
    - Load MobileNet model
    - Setup preprocessing transforms
    - Warmup model with dummy inference
    - Initialize database connection

    Shutdown:
    - Dispose database connections
    - Clear GPU cache
    """
    logger.info("=" * 80)
    logger.info("Starting KTP Image Quality Classifier API")
    logger.info("=" * 80)

    # Validate configuration before startup
    from src.core.validation import validate_startup_configuration
    validate_startup_configuration()

    # Configure CPU threading
    _configure_cpu_threading()

    # Get device
    force_cpu = config.get("device.force_cpu", False)
    device = get_device(force_cpu=force_cpu)

    # Download model from MinIO if needed
    try:
        if APP_ENVIRO == "onprem":
            download_model_minio()
        else:
            download_model_gcs()
    except Exception as e:
        logger.warning(f"Model download failed: {e}. Will attempt to use local model.")

    # Load model
    try:
        model = load_mobilenet_model(
            model_path=config.model_path,
            num_classes=config.num_classes,
            device=device,
        )
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

    # Get transform
    transform = get_transform()

    # Set global model state
    set_model_globals(model, device, transform)

    # Warmup model
    warmup_model(model, device, transform)

    # Initialize thread pool executor
    num_workers = config.thread_pool_workers
    executor = ThreadPoolExecutor(max_workers=num_workers)
    set_executor(executor)
    logger.info(f"Thread pool executor initialized with {num_workers} workers")

    # Initialize database
    try:
        await init_engine()
    except Exception as e:
        logger.warning(f"Database initialization failed: {e}. Logging will be disabled.")

    # Initialize threshold provider (refreshes from management DB hourly)
    await init_provider(
        service_name="dgc_idl",
        defaults={
            "confidence_threshold": float(config.get("prediction.confidence_threshold", 0.67)),
            "bad_crop_threshold": float(config.bad_crop_threshold),
        },
    ).initialize()

    logger.info("=" * 80)
    logger.info("API startup complete - ready to accept requests")
    logger.info("=" * 80)

    yield  # Application runs

    # Shutdown
    logger.info("=" * 80)
    logger.info("Shutting down API")
    logger.info("=" * 80)

    # Stop threshold refresh task
    try:
        await get_provider().shutdown()
    except Exception as e:
        logger.error(f"Error stopping threshold provider: {e}")

    # Dispose database
    try:
        await dispose_engine()
    except Exception as e:
        logger.error(f"Error disposing database: {e}")

    # Shutdown executor
    executor.shutdown(wait=True)
    logger.info("Thread pool executor shutdown")

    # Clear GPU cache
    if device.type == "cuda":
        torch.cuda.empty_cache()
        logger.info("GPU cache cleared")

    logger.info("Shutdown complete")


# Create FastAPI app
app = FastAPI(
    title=config.get("api.title", "KTP Image Quality Classifier API"),
    description=config.get(
        "api.description",
        "Classifies KTP image quality by analyzing OCR text crop quality",
    ),
    version=config.get("api.version", "1.0.0"),
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,  # ty: ignore[invalid-argument-type]
    allow_origins=config.get("server.cors_allowed_origins", ["*"]),
    allow_credentials=config.get("server.cors_allowed_credentials", True) and "*" not in config.get("server.cors_allowed_origins", ["*"]),  # BUG-21: no wildcard + credentials
    allow_methods=config.get("server.cors_allowed_methods", ["*"]),
    allow_headers=config.get("server.cors_allowed_headers", ["*"]),
)

# Add request ID middleware
app.add_middleware(RequestIdMiddleware)  # ty: ignore[invalid-argument-type]

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
    app.add_middleware(ElasticAPM, client=apm)  # ty: ignore[invalid-argument-type]
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

# Log middleware configuration
logger.info("CORS middleware configured")
logger.info("RequestId middleware configured")
