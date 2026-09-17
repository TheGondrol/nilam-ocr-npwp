"""
FastAPI Service for YOLO Object Detection
Detects ktp and non-ktp objects using YOLO model
"""
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .core.logging import logger
from .core.config import config
from .services.predictor import predictor
from .api.routes import router, register_exception_handlers
from .services.minio_service import download_model_minio
from src.middleware.add_requestid import RequestIdMiddleware
from .services.database_service import init_engine, dispose_engine
from .services.gcs_service import download_model_gcs, download_openvino_gcs
from .services.threshold_provider import init_provider, get_provider


APP_ENVIRO=os.getenv("APP_ENVIRO")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown events."""
    # Validate configuration before startup
    from .core.validation import validate_startup_configuration
    validate_startup_configuration()
    
    # Startup
    logger.info("Starting up the application...")
    logger.info(f"Configuration loaded: {config.get_all()}")
    if APP_ENVIRO == "onprem":
        download_model_minio()
    else:
        download_model_gcs()
        download_openvino_gcs()

    # Initialize async database engine
    await init_engine()

    # Initialize threshold provider before model warmup so the predictor
    # can read live values from get_provider().get(...)
    provider = init_provider(
        service_name="dgc_dt",
        defaults={
            "confidence": float(config.confidence),
            "iou_threshold": float(config.iou_threshold),
        },
    )
    await provider.initialize()

    predictor.load_model()

    logger.info("Application startup complete")

    yield

    # Shutdown
    logger.info("Shutting down the application...")
    await get_provider().shutdown()
    await dispose_engine()
    logger.info("Application shutdown complete")


# Initialize FastAPI app
app = FastAPI(
    title="KTP Detection API",
    description="API for YOLO-based ktp/non-ktp object detection",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
cors_settings = config.cors_config
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_settings["allow_origins"],
    allow_credentials=cors_settings["allow_credentials"] and "*" not in cors_settings["allow_origins"],  # BUG-21: no wildcard + credentials
    allow_methods=cors_settings["allow_methods"],
    allow_headers=cors_settings["allow_headers"],
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
    import multiprocessing
    import sys
    import signal

    # Define exception hook to log unhandled exceptions
    def custom_excepthook(exc_type, exc_value, exc_traceback):
        logger.critical(
            "Unhandled exception:", exc_info=(exc_type, exc_value, exc_traceback)
        )
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return

    sys.excepthook = custom_excepthook

    # Define signal handlers
    def signal_handler(sig, frame):
        sig_name = signal.Signals(sig).name
        logger.info(f"Received shutdown signal: {sig_name} ({sig})")
        # Initialize graceful shutdown here if needed, but for now just log
        # If we are running uvicorn, it should handle the signal too if we propagate or if it installed its own.
        # However, since we are installing this before uvicorn, uvicorn might overwrite it.
        # If uvicorn overwrites it, this log won't show.
        # But we can try to install it.
        sys.exit(0)

    # Note: Uvicorn installs its own signal handlers by default.
    # To see our logs, we rely on the try-except block below or unexpected crashes.

    # Auto-detect optimal worker count (can be overridden in config)
    workers = config.get("performance.workers", 0)
    if workers == 0:
        workers = multiprocessing.cpu_count()

    try:
        logger.info("Starting uvicorn server...")
        uvicorn.run(
            "src.main:app",
            host=config.server_host,
            port=config.server_port,
            workers=workers,
            reload=False,
            log_level="info",
            backlog=2048,  # Increased connection queue
            timeout_keep_alive=60,
        )
    except KeyboardInterrupt:
        logger.info("Application stopped by KeyboardInterrupt (SIGINT)")
    except Exception as e:
        logger.critical(f"Application crash detected: {str(e)}", exc_info=True)
        sys.exit(1)
    finally:
        logger.info("Uvicorn execution finished.")
