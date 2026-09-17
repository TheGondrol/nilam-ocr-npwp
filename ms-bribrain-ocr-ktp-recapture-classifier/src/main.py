"""
FastAPI Application Entry Point
Screen Recapture Detection API
"""

import os
import sys
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import torch
import numpy as np
from PIL import Image
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .core.config import config
from .core.device import get_device
from .models.ml_model import load_model, get_transform
from .api.routes import router, register_exception_handlers
from .services.recapture_service import set_model_globals
from .middleware.add_requestid import RequestIdMiddleware
from .core.logging import logger
from .services.minio_service import download_model_minio
from .services.gcs_service import download_model_gcs
from .services.database_services import init_engine, dispose_engine
from .services.threshold_provider import init_provider, get_provider

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


def _warmup_model(model, device, transform):
    """
    Warmup the model by running dummy inferences.
    This triggers any lazy compilation during startup instead of during
    the first request, improving response time consistency.
    
    Runs 3 warmup iterations to ensure:
    - torch.compile fully optimizes the model (GPU)
    - JIT tracing is complete (CPU)
    - CUDA kernels are loaded and cached
    """
    try:
        logger.info("Warming up model...")
        crop_size = config.get('model.crop_size', 224)
        
        # Create a dummy image
        dummy_image = Image.fromarray(
            np.zeros((crop_size, crop_size, 3), dtype=np.uint8)
        )
        
        # Run 3 warmup iterations
        tensor = transform(dummy_image).unsqueeze(0).to(device)
        with torch.inference_mode():
            for i in range(3):
                _ = model(tensor)
                logger.debug(f"Warmup iteration {i+1}/3 complete")
        
        # Cleanup
        del tensor
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        
        logger.info("Model warmup complete (3 iterations)")
    except Exception as e:
        logger.warning(f"Model warmup failed (non-critical): {str(e)}")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application lifespan - startup and shutdown"""
    # Configure CPU threading before model loading
    _configure_cpu_threading()
    
    # Startup
    logger.info("=" * 80)
    logger.info("Starting Screen Recapture Detection API")
    logger.info("=" * 80)
    
    # Validate configuration before startup
    from src.core.validation import validate_startup_configuration
    validate_startup_configuration()
    
    try:
        # Get device configuration
        device = get_device(force_cpu=config.get('device.force_cpu', False))
        logger.info(f"Device initialized: {device}")
        
        # Download model from MinIO if not present locally
        if APP_ENVIRO == "onprem":
            download_model_minio()
        else:
            download_model_gcs()

        # Load model
        logger.info(f"Loading model from: {config.model_path}")
        model = load_model(
            model_path=config.model_path,
            device=device,
            num_classes=config.get('model.num_classes', 2)
        )
        
        # Get transform
        transform = get_transform(crop_size=config.get('model.crop_size'))
        logger.info(f"Preprocessing: {config.get('model.crop_size')}x{config.get('model.crop_size')} center crop + ImageNet normalization")
        
        # Set globals in service module
        set_model_globals(model, device, transform)
        
        # Warmup the model
        _warmup_model(model, device, transform)
        
        logger.info("=" * 80)
        logger.info("Application startup complete")
        logger.info(f"Server running on {config.server_host}:{config.server_port}")
        logger.info("=" * 80)
        
        # Initialize async database engine
        await init_engine()

        # Initialize threshold provider (refreshes from management DB hourly)
        await init_provider(
            service_name="dgc_rct",
            defaults={"threshold": float(config.threshold)},
        ).initialize()

    except Exception as e:
        logger.error(f"Failed to initialize application: {str(e)}", exc_info=True)
        sys.exit(1)

    yield  # Application runs here

    # Shutdown
    logger.info("Shutting down Screen Recapture Detection API")
    try:
        await get_provider().shutdown()
    except Exception as e:
        logger.error(f"Error stopping threshold provider: {e}")
    await dispose_engine()
    logger.info("Database connections closed")


# Initialize FastAPI app with lifespan context manager
app = FastAPI(
    title=config.get('api.title', "Screen Recapture Detection API"),
    description=config.get('api.description', "API for detecting recaptured screen photos vs original captures using ResNet-50"),
    version=config.get('api.version', "1.0.0"),
    lifespan=lifespan
)

# Add CORS middleware with configurable origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.get('server.cors_allowed_origins', ["*"]),
    allow_credentials=config.get('server.cors_allowed_credentials', True) and "*" not in config.get('server.cors_allowed_origins', ["*"]),  # BUG-21: no wildcard + credentials
    allow_methods=config.get('server.cors_allowed_methods', ["*"]),
    allow_headers=config.get('server.cors_allowed_headers', ["*"]),
)

# Add RequestId middleware
app.add_middleware(RequestIdMiddleware)

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
    app.add_middleware(ElasticAPM, client=apm)
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
        "src.main:app",
        host=config.server_host,
        port=config.server_port,
        reload=False,
        log_level=config.get('logging.level', 'INFO').lower()
    )
