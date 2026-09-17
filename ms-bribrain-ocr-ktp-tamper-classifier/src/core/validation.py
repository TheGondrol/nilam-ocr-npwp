"""
Configuration validation module.
Validates required environment variables and config values before application startup.
"""

import logging
import os
import sys
from pathlib import Path
from typing import List, Tuple

from src.core.config import config

logger = logging.getLogger(__name__)


def validate_environment_variables() -> Tuple[bool, List[str]]:
    """
    Validate required environment variables are set.
    Checks MinIO or GCS variables depending on APP_ENVIRO.

    Returns:
        Tuple of (is_valid, missing_vars)
    """
    app_enviro = os.getenv("APP_ENVIRO", "onprem")

    required_env_vars = [
        "DATABASE_URL",
        "API_KEY",
    ]

    if app_enviro == "onprem":
        required_env_vars += [
            "MINIO_ENDPOINT",
            "MINIO_ACCESS_KEY",
            "MINIO_SECRET_KEY",
        ]
    # GCS credentials come from ADC (the VM's attached service account) or
    # optional SA_* env vars (see gcs_service._load_credentials); not required
    # at startup.

    missing_vars = []

    for var in required_env_vars:
        value = os.getenv(var)
        if not value:
            missing_vars.append(f"  - {var}")

    is_valid = len(missing_vars) == 0
    return is_valid, missing_vars


def validate_config_values() -> Tuple[bool, List[str]]:
    """
    Validate required config values are set.
    
    Returns:
        Tuple of (is_valid, missing_configs)
    """
    missing_configs = []
    
    # Validate model path
    if not config.model.path or not config.model.path.strip():
        missing_configs.append("  - model.path (Model path)")
    else:
        # Check if model file exists (will be downloaded if not)
        model_path = Path(config.model.path)
        if not model_path.exists():
            logger.warning(f"Model file not found at {config.model.path}, will be downloaded from MinIO")
    
    # Validate server settings
    if not config.server.host or not config.server.host.strip():
        missing_configs.append("  - server.host (Server host)")
    
    if not config.server.port:
        missing_configs.append("  - server.port (Server port)")
    
    # Validate MinIO settings (only required for on-prem)
    app_enviro = os.getenv("APP_ENVIRO", "onprem")
    if app_enviro == "onprem":
        if not config.minio.bucket or not config.minio.bucket.strip():
            missing_configs.append("  - minio.bucket (MinIO bucket for model)")

        if not config.minio.object or not config.minio.object.strip():
            missing_configs.append("  - minio.object (MinIO object path for model)")
    
    # Validate image size
    if config.model.image_size <= 0:
        missing_configs.append("  - model.image_size (Must be > 0)")
    
    is_valid = len(missing_configs) == 0
    return is_valid, missing_configs


def validate_startup_configuration() -> None:
    """
    Validate all required environment variables and config values.
    Exits with sys.exit(1) if validation fails.
    """
    logger.info("Validating startup configuration...")
    
    # Validate environment variables
    env_valid, missing_env_vars = validate_environment_variables()
    
    # Validate config values
    config_valid, missing_configs = validate_config_values()
    
    # Check if validation passed
    if env_valid and config_valid:
        logger.info("Configuration validation passed ✓")
        return
    
    # Log errors
    logger.error("Configuration validation failed:")
    
    if not env_valid:
        logger.error("Missing required environment variables:")
        for var in missing_env_vars:
            logger.error(var)
    
    if not config_valid:
        logger.error("Missing required configuration values:")
        for config_item in missing_configs:
            logger.error(config_item)
    
    logger.error("Please set all required environment variables and config values before starting the application.")
    sys.exit(1)
