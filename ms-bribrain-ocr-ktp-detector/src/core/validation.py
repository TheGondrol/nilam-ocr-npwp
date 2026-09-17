"""
Configuration validation module.

Validates that all required environment variables and config values are set.
"""

import os
import sys
from typing import List, Tuple

from .config import config
from .logging import logger


def validate_environment_variables() -> Tuple[bool, List[str]]:
    """
    Validate required environment variables are set.
    
    Returns:
        Tuple of (is_valid, missing_vars)
    """
    required_env_vars = [
        "MINIO_ENDPOINT",
        "MINIO_ACCESS_KEY",
        "MINIO_SECRET_KEY",
        "DATABASE_URL",
        "API_KEY",
    ]
    
    missing_vars = []
    for var in required_env_vars:
        value = os.getenv(var)
        if not value or value.strip() == "":
            missing_vars.append(var)
    
    return len(missing_vars) == 0, missing_vars


def validate_config_values() -> Tuple[bool, List[str]]:
    """
    Validate required config values are set.
    
    Returns:
        Tuple of (is_valid, missing_configs)
    """
    required_configs = [
        ("model.path", "Model path"),
        ("model.confidence", "Model confidence threshold"),
        ("model.iou_threshold", "Model IOU threshold"),
        ("server.host", "Server host"),
        ("server.port", "Server port"),
        ("minio.bucket", "MinIO bucket"),
        ("minio.object", "MinIO object path"),
    ]
    
    missing_configs = []
    for config_key, description in required_configs:
        value = config.get(config_key)
        if value is None or (isinstance(value, str) and value.strip() == ""):
            missing_configs.append(f"{config_key} ({description})")
    
    return len(missing_configs) == 0, missing_configs


def validate_startup_configuration() -> None:
    """
    Validate all required environment variables and config values.
    
    Raises:
        SystemExit: If validation fails
    """
    logger.info("Validating startup configuration...")
    
    errors = []
    
    # Validate environment variables
    env_valid, missing_env = validate_environment_variables()
    if not env_valid:
        errors.append("Missing required environment variables:")
        for var in missing_env:
            errors.append(f"  - {var}")
    
    # Validate config values
    config_valid, missing_config = validate_config_values()
    if not config_valid:
        errors.append("Missing required configuration values:")
        for cfg in missing_config:
            errors.append(f"  - {cfg}")
    
    # If there are errors, log and exit
    if errors:
        logger.error("Configuration validation failed:")
        for error in errors:
            logger.error(error)
        logger.error("Please set all required environment variables and config values before starting the application.")
        sys.exit(1)
    
    logger.info("Configuration validation passed ✓")
