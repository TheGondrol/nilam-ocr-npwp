"""
Configuration validation module.
Validates required environment variables and config values before application startup.
"""

import logging
import os
import sys
from pathlib import Path
from typing import List, Tuple

from src.core.config import settings

logger = logging.getLogger(__name__)


def validate_environment_variables() -> Tuple[bool, List[str]]:
    """
    Validate required environment variables are set.
    Checks GCS variables only when APP_ENVIRO is not 'onprem'.

    Returns:
        Tuple of (is_valid, missing_vars)
    """
    app_enviro = os.getenv("APP_ENVIRO", "onprem")

    required_env_vars = [
        "API_KEY",
    ]

    # GCS credentials come from ADC (the VM's attached service account) or
    # optional SA_* env vars (see gcs_service._load_credentials); not required
    # at startup.

    # Encryption key is required whenever request logs are persisted (DPIA).
    if settings.log_to_database:
        required_env_vars.append("LOG_ENCRYPTION_KEY")

    missing_vars = []

    for var in required_env_vars:
        value = os.getenv(var)
        if not value or value.strip() == "":
            missing_vars.append(f"  - {var}")

    return len(missing_vars) == 0, missing_vars


def validate_config_values() -> Tuple[bool, List[str]]:
    """
    Validate required config values are set.
    
    Returns:
        Tuple of (is_valid, missing_configs)
    """
    required_configs = [
        ("server.host", "Server host", settings.server_host),
        ("server.port", "Server port", settings.server_port),
        ("ocr.server_config_path", "OCR server config path", settings.ocr_server_config_path),
        ("ocr.mobile_config_path", "OCR mobile config path", settings.ocr_mobile_config_path),
    ]
    
    missing_configs = []
    
    for config_path, description, value in required_configs:
        if value is None or (isinstance(value, str) and not value.strip()):
            missing_configs.append(f"  - {config_path} ({description})")
    
    # Validate OCR config files exist
    ocr_configs = [
        (settings.ocr_server_config_path, "OCR server config"),
        (settings.ocr_mobile_config_path, "OCR mobile config"),
    ]
    
    for config_file, description in ocr_configs:
        if not Path(config_file).exists():
            missing_configs.append(f"  - {config_file} ({description} file not found)")
    
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
        for config in missing_configs:
            logger.error(config)
    
    logger.error("Please set all required environment variables and config values before starting the application.")
    sys.exit(1)
