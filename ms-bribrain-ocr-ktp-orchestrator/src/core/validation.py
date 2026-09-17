"""
Configuration validation module.
Validates required environment variables and config values before application startup.
"""

import logging
import os
import sys
from typing import List, Tuple

from src.core.config import get_settings

logger = logging.getLogger(__name__)


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
        "OCR_SERVICE_API",
        "LAMINATION_SERVICE_API",
        "RECAPTURE_SERVICE_API",
        "GRAYCOPY_SERVICE_API",
        "TEMPER_SERVICE_API",
        "CLASSIFIER_SERVICE_API",
        "QUALITY_SERVICE_API",
        "QUALITYDL_SERVICE_API",
        "POSTPROCESS_SERVICE_API",
        "ORCHESTRATOR_SERVICE_API",
    ]

    # Encryption key is required whenever request logs are persisted (DPIA).
    if get_settings().logging.log_to_database:
        required_env_vars.append("LOG_ENCRYPTION_KEY")

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
    settings = get_settings()
    
    missing_configs = []
    
    # Validate app settings
    if not settings.app.host or not settings.app.host.strip():
        missing_configs.append("  - app.host (Application host)")
    
    if not settings.app.port:
        missing_configs.append("  - app.port (Application port)")
    
    # Validate service URLs (only for enabled services)
    if settings.run_services.ocr and (not settings.services.ocr.url or not settings.services.ocr.url.strip()):
        missing_configs.append("  - services.ocr.url (OCR service URL)")
    
    if settings.run_services.quality and (not settings.services.quality.url or not settings.services.quality.url.strip()):
        missing_configs.append("  - services.quality.url (Quality service URL)")
    
    if settings.run_services.postprocess and (not settings.services.postprocess.url or not settings.services.postprocess.url.strip()):
        missing_configs.append("  - services.postprocess.url (Postprocess service URL)")
    
    if settings.run_services.lamination and (not settings.services.lamination.url or not settings.services.lamination.url.strip()):
        missing_configs.append("  - services.lamination.url (Lamination service URL)")
    
    if settings.run_services.recapture and (not settings.services.recapture.url or not settings.services.recapture.url.strip()):
        missing_configs.append("  - services.recapture.url (Recapture service URL)")
    
    if settings.run_services.graycopy and (not settings.services.graycopy.url or not settings.services.graycopy.url.strip()):
        missing_configs.append("  - services.graycopy.url (Graycopy service URL)")
    
    if settings.run_services.qualitydl and (not settings.services.qualitydl.url or not settings.services.qualitydl.url.strip()):
        missing_configs.append("  - services.qualitydl.url (QualityDL service URL)")
    
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
        logger.info("Configuration validation passed [OK]")
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
