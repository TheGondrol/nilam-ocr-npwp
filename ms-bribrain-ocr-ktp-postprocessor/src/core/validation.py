"""
Configuration validation module.
Validates required environment variables and config values before application startup.
"""

import logging
import os
import sys
from typing import List, Tuple

from src.core.config import get_config

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
        "API_KEY",
    ]

    # Encryption key is required whenever request logs are persisted (DPIA).
    if get_config().get_logging_config().get("insert_to_database", False):
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
    config = get_config()
    
    missing_configs = []
    
    # Validate API settings
    api_config = config.get_api_config()
    _host = api_config.get("host")
    if not _host or not str(_host).strip():
        missing_configs.append("  - api.host (API host)")
    
    if not api_config.get("port"):
        missing_configs.append("  - api.port (API port)")
    
    # Validate thresholds
    thresholds = config.get_thresholds()
    if not thresholds:
        missing_configs.append("  - thresholds (Threshold configuration)")
    else:
        if "partial" not in thresholds:
            missing_configs.append("  - thresholds.partial (Partial threshold)")
        if "ratio" not in thresholds:
            missing_configs.append("  - thresholds.ratio (Ratio threshold)")
        if "confidence" not in thresholds:
            missing_configs.append("  - thresholds.confidence (Confidence threshold)")
    
    # Validate database settings
    db_config = config.get_database_config()
    _table_name = db_config.get("table_name")
    if not _table_name or not str(_table_name).strip():
        missing_configs.append("  - database.table_name (Database table name)")
    
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
