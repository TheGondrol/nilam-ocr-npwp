"""
Configuration validation module.
Validates required environment variables and config values before application startup.
"""

import os
import sys
from typing import List, Tuple

from src.core.config import get_config
from src.core.logging import get_logger

logger = get_logger(__name__)


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
    if get_config().logging.log_to_database:
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
    
    # Validate server settings
    if not config.server.host or not config.server.host.strip():
        missing_configs.append("  - server.host (Server host)")
    
    if not config.server.port:
        missing_configs.append("  - server.port (Server port)")
    
    # Validate database settings
    if not config.database.table_name or not config.database.table_name.strip():
        missing_configs.append("  - database.table_name (Database table name)")
    
    if not config.database.db_schema or not config.database.db_schema.strip():
        missing_configs.append("  - database.schema (Database schema)")
    
    # Validate quality thresholds
    if config.quality.blur.threshold <= 0:
        missing_configs.append("  - quality.blur.threshold (Must be > 0)")
    
    if config.quality.confidence.threshold_median < 0 or config.quality.confidence.threshold_median > 1:
        missing_configs.append("  - quality.confidence.threshold_median (Must be between 0 and 1)")
    
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
        logger.info("Configuration validation passed")
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
