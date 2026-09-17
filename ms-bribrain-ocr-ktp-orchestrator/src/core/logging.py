"""
Logging configuration module.

Sets up structured logging with file rotation and console output.
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

from src.core.config import get_settings


_root_configured = False


def setup_logging(logger_name: Optional[str] = None) -> logging.Logger:
    """
    Set up logging with file rotation and console output.

    Configures the root logger so all child loggers (created via
    ``logging.getLogger(__name__)``) inherit the handlers automatically.

    Args:
        logger_name: Name for the returned logger. If None, returns root logger.

    Returns:
        Configured logger instance.
    """
    global _root_configured
    settings = get_settings()
    log_config = settings.logging
    level = getattr(logging, log_config.level)

    if not _root_configured:
        # Create formatter
        formatter = logging.Formatter(log_config.format)

        # Create logs directory if it doesn't exist
        log_file_path = Path(log_config.file)
        log_file_path.parent.mkdir(parents=True, exist_ok=True)

        # File handler with rotation
        file_handler = RotatingFileHandler(
            log_config.file,
            maxBytes=log_config.max_bytes,
            backupCount=log_config.backup_count
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)

        # Configure root logger so ALL child loggers inherit these handlers
        root = logging.getLogger()
        root.setLevel(level)
        root.handlers.clear()
        root.addHandler(file_handler)

        # Console handler (for Docker/development)
        if log_config.console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(level)
            console_handler.setFormatter(formatter)
            root.addHandler(console_handler)

        _root_configured = True

    return logging.getLogger(logger_name)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific module.

    Args:
        name: Module name (typically __name__).

    Returns:
        Configured logger instance.
    """
    return logging.getLogger(name)
