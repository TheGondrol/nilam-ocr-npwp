"""Logging configuration module.

This module sets up logging with file and console handlers,
including log rotation and structured formatting.
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

from src.core.config import get_config
from contextvars import ContextVar

# ============================================================================
# Request Context Management
# ============================================================================

request_id_ctx: ContextVar[str] = ContextVar("request_id", default="-")
"""
Context variable for storing request_id.
Uses contextvars for async-safe storage across the request lifecycle.
Default value is "-" when no request is active.
"""


# ============================================================================
# Logging Filter
# ============================================================================

class RequestIdFilter(logging.Filter):
    """
    Logging filter that automatically injects request_id into log records.
    
    This filter retrieves the request_id from the context variable and
    adds it to each log record, making it available in the log format.
    """
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Add request_id from context to the log record."""
        record.request_id = request_id_ctx.get()
        return True

def setup_logging() -> None:
    """Set up logging configuration based on config.yaml settings."""
    config = get_config()
    log_config = config.get_logging_config()

    # Get logging settings
    log_level = getattr(logging, log_config.get("level", "INFO").upper())
    log_format = log_config.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    date_format = log_config.get("date_format", "%Y-%m-%d %H:%M:%S")

    # Create formatter
    formatter = logging.Formatter(log_format, datefmt=date_format)

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Remove existing handlers
    root_logger.handlers.clear()

    # Console handler
    console_config = log_config.get("console", {})
    if console_config.get("enabled", True):
        console_handler = logging.StreamHandler(sys.stdout)
        console_level = getattr(
            logging,
            console_config.get("level", "INFO").upper()
        )
        console_handler.setLevel(console_level)
        console_handler.setFormatter(formatter)
        console_handler.addFilter(RequestIdFilter())
        root_logger.addHandler(console_handler)

    # File handler with rotation
    file_config = log_config.get("file", {})
    if file_config.get("enabled", True):
        log_file_path = file_config.get("path", "logs/app.log")
        
        # Create logs directory if it doesn't exist
        log_dir = Path(log_file_path).parent
        log_dir.mkdir(parents=True, exist_ok=True)

        # Create rotating file handler
        max_bytes = file_config.get("max_bytes", 10485760)  # 10MB default
        backup_count = file_config.get("backup_count", 5)

        file_handler = RotatingFileHandler(
            log_file_path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8"
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        file_handler.addFilter(RequestIdFilter())
        root_logger.addHandler(file_handler)

    # Log startup message
    logger = logging.getLogger(__name__)
    logger.info("Logging system initialized")
    logger.info(f"Log level: {logging.getLevelName(log_level)}")
    if file_config.get("enabled", True):
        logger.info(f"Log file: {log_file_path}")


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Get a logger instance.

    Args:
        name: Logger name (typically __name__ of the calling module)

    Returns:
        Logger instance
    """
    return logging.getLogger(name)
