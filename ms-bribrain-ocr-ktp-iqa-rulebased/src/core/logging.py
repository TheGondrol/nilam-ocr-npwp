"""
Logging configuration for OCR Quality Service.
Sets up file and console logging with rotation.
"""
import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
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


class RequestIdFormatter(logging.Formatter):
    """
    Custom formatter that ensures request_id is always present in log records.
    Injects request_id from context variable at format time as a safety net.
    """

    def format(self, record: logging.LogRecord) -> str:
        if not hasattr(record, 'request_id'):
            record.request_id = request_id_ctx.get()
        return super().format(record)


def setup_logging(name: Optional[str] = None) -> logging.Logger:
    """
    Set up logging with file and console handlers.
    
    Args:
        name: Logger name (defaults to root logger)
        
    Returns:
        logging.Logger: Configured logger instance
    """
    config = get_config()
    log_config = config.logging

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_config.level.upper()))

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    # Create formatter with built-in request_id injection
    formatter = RequestIdFormatter(log_config.format)

    # Set up file handler if enabled
    if log_config.file.enabled:
        # Ensure log directory exists
        log_path = Path(log_config.file.path)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = RotatingFileHandler(
            log_config.file.path,
            maxBytes=log_config.file.max_bytes,
            backupCount=log_config.file.backup_count
        )
        file_handler.setLevel(getattr(logging, log_config.level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Set up console handler if enabled
    if log_config.console.enabled:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_config.level.upper()))
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the specified name.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        logging.Logger: Logger instance
    """
    logger = logging.getLogger(name)

    # Ensure the filter is added if not already present
    if not any(isinstance(f, RequestIdFilter) for f in logger.filters):
        logger.addFilter(RequestIdFilter())

    return logger
