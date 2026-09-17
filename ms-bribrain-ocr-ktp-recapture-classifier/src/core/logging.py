"""
Logging Configuration Module
Sets up logging with both console and file handlers
"""

import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
from typing import Optional

from .config import config
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

def setup_logging(name: Optional[str] = None) -> logging.Logger:
    """
    Setup logging with console and file handlers.

    Configures the root logger so all child loggers (created via
    ``logging.getLogger(__name__)``) inherit the handlers automatically.

    Args:
        name: Logger name for the returned logger instance.

    Returns:
        Configured logger instance
    """
    # Get logging configuration
    log_level = config.get('logging.level', 'INFO')
    log_format = config.get('logging.format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    level = getattr(logging, log_level.upper())

    # Create formatter
    formatter = logging.Formatter(log_format)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    console_handler.addFilter(RequestIdFilter())

    # File handler (if enabled)
    file_handler = None
    if config.get('logging.file.enabled', True):
        try:
            # Create logs directory
            log_dir = Path(config.get('logging.file.directory', './logs'))
            log_dir.mkdir(parents=True, exist_ok=True)

            # Generate log filename
            log_filename = config.get('logging.file.filename', 'ocr_recapture.log')
            log_file = log_dir / log_filename

            # Create rotating file handler
            max_bytes = config.get('logging.file.max_bytes', 10485760)  # 10MB default
            backup_count = config.get('logging.file.backup_count', 5)

            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding='utf-8'
            )
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            file_handler.addFilter(RequestIdFilter())

        except Exception as e:
            logging.warning(f"Failed to setup file logging: {str(e)}")

    # Configure root logger so ALL child loggers inherit these handlers
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()
    root.addHandler(console_handler)
    if file_handler:
        root.addHandler(file_handler)

    # Return the named logger for the application
    logger = logging.getLogger(name)
    return logger


# Create default logger for the application
logger = setup_logging(config.get('logging.logger_name', 'recapture'))
