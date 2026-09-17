"""Logging configuration"""
import logging
import logging.handlers
from pathlib import Path
from src.core.config import settings
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
    """
    Configure application logging based on settings from config.yaml
    
    Sets up:
    - Console logging (if enabled)
    - File logging with rotation (if enabled)
    - Log format and levels
    """
    # Create logs directory if it doesn't exist
    if settings.log_file_enabled:
        log_path = Path(settings.log_file_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # Set to DEBUG to capture all levels
    
    # Remove existing handlers to avoid duplicates
    root_logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        fmt=settings.log_format,
        datefmt=settings.log_date_format
    )
    
    # Console handler
    if settings.log_console_enabled:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, settings.log_console_level))
        console_handler.setFormatter(formatter)
        console_handler.addFilter(RequestIdFilter())
        root_logger.addHandler(console_handler)
    
    # File handler with rotation
    if settings.log_file_enabled:
        file_handler = logging.handlers.RotatingFileHandler(
            filename=settings.log_file_path,
            maxBytes=settings.log_file_max_bytes,
            backupCount=settings.log_file_backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(getattr(logging, settings.log_file_level))
        file_handler.setFormatter(formatter)
        file_handler.addFilter(RequestIdFilter())
        root_logger.addHandler(file_handler)
    
    # Set the overall log level
    root_logger.setLevel(getattr(logging, settings.log_level))
    
    # Log initial message
    logger = logging.getLogger(__name__)
    logger.info("Logging system initialized")
    logger.info(f"Console logging: {'enabled' if settings.log_console_enabled else 'disabled'}")
    logger.info(f"File logging: {'enabled' if settings.log_file_enabled else 'disabled'}")
    if settings.log_file_enabled:
        logger.info(f"Log file: {settings.log_file_path}")
