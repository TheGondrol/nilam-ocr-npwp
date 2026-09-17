"""Logging configuration with request ID context tracking."""

from __future__ import annotations

import logging
import sys
from contextvars import ContextVar
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

from src.core.config import config

# Context variable for request ID tracking across async operations
request_id_ctx: ContextVar[str] = ContextVar("request_id", default="-")


class RequestIdFilter(logging.Filter):
    """Filter to inject request_id into log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        """
        Add request_id to log record.

        Args:
            record: Log record to modify

        Returns:
            True to include the record
        """
        record.request_id = request_id_ctx.get()
        return True


def setup_logging() -> logging.Logger:
    """
    Setup logging with console and file handlers.

    Configures the root logger so all child loggers (created via
    ``logging.getLogger(__name__)``) inherit the handlers automatically.

    Returns:
        Configured logger instance (named logger for the application)
    """
    logger_name = config.get("logging.logger_name", "quality")
    log_level = config.get("logging.level", "INFO")
    log_format = config.get(
        "logging.format",
        "%(asctime)s - %(name)s - %(levelname)s - [%(request_id)s] - %(message)s",
    )

    level = getattr(logging, log_level.upper(), logging.INFO)

    # Create formatter
    formatter = logging.Formatter(log_format)

    # Create request ID filter
    request_filter = RequestIdFilter()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)
    console_handler.addFilter(request_filter)

    # File handler (if enabled)
    file_handler = None
    if config.get("logging.file.enabled", True):
        log_file_path = config.get("logging.file.path", "logs/ocr_quality.log")
        log_path = Path(log_file_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        max_bytes = config.get("logging.file.max_bytes", 10485760)  # 10MB
        backup_count = config.get("logging.file.backup_count", 5)

        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        file_handler.addFilter(request_filter)

    # Configure root logger so ALL child loggers inherit these handlers
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()
    root.addHandler(console_handler)
    if file_handler:
        root.addHandler(file_handler)

    # Return the named logger for the application
    logger = logging.getLogger(logger_name)
    logger.info(f"Logging initialized with level: {log_level}")
    return logger
