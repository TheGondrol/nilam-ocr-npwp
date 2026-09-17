"""
Standardized logging system with request ID correlation and structured output.

This module provides a comprehensive logging setup with:
- File rotation based on size and count
- Structured JSON logging for production
- Request ID correlation for tracing
- Separate error log files
- Performance timing decorators

Usage:
    from src.core.logging import setup_logging, get_logger, log_performance
    
    # Setup logging (call once at application startup)
    setup_logging()
    
    # Get logger for your module
    logger = get_logger(__name__)
    logger.info("Processing started", extra={"user_id": 123})
    
    # Use performance decorator
    @log_performance
    async def process_data(data):
        # ... processing logic
        pass
"""

import inspect
import json
import logging
import logging.handlers
import sys
import time
from contextvars import ContextVar
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

from src.core.config import get_config

request_id_ctx: ContextVar[str] = ContextVar("request_id", default="-")
"""
Context variable for storing request_id.
Uses contextvars for async-safe storage across the request lifecycle.
Default value is "-" when no request is active.
"""


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


class JSONFormatter(logging.Formatter):
    """Format log records as JSON for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON string."""
        log_data: Dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add request ID if available
        if hasattr(record, "request_id") and record.request_id != "N/A":
            log_data["request_id"] = record.request_id
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields from log call
        if hasattr(record, "extra_data"):
            log_data["extra"] = record.extra_data
        
        return json.dumps(log_data)


class TextFormatter(logging.Formatter):
    """Format log records as human-readable text."""
    
    def __init__(self, include_request_id: bool = True):
        """Initialize text formatter."""
        if include_request_id:
            fmt = "[%(asctime)s] [%(request_id)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s"
        else:
            fmt = "[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s"
        
        super().__init__(fmt=fmt, datefmt="%Y-%m-%d %H:%M:%S")


def setup_logging(config_path: str = "config.yaml") -> None:
    """
    Setup logging system based on configuration.
    
    Args:
        config_path: Path to configuration file
    """
    config = get_config(config_path)
    log_config = config.logging
    
    # Create logs directory if it doesn't exist
    log_dir = Path(log_config.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine log level
    log_level = getattr(logging, log_config.level.upper(), logging.INFO)
    
    # Create formatters
    if log_config.format.lower() == "json":
        formatter: Union[JSONFormatter, TextFormatter] = JSONFormatter()
    else:
        formatter = TextFormatter(include_request_id=log_config.include_request_id)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    console_handler.addFilter(RequestIdFilter())
    
    # Application log file handler (rotating)
    app_log_path = log_dir / log_config.app_log_file
    max_bytes = log_config.max_file_size_mb * 1024 * 1024  # Convert MB to bytes
    
    app_file_handler = logging.handlers.RotatingFileHandler(
        filename=app_log_path,
        maxBytes=max_bytes,
        backupCount=log_config.backup_count,
        encoding="utf-8",
    )
    app_file_handler.setLevel(log_level)
    app_file_handler.setFormatter(formatter)
    app_file_handler.addFilter(RequestIdFilter())
    
    # Error log file handler (only ERROR and above)
    error_log_path = log_dir / log_config.error_log_file
    error_file_handler = logging.handlers.RotatingFileHandler(
        filename=error_log_path,
        maxBytes=max_bytes,
        backupCount=log_config.backup_count,
        encoding="utf-8",
    )
    error_file_handler.setLevel(logging.ERROR)
    error_file_handler.setFormatter(formatter)
    error_file_handler.addFilter(RequestIdFilter())
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Remove existing handlers (close file handlers to avoid resource warnings)
    for handler in root_logger.handlers[:]:
        handler.close()
        root_logger.removeHandler(handler)
    
    # Add handlers
    root_logger.addHandler(console_handler)
    root_logger.addHandler(app_file_handler)
    root_logger.addHandler(error_file_handler)
    
    # Suppress noisy third-party loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("multipart").setLevel(logging.WARNING)
    
    root_logger.info(f"Logging initialized: level={log_config.level}, format={log_config.format}")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for the specified module.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        logging.Logger: Logger instance
    """
    return logging.getLogger(name)


def set_request_id(request_id: str) -> None:
    """
    Set request ID in context for current request.
    
    Args:
        request_id: Unique request identifier
    """
    request_id_ctx.set(request_id)


def get_request_id() -> Optional[str]:
    """
    Get current request ID from context.
    
    Returns:
        Optional[str]: Request ID if set, None otherwise
    """
    return request_id_ctx.get()


def log_performance(func: Callable) -> Callable:
    """
    Decorator to log function performance timing.
    
    Usage:
        @log_performance
        async def my_async_function():
            # ... logic
            pass
            
        @log_performance
        def my_sync_function():
            # ... logic
            pass
    
    Args:
        func: Function to decorate
        
    Returns:
        Callable: Wrapped function with performance logging
    """
    logger = get_logger(getattr(func, "__module__", __name__))
    func_name = getattr(func, "__name__", repr(func))

    # Check if function is async
    if inspect.iscoroutinefunction(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                logger.info(
                    f"Function {func_name} completed",
                    extra={"duration_seconds": round(duration, 3), "function": func_name}
                )
                return result
            except Exception as e:
                duration = time.time() - start_time
                logger.error(
                    f"Function {func_name} failed",
                    extra={"duration_seconds": round(duration, 3), "function": func_name, "error": str(e)},
                    exc_info=True
                )
                raise
        return async_wrapper
    else:
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                logger.info(
                    f"Function {func_name} completed",
                    extra={"duration_seconds": round(duration, 3), "function": func_name}
                )
                return result
            except Exception as e:
                duration = time.time() - start_time
                logger.error(
                    f"Function {func_name} failed",
                    extra={"duration_seconds": round(duration, 3), "function": func_name, "error": str(e)},
                    exc_info=True
                )
                raise
        return sync_wrapper


class LogContext:
    """Context manager for adding extra context to logs."""
    
    def __init__(self, logger: logging.Logger, **context):
        """
        Initialize log context.
        
        Args:
            logger: Logger instance
            **context: Key-value pairs to add to log context
        """
        self.logger = logger
        self.context = context
        self.original_log_methods: Dict[str, Callable[..., Any]] = {}
    
    def __enter__(self):
        """Enter context - wrap log methods to add context."""
        for method_name in ["debug", "info", "warning", "error", "critical"]:
            original_method = getattr(self.logger, method_name)
            self.original_log_methods[method_name] = original_method
            
            def make_wrapper(orig_method):
                def wrapper(msg, *args, **kwargs):
                    if "extra" not in kwargs:
                        kwargs["extra"] = {}
                    kwargs["extra"].update(self.context)
                    return orig_method(msg, *args, **kwargs)
                return wrapper
            
            setattr(self.logger, method_name, make_wrapper(original_method))
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context - restore original log methods."""
        for method_name, original_method in self.original_log_methods.items():
            setattr(self.logger, method_name, original_method)
