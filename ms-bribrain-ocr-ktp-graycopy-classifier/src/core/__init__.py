"""
Core module for OCR Graycopy service.
Contains configuration, logging, and device utilities.
"""

from src.core.config import Config, get_config, load_config
from src.core.device import get_device, get_device_info, is_cuda_available
from src.core.logging import get_logger, request_id_ctx, setup_logging

__all__ = [
    "Config",
    "get_config",
    "load_config",
    "setup_logging",
    "get_logger",
    "request_id_ctx",
    "get_device",
    "is_cuda_available",
    "get_device_info",
]
