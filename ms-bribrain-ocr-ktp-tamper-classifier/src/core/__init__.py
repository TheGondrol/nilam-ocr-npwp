"""Core module initialization"""

from src.core.config import config
from src.core.device import cleanup_device_memory, get_device, get_device_manager
from src.core.logging import get_logger, setup_logging

__all__ = ['config', 'setup_logging', 'get_logger', 'get_device', 'get_device_manager', 'cleanup_device_memory']
