"""
Core module for configuration, logging, and device management
"""

from .config import config
from .logging import setup_logging
from .device import get_device

__all__ = ["config", "setup_logging", "get_device"]
