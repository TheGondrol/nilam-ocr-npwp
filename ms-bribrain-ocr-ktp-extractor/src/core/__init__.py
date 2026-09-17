"""Core module - Configuration, logging, and device detection"""
from src.core.config import settings
from src.core.device import detect_device, get_use_gpu

__all__ = ["settings", "detect_device", "get_use_gpu"]
