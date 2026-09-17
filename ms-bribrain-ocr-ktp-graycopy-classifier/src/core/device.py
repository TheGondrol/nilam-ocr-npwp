"""
Device detection and initialization utility
Handles GPU/CPU detection with automatic fallback
"""

import torch

from src.core.logging import get_logger

logger = get_logger()


def get_device(prefer_gpu: bool = True, force_cpu: bool = False) -> torch.device:
    """
    Get appropriate device (GPU or CPU) based on availability and preference.

    Args:
        prefer_gpu: Whether to prefer GPU if available (default: True)
        force_cpu: Force CPU usage even if GPU is available (default: False)

    Returns:
        torch.device: CUDA device if available and preferred, otherwise CPU
    """
    if force_cpu:
        device = torch.device("cpu")
        logger.info("Forcing CPU usage as configured")
    elif prefer_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"Using GPU: {gpu_name}")
        logger.info(f"GPU Memory: {gpu_memory:.2f} GB")
        logger.info(f"CUDA Version: {torch.version.cuda}")
    else:
        device = torch.device("cpu")
        if prefer_gpu and not torch.cuda.is_available():
            logger.warning("GPU preferred but CUDA not available, using CPU")
        else:
            logger.info("Using CPU as configured")

    logger.info(f"Device initialized: {device}")
    return device


def is_cuda_available() -> bool:
    """
    Check if CUDA is available.

    Returns:
        bool: True if CUDA is available, False otherwise
    """
    return torch.cuda.is_available()


def get_device_info(device: torch.device) -> dict:
    """
    Get detailed device information.

    Args:
        device: torch.device object

    Returns:
        dict: Device information including type, name, memory (if GPU)
    """
    info = {
        "type": str(device),
        "is_cuda": device.type == "cuda",
    }

    if device.type == "cuda":
        info["name"] = torch.cuda.get_device_name(0)
        info["memory_gb"] = torch.cuda.get_device_properties(0).total_memory / 1024**3
        info["cuda_version"] = torch.version.cuda

    return info
