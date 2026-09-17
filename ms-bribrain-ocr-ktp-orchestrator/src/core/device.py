"""
Device detection and selection module.

Detects GPU availability and provides CPU fallback support.
"""

import logging
from typing import Literal

from src.core.config import get_settings

logger = logging.getLogger(__name__)

DeviceType = Literal["cuda", "mps", "cpu"]


def detect_gpu() -> tuple[bool, DeviceType]:
    """
    Detect if GPU is available.
    
    Returns:
        Tuple of (is_available, device_type).
        - is_available: True if GPU is detected
        - device_type: "cuda" for NVIDIA, "mps" for Apple Silicon, "cpu" otherwise
    """
    # Try CUDA (NVIDIA GPU)
    try:
        import torch  # ty: ignore[unresolved-import]
        if torch.cuda.is_available():
            return True, "cuda"
    except ImportError:
        pass
    except Exception as e:
        logger.debug(f"CUDA check failed: {e}")

    # Try MPS (Apple Silicon GPU)
    try:
        import torch  # ty: ignore[unresolved-import]
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return True, "mps"
    except ImportError:
        pass
    except Exception as e:
        logger.debug(f"MPS check failed: {e}")
    
    # No GPU available
    return False, "cpu"


def get_device() -> DeviceType:
    """
    Get the device to use for computation.
    
    Respects configuration settings for GPU preference and CPU fallback.
    
    Returns:
        Device type: "cuda", "mps", or "cpu"
    """
    settings = get_settings()
    device_config = settings.device
    
    gpu_available, gpu_type = detect_gpu()
    
    if device_config.prefer_gpu and gpu_available:
        logger.info(f"GPU detected: {gpu_type}. Using GPU for computation.")
        return gpu_type
    elif device_config.prefer_gpu and not gpu_available:
        if device_config.fallback_to_cpu:
            logger.warning(
                "GPU preferred but not available. Falling back to CPU."
            )
            return "cpu"
        else:
            logger.error(
                "GPU preferred but not available, and CPU fallback is disabled."
            )
            raise RuntimeError(
                "GPU not available and CPU fallback is disabled in configuration."
            )
    else:
        logger.info("Using CPU for computation (as configured).")
        return "cpu"


def log_device_info():
    """Log information about the selected device."""
    try:
        device = get_device()
        gpu_available, gpu_type = detect_gpu()
        
        logger.info(f"Selected device: {device}")
        if gpu_available:
            logger.info(f"GPU available: {gpu_type}")
        else:
            logger.info("No GPU detected")
            
    except Exception as e:
        logger.error(f"Error detecting device: {e}")
