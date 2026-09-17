"""
Device detection and management utilities.
Handles GPU/CPU detection and fallback logic.
"""
from typing import Dict, Any

from src.core.config import get_config
from src.core.logging import get_logger

logger = get_logger(__name__)


def get_device_info() -> Dict[str, Any]:
    """
    Get information about available compute devices.
    
    Returns:
        Dict containing device information
    """
    config = get_config()
    device_config = config.device

    device_info = {
        "prefer_gpu": device_config.prefer_gpu,
        "fallback_to_cpu": device_config.fallback_to_cpu,
        "gpu_available": False,
        "selected_device": "cpu"
    }

    # Try to detect GPU availability
    try:
        import torch  # type: ignore[import-not-found]

        # Check for CUDA (NVIDIA GPU)
        if torch.cuda.is_available():
            device_info["gpu_available"] = True
            device_info["gpu_type"] = "cuda"
            device_info["gpu_name"] = torch.cuda.get_device_name(0)
            logger.info(f"CUDA GPU detected: {device_info['gpu_name']}")

        # Check for Metal (Apple Silicon)
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device_info["gpu_available"] = True
            device_info["gpu_type"] = "mps"
            device_info["gpu_name"] = "Apple Silicon GPU"
            logger.info("Apple Silicon GPU (Metal) detected")

    except ImportError:
        pass

    except Exception as e:
        logger.warning(f"Error detecting GPU: {e}")

    # Determine selected device
    if device_config.prefer_gpu and device_info["gpu_available"]:
        device_info["selected_device"] = "gpu"
        logger.info(f"Using GPU: {device_info.get('gpu_name', 'Unknown')}")
    else:
        device_info["selected_device"] = "cpu"
        if device_config.prefer_gpu and not device_info["gpu_available"]:
            logger.info("GPU preferred but not available, falling back to CPU")
        else:
            logger.info("Using CPU")

    return device_info


def should_use_gpu() -> bool:
    """
    Determine if GPU should be used based on configuration and availability.
    
    Returns:
        bool: True if GPU should be used, False otherwise
    """
    device_info = get_device_info()
    return device_info["selected_device"] == "gpu"
