"""Device detection and configuration"""

import logging
from typing import Optional
from src.core.config import settings
from src.core.logging import setup_logging

logger = logging.getLogger(__name__)

# Cache for device detection result
_detected_device: Optional[str] = None


def detect_device() -> str:
    """
    Detect available device (GPU/CPU) and return the appropriate configuration.

    The result is cached after first detection to avoid repeated checks.

    Returns:
        str: "gpu" if GPU is available and preferred, "cpu" otherwise
    """
    global _detected_device, logger

    # Return cached result if available
    if _detected_device is not None:
        return _detected_device

    # Check if CPU is forced
    if settings.device_force_cpu:
        logger.info("CPU mode forced by configuration")
        _detected_device = "cpu"
        return _detected_device

    # Check device preference
    preferred = settings.device_preferred.lower()

    if preferred == "cpu":
        logger.info("CPU mode selected by configuration")
        _detected_device = "cpu"
        return _detected_device

    # Try to detect GPU availability
    try:
        import paddle
        setup_logging()
        logger = logging.getLogger(__name__)
        # Check if CUDA is available
        if paddle.device.is_compiled_with_cuda():
            gpu_count = paddle.device.cuda.device_count()
            if gpu_count > 0:
                logger.info(f"GPU detected: {gpu_count} CUDA device(s) available")

                # If preferred is "gpu" or "auto", use GPU
                if preferred in ["gpu", "auto"]:
                    logger.info("Using GPU for inference")
                    _detected_device = "gpu"
                else:
                    logger.warning(f"Unknown device preference: {preferred}, defaulting to CPU")
                    _detected_device = "cpu"
            else:
                logger.warning("CUDA is compiled but no GPU devices found, falling back to CPU")
                _detected_device = "cpu"
        else:
            logger.info("PaddlePaddle not compiled with CUDA, using CPU")
            _detected_device = "cpu"

    except ImportError:
        logger.info("PaddlePaddle not installed, checking torch for GPU detection")
        try:
            import torch
            if torch.cuda.is_available() and torch.cuda.device_count() > 0 and preferred in ("gpu", "auto"):
                logger.info(f"GPU detected via torch: {torch.cuda.device_count()} CUDA device(s)")
                _detected_device = "gpu"
            else:
                _detected_device = "cpu"
        except ImportError:
            logger.warning("Neither PaddlePaddle nor torch installed, defaulting to CPU")
            _detected_device = "cpu"
    except Exception as e:
        logger.error(f"Error detecting device: {str(e)}, falling back to CPU")
        _detected_device = "cpu"

    return _detected_device


def get_use_gpu() -> bool:
    """
    Get whether to use GPU for PaddleOCR.

    Returns:
        bool: True if GPU should be used, False for CPU
    """
    device = detect_device()
    return device == "gpu"


def reset_device_cache() -> None:
    """
    Reset the device detection cache.

    Useful for testing or when device configuration changes.
    """
    global _detected_device
    _detected_device = None

