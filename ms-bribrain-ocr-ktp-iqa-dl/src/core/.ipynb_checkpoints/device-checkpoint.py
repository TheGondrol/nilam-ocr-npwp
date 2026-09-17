"""Device management for GPU/CPU selection."""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger("quality")


def get_device(force_cpu: bool = False) -> torch.device:
    """
    Get the appropriate device for PyTorch operations.

    Checks CUDA availability and returns the appropriate device.
    Logs GPU information if CUDA is available.

    Args:
        force_cpu: If True, force CPU usage even if CUDA is available

    Returns:
        torch.device: Either 'cuda' or 'cpu' device
    """
    if force_cpu:
        logger.info("Forcing CPU usage as per configuration")
        return torch.device("cpu")

    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        cuda_version = torch.version.cuda

        logger.info(f"CUDA is available. Using GPU: {gpu_name}")
        logger.info(f"GPU Memory: {gpu_memory:.2f} GB")
        logger.info(f"CUDA Version: {cuda_version}")

        return device
    else:
        logger.info("CUDA not available. Using CPU")
        return torch.device("cpu")
