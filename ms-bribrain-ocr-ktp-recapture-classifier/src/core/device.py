"""
Device detection and management (GPU/CPU)
"""

import torch
import logging

logger = logging.getLogger(__name__)


def get_device(force_cpu: bool = False) -> torch.device:
    """
    Detect and return available device (CUDA GPU or CPU)
    
    Args:
        force_cpu: Force CPU mode even if GPU is available
        
    Returns:
        torch.device object (cuda or cpu)
    """
    if force_cpu:
        logger.info("CPU mode forced by configuration")
        return torch.device("cpu")
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
        
        # Log detailed GPU information
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        cuda_version = torch.version.cuda
        
        logger.info(f"Using GPU: {gpu_name}")
        logger.info(f"GPU Memory: {gpu_memory:.2f} GB")
        logger.info(f"CUDA Version: {cuda_version}")
        
        return device
    else:
        logger.warning("CUDA not available, falling back to CPU")
        logger.info("Using CPU for inference")
        return torch.device("cpu")


def get_device_info() -> dict:
    """
    Get detailed device information
    
    Returns:
        Dictionary with device information
    """
    info = {
        "device_type": "cpu",
        "cuda_available": torch.cuda.is_available(),
    }
    
    if torch.cuda.is_available():
        info.update({
            "device_type": "cuda",
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_memory_gb": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
            "cuda_version": torch.version.cuda,
            "device_count": torch.cuda.device_count(),
        })
    
    return info
