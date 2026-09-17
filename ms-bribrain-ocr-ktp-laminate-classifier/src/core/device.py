"""
Device Detection Module
Detects GPU availability and provides CPU fallback
"""

import torch
import os
from typing import Any, Tuple

from .logging import logger
from .config import config


def get_device() -> Tuple[str, dict[str, Any]]:
    """
    Detect and return the appropriate device for PyTorch
    
    Returns:
        Tuple of (device_string, device_info_dict)
    """
    device_info: dict[str, Any] = {
        'type': 'cpu',
        'name': 'CPU',
        'cuda_available': False
    }
    
    # Check if CPU is forced
    force_cpu = config.get('device.force_cpu', False)
    if force_cpu:
        logger.info("CPU mode forced by configuration")
        return 'cpu', device_info
    
    # Check for CUDA_VISIBLE_DEVICES environment variable
    cuda_visible = os.getenv('CUDA_VISIBLE_DEVICES', None)
    if cuda_visible == '':
        logger.info("CPU mode forced by CUDA_VISIBLE_DEVICES environment variable")
        return 'cpu', device_info
    
    # Check CUDA availability
    prefer_gpu = config.get('device.prefer_gpu', True)
    
    if prefer_gpu and torch.cuda.is_available():
        try:
            device_info['type'] = 'cuda'
            device_info['cuda_available'] = True
            device_info['name'] = torch.cuda.get_device_name(0)
            device_info['memory_gb'] = torch.cuda.get_device_properties(0).total_memory / 1024**3
            device_info['cuda_version'] = torch.version.cuda
            device_info['device_count'] = torch.cuda.device_count()
            
            logger.info(f"Using GPU: {device_info['name']}")
            logger.info(f"GPU Memory: {device_info['memory_gb']:.2f} GB")
            logger.info(f"CUDA Version: {device_info['cuda_version']}")
            logger.info(f"Device Count: {device_info['device_count']}")
            
            return 'cuda', device_info
            
        except Exception as e:
            logger.warning(f"Failed to initialize CUDA: {str(e)}")
            logger.warning("Falling back to CPU")
            device_info['type'] = 'cpu'
            device_info['name'] = 'CPU (CUDA failed)'
            return 'cpu', device_info
    else:
        if not prefer_gpu:
            logger.info("GPU disabled by configuration, using CPU")
        else:
            logger.warning("CUDA not available, using CPU")
        
        return 'cpu', device_info


def log_device_info(device: str, device_info: dict):
    """
    Log detailed device information
    
    Args:
        device: Device string ('cuda' or 'cpu')
        device_info: Device information dictionary
    """
    logger.info(f"Device: {device}")
    logger.info(f"Device Type: {device_info['type']}")
    logger.info(f"Device Name: {device_info['name']}")
    
    if device_info['cuda_available']:
        logger.info("CUDA Available: Yes")
        if 'memory_gb' in device_info:
            logger.info(f"GPU Memory: {device_info['memory_gb']:.2f} GB")
        if 'cuda_version' in device_info:
            logger.info(f"CUDA Version: {device_info['cuda_version']}")
    else:
        logger.info("CUDA Available: No")
