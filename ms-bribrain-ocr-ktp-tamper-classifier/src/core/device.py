"""
Device management for GPU/CPU selection and resource management.

This module provides intelligent device detection and management for
ML/compute workloads. Supports CUDA (NVIDIA), MPS (Apple Silicon), and
CPU with automatic fallback.

Usage:
    from src.core.device import get_device, DeviceManager
    
    # Get device based on config and availability
    device = get_device()
    print(f"Using device: {device}")
    
    # Use DeviceManager for advanced operations
    dm = DeviceManager()
    device = dm.get_optimal_device()
    dm.cleanup_memory()
"""

import logging
import platform
from typing import Optional

from src.core.config import get_config

logger = logging.getLogger(__name__)


class DeviceManager:
    """Manage compute device selection and resources."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize device manager.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = get_config(config_path)
        self.device_config = self.config.device
        self._device: Optional[str] = None
        self._device_info: Optional[dict] = None
    
    def get_optimal_device(self) -> str:
        """
        Get optimal device based on configuration and availability.
        
        Returns:
            str: Device string (e.g., "cuda:0", "mps", "cpu")
        """
        if self._device is not None:
            return self._device
        
        preferred = self.device_config.preferred.lower()
        
        if preferred == "auto":
            self._device = self._auto_detect_device()
        elif preferred == "cuda":
            if self._is_cuda_available():
                cuda_id = self.device_config.cuda_device_id
                self._device = f"cuda:{cuda_id}"
                logger.info(f"Using CUDA device: {self._device}")
            elif self.device_config.allow_cpu_fallback:
                self._device = "cpu"
                logger.warning("CUDA not available, falling back to CPU")
            else:
                raise RuntimeError("CUDA requested but not available, and CPU fallback is disabled")
        elif preferred == "mps":
            if self._is_mps_available():
                self._device = "mps"
                logger.info("Using MPS device (Apple Silicon)")
            elif self.device_config.allow_cpu_fallback:
                self._device = "cpu"
                logger.warning("MPS not available, falling back to CPU")
            else:
                raise RuntimeError("MPS requested but not available, and CPU fallback is disabled")
        else:  # cpu
            self._device = "cpu"
            logger.info("Using CPU device")
        
        return self._device
    
    def _auto_detect_device(self) -> str:
        """
        Auto-detect best available device.
        
        Returns:
            str: Device string
        """
        # Priority: CUDA > MPS > CPU
        if self._is_cuda_available():
            cuda_id = self.device_config.cuda_device_id
            device = f"cuda:{cuda_id}"
            logger.info(f"Auto-detected CUDA device: {device}")
            return device
        elif self._is_mps_available():
            logger.info("Auto-detected MPS device (Apple Silicon)")
            return "mps"
        else:
            logger.info("Auto-detected CPU device (no GPU available)")
            return "cpu"
    
    def _is_cuda_available(self) -> bool:
        """
        Check if CUDA is available.
        
        Returns:
            bool: True if CUDA is available
        """
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            logger.debug("PyTorch not installed, CUDA not available")
            return False
        except Exception as e:
            logger.debug(f"Error checking CUDA availability: {e}")
            return False
    
    def _is_mps_available(self) -> bool:
        """
        Check if MPS (Metal Performance Shaders) is available.
        
        Returns:
            bool: True if MPS is available
        """
        # MPS is only available on macOS with Apple Silicon
        if platform.system() != "Darwin":
            return False
        
        try:
            import torch
            return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        except ImportError:
            logger.debug("PyTorch not installed, MPS not available")
            return False
        except Exception as e:
            logger.debug(f"Error checking MPS availability: {e}")
            return False
    
    def get_device_info(self) -> dict:
        """
        Get detailed information about the current device.
        
        Returns:
            dict: Device information (name, memory, etc.)
        """
        if self._device_info is not None:
            return self._device_info
        
        device = self.get_optimal_device()
        info = {"device": device, "type": device.split(":")[0]}
        
        if device.startswith("cuda"):
            info.update(self._get_cuda_info())
        elif device == "mps":
            info.update(self._get_mps_info())
        else:  # cpu
            info.update(self._get_cpu_info())
        
        self._device_info = info
        return info
    
    def _get_cuda_info(self) -> dict:
        """Get CUDA device information."""
        try:
            import torch
            device_id = self.device_config.cuda_device_id
            return {
                "name": torch.cuda.get_device_name(device_id),
                "total_memory_gb": round(torch.cuda.get_device_properties(device_id).total_memory / 1024**3, 2),
                "cuda_version": torch.version.cuda,
            }
        except Exception as e:
            logger.warning(f"Could not get CUDA info: {e}")
            return {}
    
    def _get_mps_info(self) -> dict:
        """Get MPS device information."""
        return {
            "name": "Apple MPS (Metal Performance Shaders)",
            "architecture": platform.machine(),
        }
    
    def _get_cpu_info(self) -> dict:
        """Get CPU information."""
        import os
        return {
            "name": "CPU",
            "cores": os.cpu_count() or 1,
            "platform": platform.platform(),
        }
    
    def cleanup_memory(self) -> None:
        """
        Clean up device memory (useful for GPU).
        """
        device = self.get_optimal_device()
        
        if device.startswith("cuda"):
            try:
                import torch
                torch.cuda.empty_cache()
                logger.debug("CUDA cache cleared")
            except Exception as e:
                logger.warning(f"Could not clear CUDA cache: {e}")
        elif device == "mps":
            try:
                import torch
                if hasattr(torch.mps, "empty_cache"):
                    torch.mps.empty_cache()
                    logger.debug("MPS cache cleared")
            except Exception as e:
                logger.warning(f"Could not clear MPS cache: {e}")
    
    def set_memory_fraction(self, fraction: Optional[float] = None) -> None:
        """
        Set memory fraction for GPU (CUDA only).
        
        Args:
            fraction: Fraction of GPU memory to use (0.0-1.0), uses config if None
        """
        device = self.get_optimal_device()
        
        if not device.startswith("cuda"):
            logger.debug("Memory fraction only applicable to CUDA devices")
            return
        
        if fraction is None:
            fraction = self.device_config.memory_fraction
        
        try:
            import torch
            device_id = self.device_config.cuda_device_id
            torch.cuda.set_per_process_memory_fraction(fraction, device_id)
            logger.info(f"Set CUDA memory fraction to {fraction}")
        except Exception as e:
            logger.warning(f"Could not set CUDA memory fraction: {e}")


# Singleton instance
_device_manager: Optional[DeviceManager] = None


def get_device_manager(config_path: str = "config.yaml") -> DeviceManager:
    """
    Get DeviceManager singleton instance.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        DeviceManager: Device manager instance
    """
    global _device_manager
    
    if _device_manager is None:
        _device_manager = DeviceManager(config_path)
    
    return _device_manager


def get_device(config_path: str = "config.yaml") -> str:
    """
    Get optimal device string (convenience function).
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        str: Device string (e.g., "cuda:0", "mps", "cpu")
    """
    dm = get_device_manager(config_path)
    return dm.get_optimal_device()


def cleanup_device_memory() -> None:
    """Clean up device memory (convenience function)."""
    dm = get_device_manager()
    dm.cleanup_memory()


def log_device_info(device: str) -> None:
    """
    Log device information (convenience function).
    
    Args:
        device: Device string to log info about
    """
    dm = get_device_manager()
    info = dm.get_device_info()
    logger.info(f"Device: {device}")
    logger.info(f"Device info: {info}")
