"""
Model loader service
Loads transformer model with forced CPU usage and JIT optimization
"""

import logging
import os
from typing import Any, Tuple

import torch
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModelForImageClassification

from src.core.device import log_device_info

logger = logging.getLogger(__name__)


def get_optimal_num_threads() -> int:
    """
    Dynamically detect the optimal number of threads for CPU inference.
    Works correctly in K8s pods with CPU limits.

    Priority:
    1. Explicit OMP_NUM_THREADS environment variable
    2. K8s CPU limit from cgroup (for containerized environments)
    3. os.cpu_count() as fallback

    Returns:
        int: Optimal number of threads to use
    """
    # Check if explicitly set via environment variable
    env_threads = os.environ.get("OMP_NUM_THREADS")
    if env_threads:
        try:
            threads = int(env_threads)
            logger.info(f"Using OMP_NUM_THREADS from environment: {threads}")
            return threads
        except ValueError:
            pass

    # Try to read CPU quota from cgroup v2 (K8s/Docker containers)
    try:
        with open("/sys/fs/cgroup/cpu.max", "r") as f:
            content = f.read().strip()
            if content != "max":
                quota_str, period_str = content.split()
                if quota_str != "max":
                    cpu_limit_v2 = int(quota_str) / int(period_str)
                    threads = max(1, int(cpu_limit_v2))
                    logger.info(
                        f"Detected K8s/cgroup v2 CPU limit: {cpu_limit_v2:.2f} cores -> {threads} threads"
                    )
                    return threads
    except (FileNotFoundError, PermissionError, ValueError):
        pass

    # Try cgroup v1 (older K8s/Docker)
    try:
        with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us", "r") as f:
            quota: int = int(f.read().strip())
        with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us", "r") as f:
            period: int = int(f.read().strip())
        if quota > 0:
            cpu_limit: float = quota / period
            threads = max(1, int(cpu_limit))
            logger.info(
                f"Detected K8s/cgroup v1 CPU limit: {cpu_limit:.2f} cores -> {threads} threads"
            )
            return threads
    except (FileNotFoundError, PermissionError, ValueError):
        pass

    # Fallback to os.cpu_count()
    cpu_count = os.cpu_count() or 1
    logger.info(f"Using os.cpu_count() for thread count: {cpu_count}")
    return cpu_count


def configure_cpu_threading() -> int:
    """
    Configure PyTorch and OpenMP threading for optimal CPU inference.

    Returns:
        int: Number of threads configured
    """
    num_threads = get_optimal_num_threads()

    # Set PyTorch intra-op parallelism (within single operation)
    torch.set_num_threads(num_threads)

    # Set PyTorch inter-op parallelism (between operations)
    # Use fewer threads for inter-op to avoid oversubscription
    inter_op_threads = max(1, num_threads // 2)
    torch.set_num_interop_threads(inter_op_threads)

    # Set environment variables for OpenMP and MKL (affects numpy, scipy, etc.)
    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    os.environ["MKL_NUM_THREADS"] = str(num_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(num_threads)

    # Optimize memory allocator for multi-threaded inference
    os.environ.setdefault("MALLOC_TRIM_THRESHOLD_", "0")

    logger.info(f"CPU threading configured: intra_op={num_threads}, inter_op={inter_op_threads}")
    logger.info(f"Environment: OMP_NUM_THREADS={num_threads}, MKL_NUM_THREADS={num_threads}")

    return num_threads


class JITModelWrapper(nn.Module):
    """
    Wrapper for HuggingFace models to make them compatible with JIT tracing.
    Extracts only the logits from the model output for tracing compatibility.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Forward pass that returns only logits tensor."""
        outputs = self.model(pixel_values)
        return outputs.logits


class JITOptimizedModel:
    """
    Container for JIT-optimized model that provides a compatible interface
    with the original HuggingFace model output format.
    """

    def __init__(self, jit_model: torch.jit.ScriptModule, config: Any):
        self.jit_model = jit_model
        self.config = config

    def __call__(self, pixel_values: torch.Tensor):
        """Call the JIT model and wrap output in a compatible format."""
        logits = self.jit_model(pixel_values)
        return _ModelOutput(logits)

    def eval(self):
        """Set model to evaluation mode."""
        return self

    def to(self, device):
        """Move model to device (no-op for frozen JIT model)."""
        return self


class _ModelOutput:
    """Simple output container to mimic HuggingFace model output."""

    def __init__(self, logits: torch.Tensor):
        self.logits = logits


def apply_jit_optimization(model: Any, device: str) -> Any:
    """
    Apply JIT (TorchScript) optimization to the model for CPU inference.

    Args:
        model: The PyTorch model to optimize
        device: The device string

    Returns:
        Optimized model (JIT compiled if on CPU, original otherwise)
    """
    if device != "cpu":
        logger.info("JIT optimization skipped (not using CPU)")
        return model

    try:
        # Store config before wrapping
        config = model.config

        # Wrap model for JIT compatibility
        wrapper = JITModelWrapper(model)
        wrapper.eval()

        # Create dummy input for tracing
        # Using the model's expected input size (320x320 based on config)
        dummy_input = torch.randn(1, 3, 320, 320, device=device)

        # Use torch.jit.trace for model optimization
        with torch.inference_mode():
            jit_model = torch.jit.trace(wrapper, dummy_input)
            jit_model = torch.jit.freeze(jit_model)

        # Warm up the JIT model
        for _ in range(3):
            with torch.inference_mode():
                _ = jit_model(dummy_input)

        logger.info("JIT optimization applied successfully for CPU inference")

        # Return wrapped JIT model with compatible interface
        return JITOptimizedModel(jit_model, config)

    except Exception as e:
        logger.warning(f"JIT optimization failed, using original model: {str(e)}")
        return model


def load_model(model_path: str, force_cpu: bool = True) -> Tuple[Any, Any, str]:
    """
    Load the fine-tuned tamper detection model with forced CPU and JIT optimization

    Args:
        model_path: Path to the model directory
        force_cpu: If True, force CPU usage (default: True)

    Returns:
        tuple: (model, processor, device)
    """
    # Configure CPU threading for optimal performance
    configure_cpu_threading()

    # Force CPU usage
    device = "cpu"
    logger.info("Forcing CPU device for inference")

    # Log detailed device information
    log_device_info(device)

    try:
        # Load processor
        processor = AutoImageProcessor.from_pretrained(model_path)
        logger.info(f"Processor loaded from {model_path}")

        # Load model
        model = AutoModelForImageClassification.from_pretrained(model_path)
        model.to(device)
        model.eval()
        logger.info(f"Model loaded successfully from {model_path}")
        logger.info(f"Model classes: {model.config.id2label}")

        # Apply JIT optimization for CPU
        optimized_model = apply_jit_optimization(model, device)

        return optimized_model, processor, device

    except Exception as e:
        logger.error(f"Failed to load model from {model_path}: {str(e)}")
        raise
