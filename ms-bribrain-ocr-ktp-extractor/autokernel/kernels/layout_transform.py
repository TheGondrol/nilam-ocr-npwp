"""
AutoKernel -- Layout transform starter kernel.

Current kernel: channels-last contiguous conversion.
Target metric: end-to-end model latency, not standalone TFLOPS.
"""

KERNEL_TYPE = "layout_transform"

import torch


def kernel_fn(x: torch.Tensor) -> torch.Tensor:
    """Entry point called by bench.py. Materializes a channels-last view."""
    if x.ndim == 4:
        return x.contiguous(memory_format=torch.channels_last)
    return x.contiguous()
