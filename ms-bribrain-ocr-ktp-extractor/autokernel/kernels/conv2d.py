"""
AutoKernel -- Conv2d starter kernel.

Current kernel: PyTorch conv2d fallback wrapper.
Target metric: end-to-end model latency, not standalone TFLOPS.
"""

KERNEL_TYPE = "conv2d"

import torch
import torch.nn.functional as F


def kernel_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
    dilation: int | tuple[int, int] = 1,
    groups: int = 1,
) -> torch.Tensor:
    """Entry point called by bench.py. Matches torch.nn.functional.conv2d."""
    return F.conv2d(
        x,
        weight,
        bias,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )
