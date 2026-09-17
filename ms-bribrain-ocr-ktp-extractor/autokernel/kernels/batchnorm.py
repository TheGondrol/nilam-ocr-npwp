"""
AutoKernel -- BatchNorm starter kernel.

Current kernel: PyTorch batch_norm fallback wrapper.
Target metric: end-to-end model latency, not standalone TFLOPS.
"""

KERNEL_TYPE = "batchnorm"

import torch
import torch.nn.functional as F


def kernel_fn(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Entry point called by bench.py. Uses inference-mode batch norm."""
    return F.batch_norm(
        x,
        running_mean,
        running_var,
        weight,
        bias,
        training=False,
        eps=eps,
    )
