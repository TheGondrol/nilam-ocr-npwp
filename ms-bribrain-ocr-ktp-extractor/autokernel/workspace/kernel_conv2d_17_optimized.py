"""
AutoKernel -- Extracted kernel from model profiling.
Op type: conv2d
Rank: 17 (2.1% of GPU time)
Model shape: N=1, C_in=48, C_out=48, H=12, W=160, KH=3, KW=3, stride_h=1, stride_w=1, pad_h=1, pad_w=1, dil_h=1, dil_w=1, groups=1

This kernel was extracted from profiling models/ppocrv5_server.py.
The agent optimizes this only if it improves model latency at the model-specific shapes.
"""

KERNEL_TYPE = "conv2d"

# Model-specific shapes (the shapes that matter for THIS model)
MODEL_SHAPES = {'N': 1, 'C_in': 48, 'C_out': 48, 'H': 12, 'W': 160, 'KH': 3, 'KW': 3, 'stride_h': 1, 'stride_w': 1, 'pad_h': 1, 'pad_w': 1, 'dil_h': 1, 'dil_w': 1, 'groups': 1}

# Benchmark config (self-describing -- bench.py can load this dynamically)
TEST_SIZES = [
    ("model_primary", {'N': 1, 'C_in': 48, 'C_out': 48, 'H': 12, 'W': 160, 'KH': 3, 'KW': 3, 'stride_h': 1, 'stride_w': 1, 'pad_h': 1, 'pad_w': 1, 'dil_h': 1, 'dil_w': 1, 'groups': 1}),
    # Also test nearby sizes for robustness
    ("model_half", {'N': 1, 'C_in': 24, 'C_out': 24, 'H': 6, 'W': 80, 'KH': 3, 'KW': 3, 'stride_h': 1, 'stride_w': 1, 'pad_h': 1, 'pad_w': 1, 'dil_h': 1, 'dil_w': 1, 'groups': 1}),
    ("model_double", {'N': 2, 'C_in': 96, 'C_out': 96, 'H': 24, 'W': 320, 'KH': 3, 'KW': 3, 'stride_h': 1, 'stride_w': 1, 'pad_h': 1, 'pad_w': 1, 'dil_h': 1, 'dil_w': 1, 'groups': 1}),
]

TOLERANCES = {'float16': {'atol': 0.01, 'rtol': 0.01}, 'bfloat16': {'atol': 0.02, 'rtol': 0.02}, 'float32': {'atol': 0.0001, 'rtol': 0.0001}}


def FLOPS_FN(s):
    return 2 * s["N"] * s["C_out"] * (((s["H"] + 2 * s["pad_h"] - s["dil_h"] * (s["KH"] - 1) - 1) // s["stride_h"]) + 1) * (((s["W"] + 2 * s["pad_w"] - s["dil_w"] * (s["KW"] - 1) - 1) // s["stride_w"]) + 1) * (s["C_in"] // s["groups"]) * s["KH"] * s["KW"]


def BYTES_FN(s, dt_bytes):
    return (s["N"] * s["C_in"] * s["H"] * s["W"] + s["C_out"] * (s["C_in"] // s["groups"]) * s["KH"] * s["KW"] + s["N"] * s["C_out"] * (((s["H"] + 2 * s["pad_h"] - s["dil_h"] * (s["KH"] - 1) - 1) // s["stride_h"]) + 1) * (((s["W"] + 2 * s["pad_w"] - s["dil_w"] * (s["KW"] - 1) - 1) // s["stride_w"]) + 1)) * dt_bytes


# ======================================================================
# Triton kernel code (from kernels/conv2d.py)
# ======================================================================

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
