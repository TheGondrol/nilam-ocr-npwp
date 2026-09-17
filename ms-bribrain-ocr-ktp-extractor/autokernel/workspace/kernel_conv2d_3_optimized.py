"""
AutoKernel -- Extracted kernel from model profiling.
Op type: conv2d
Rank: 3 (5.3% of GPU time)
Model shape: N=1, C_in=4096, C_out=256, H=1, W=40, KH=3, KW=3, stride_h=1, stride_w=1, pad_h=1, pad_w=1, dil_h=1, dil_w=1, groups=1

This kernel was extracted from profiling models/ppocrv5_server.py.
The agent optimizes this only if it improves model latency at the model-specific shapes.
"""

KERNEL_TYPE = "conv2d"

# Model-specific shapes (the shapes that matter for THIS model)
MODEL_SHAPES = {'N': 1, 'C_in': 4096, 'C_out': 256, 'H': 1, 'W': 40, 'KH': 3, 'KW': 3, 'stride_h': 1, 'stride_w': 1, 'pad_h': 1, 'pad_w': 1, 'dil_h': 1, 'dil_w': 1, 'groups': 1}

# Benchmark config (self-describing -- bench.py can load this dynamically)
TEST_SIZES = [
    ("model_primary", {'N': 1, 'C_in': 4096, 'C_out': 256, 'H': 1, 'W': 40, 'KH': 3, 'KW': 3, 'stride_h': 1, 'stride_w': 1, 'pad_h': 1, 'pad_w': 1, 'dil_h': 1, 'dil_w': 1, 'groups': 1}),
    # Also test nearby sizes for robustness
    ("model_half", {'N': 1, 'C_in': 2048, 'C_out': 128, 'H': 1, 'W': 20, 'KH': 3, 'KW': 3, 'stride_h': 1, 'stride_w': 1, 'pad_h': 1, 'pad_w': 1, 'dil_h': 1, 'dil_w': 1, 'groups': 1}),
    ("model_double", {'N': 2, 'C_in': 8192, 'C_out': 512, 'H': 2, 'W': 80, 'KH': 3, 'KW': 3, 'stride_h': 1, 'stride_w': 1, 'pad_h': 1, 'pad_w': 1, 'dil_h': 1, 'dil_w': 1, 'groups': 1}),
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


_WEIGHT_CL_CACHE: dict[tuple[int, tuple[int, ...], torch.dtype, int], torch.Tensor] = {}


def _channels_last_weight(weight: torch.Tensor) -> torch.Tensor:
    key = (
        int(weight.data_ptr()),
        tuple(int(dim) for dim in weight.shape),
        weight.dtype,
        int(weight.device.index or 0),
    )
    cached = _WEIGHT_CL_CACHE.get(key)
    if cached is None:
        cached = weight.detach().contiguous(memory_format=torch.channels_last)
        _WEIGHT_CL_CACHE[key] = cached
    return cached


def kernel_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
    dilation: int | tuple[int, int] = 1,
    groups: int = 1,
) -> torch.Tensor:
    """Use a cached channels-last weight for the exact PP-OCR conv path."""
    if x.ndim == 4 and x.is_contiguous(memory_format=torch.channels_last):
        return F.conv2d(
            x,
            _channels_last_weight(weight),
            bias,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )
    return F.conv2d(
        x,
        weight,
        bias,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )
