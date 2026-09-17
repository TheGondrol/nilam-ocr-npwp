"""
AutoKernel -- Extracted kernel from model profiling.
Op type: batchnorm
Rank: 2 (5.3% of GPU time)
Model shape: N=1, C=192, H=6, W=80

This kernel was extracted from profiling models/ppocrv5_server.py.
The agent optimizes this only if it improves model latency at the model-specific shapes.
"""

KERNEL_TYPE = "batchnorm"
GENERIC_FALLBACK = True
INCLUDE_WITHOUT_STATE = True

# Model-specific shapes (the shapes that matter for THIS model)
MODEL_SHAPES = {'N': 1, 'C': 192, 'H': 6, 'W': 80}

# Benchmark config (self-describing -- bench.py can load this dynamically)
TEST_SIZES = [
    ("model_primary", {'N': 1, 'C': 192, 'H': 6, 'W': 80}),
    # Also test nearby sizes for robustness
    ("model_half", {'N': 1, 'C': 96, 'H': 3, 'W': 40}),
    ("model_double", {'N': 2, 'C': 384, 'H': 12, 'W': 160}),
]

TOLERANCES = {'float16': {'atol': 0.001, 'rtol': 0.001}, 'bfloat16': {'atol': 0.002, 'rtol': 0.002}, 'float32': {'atol': 1e-05, 'rtol': 1e-05}}


def FLOPS_FN(s):
    return 4 * s["N"] * s["C"] * s["H"] * s["W"]


def BYTES_FN(s, dt_bytes):
    return (3 * s["N"] * s["C"] * s["H"] * s["W"] + 4 * s["C"]) * dt_bytes


# ======================================================================
# Triton kernel code (from kernels/batchnorm.py)
# ======================================================================

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
