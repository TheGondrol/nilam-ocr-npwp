"""
AutoKernel -- Extracted kernel from model profiling.
Op type: layout_transform
Rank: 4 (4.4% of GPU time)
Model shape: N=1, C=192, H=6, W=80

This kernel was extracted from profiling models/ppocrv5_server.py.
The agent optimizes this only if it improves model latency at the model-specific shapes.
"""

KERNEL_TYPE = "layout_transform"

# Model-specific shapes (the shapes that matter for THIS model)
MODEL_SHAPES = {'N': 1, 'C': 192, 'H': 6, 'W': 80}
GLOBAL_CHANNELS_LAST = True
INCLUDE_WITHOUT_STATE = True
VERIFY_ATOL = 0.002
VERIFY_RTOL = 0.002

# Benchmark config (self-describing -- bench.py can load this dynamically)
TEST_SIZES = [
    ("model_primary", {'N': 1, 'C': 192, 'H': 6, 'W': 80}),
    # Also test nearby sizes for robustness
    ("model_half", {'N': 1, 'C': 96, 'H': 3, 'W': 40}),
    ("model_double", {'N': 2, 'C': 384, 'H': 12, 'W': 160}),
]

TOLERANCES = {'float16': {'atol': 0.0, 'rtol': 0.0}, 'bfloat16': {'atol': 0.0, 'rtol': 0.0}, 'float32': {'atol': 0.0, 'rtol': 0.0}}


def FLOPS_FN(s):
    return s["N"] * s["C"] * s["H"] * s["W"]


def BYTES_FN(s, dt_bytes):
    return 2 * s["N"] * s["C"] * s["H"] * s["W"] * dt_bytes


# ======================================================================
# Triton kernel code (from kernels/layout_transform.py)
# ======================================================================

import torch


def kernel_fn(x: torch.Tensor) -> torch.Tensor:
    """Entry point called by bench.py. Materializes a channels-last view."""
    if x.ndim == 4:
        return x.contiguous(memory_format=torch.channels_last)
    return x.contiguous()
