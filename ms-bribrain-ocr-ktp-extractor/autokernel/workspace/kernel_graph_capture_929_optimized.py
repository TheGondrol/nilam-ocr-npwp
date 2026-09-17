"""
AutoKernel -- Whole-model CUDA graph capture with reusable output buffer for PP-OCRv5 recognizer.
Op type: graph_capture
Rank: 929 (manual graph-level optimization)
Model shape: N=1, C=3, H=48, W=320
"""

KERNEL_TYPE = "graph_capture"

MODEL_SHAPES = {"N": 1, "C": 3, "H": 48, "W": 320}
GLOBAL_CUDA_GRAPH = True
CUDA_GRAPH_WARMUP_ITERS = 3
CUDA_GRAPH_OUTPUT_MODE = "reuse_copy"

VERIFY_ATOL = 0.0
VERIFY_RTOL = 0.0

TEST_SIZES = [
    ("model_primary", {"N": 1, "C": 3, "H": 48, "W": 320}),
]

TOLERANCES = {
    "float16": {"atol": 0.0, "rtol": 0.0},
    "bfloat16": {"atol": 0.0, "rtol": 0.0},
    "float32": {"atol": 0.0, "rtol": 0.0},
}


def FLOPS_FN(s):
    return 0


def BYTES_FN(s, dt_bytes):
    return 0


import torch


def kernel_fn(x: torch.Tensor) -> torch.Tensor:
    return x
