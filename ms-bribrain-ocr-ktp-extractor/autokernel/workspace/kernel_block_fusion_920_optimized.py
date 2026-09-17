"""
AutoKernel -- Compiled head wrapper for the recognizer graph stack.
Op type: block_fusion
Rank: 920 (manual graph/block optimization)
Model target: net.head on PPOCRv5ServerRecModel
"""

KERNEL_TYPE = "block_fusion"

BLOCK_FUSION_KIND = "compile_original"
TARGET_MODULES = ["net.head"]

TORCH_COMPILE_BACKEND = "inductor"
TORCH_COMPILE_OPTIONS = {"triton.cudagraphs": False}

VERIFY_ATOL = 0.003
VERIFY_RTOL = 0.003

TEST_SIZES = [
    ("model_primary", {"N": 1, "C": 3, "H": 48, "W": 320}),
]

TOLERANCES = {
    "float16": {"atol": 0.003, "rtol": 0.003},
    "bfloat16": {"atol": 0.004, "rtol": 0.004},
    "float32": {"atol": 0.0001, "rtol": 0.0001},
}


def FLOPS_FN(s):
    return 0


def BYTES_FN(s, dt_bytes):
    return 0


import torch


def kernel_fn(x: torch.Tensor) -> torch.Tensor:
    return x
