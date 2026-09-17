"""
AutoKernel -- Graph-stack candidate for rank-20 depthwise ConvBNReLU fusion.
Op type: conv2d
Rank: 20 (1.9% of GPU time)
Model shape: N=1, C_in=192, C_out=192, H=6, W=80, KH=5, KW=5, stride_h=1, stride_w=1, pad_h=2, pad_w=2, dil_h=1, dil_w=1, groups=192
"""

KERNEL_TYPE = "conv2d"

MODEL_SHAPES = {
    "N": 1,
    "C_in": 192,
    "C_out": 192,
    "H": 6,
    "W": 80,
    "KH": 5,
    "KW": 5,
    "stride_h": 1,
    "stride_w": 1,
    "pad_h": 2,
    "pad_w": 2,
    "dil_h": 1,
    "dil_w": 1,
    "groups": 192,
}

TEST_SIZES = [
    ("model_primary", MODEL_SHAPES),
    (
        "model_half",
        {
            "N": 1,
            "C_in": 96,
            "C_out": 96,
            "H": 3,
            "W": 40,
            "KH": 5,
            "KW": 5,
            "stride_h": 1,
            "stride_w": 1,
            "pad_h": 2,
            "pad_w": 2,
            "dil_h": 1,
            "dil_w": 1,
            "groups": 96,
        },
    ),
    (
        "model_double",
        {
            "N": 2,
            "C_in": 384,
            "C_out": 384,
            "H": 12,
            "W": 160,
            "KH": 5,
            "KW": 5,
            "stride_h": 1,
            "stride_w": 1,
            "pad_h": 2,
            "pad_w": 2,
            "dil_h": 1,
            "dil_w": 1,
            "groups": 384,
        },
    ),
]

TOLERANCES = {
    "float16": {"atol": 0.01, "rtol": 0.01},
    "bfloat16": {"atol": 0.02, "rtol": 0.02},
    "float32": {"atol": 0.0001, "rtol": 0.0001},
}


def FLOPS_FN(s):
    return (
        2
        * s["N"]
        * s["C_out"]
        * (((s["H"] + 2 * s["pad_h"] - s["dil_h"] * (s["KH"] - 1) - 1) // s["stride_h"]) + 1)
        * (((s["W"] + 2 * s["pad_w"] - s["dil_w"] * (s["KW"] - 1) - 1) // s["stride_w"]) + 1)
        * (s["C_in"] // s["groups"])
        * s["KH"]
        * s["KW"]
    )


def BYTES_FN(s, dt_bytes):
    return (
        s["N"] * s["C_in"] * s["H"] * s["W"]
        + s["C_out"] * (s["C_in"] // s["groups"]) * s["KH"] * s["KW"]
        + s["N"]
        * s["C_out"]
        * (((s["H"] + 2 * s["pad_h"] - s["dil_h"] * (s["KH"] - 1) - 1) // s["stride_h"]) + 1)
        * (((s["W"] + 2 * s["pad_w"] - s["dil_w"] * (s["KW"] - 1) - 1) // s["stride_w"]) + 1)
    ) * dt_bytes


import torch
import torch.nn.functional as F


FUSE_BATCHNORM_ACT = True


def kernel_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
    dilation: int | tuple[int, int] = 1,
    groups: int = 1,
) -> torch.Tensor:
    return F.conv2d(
        x,
        weight,
        bias,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )
