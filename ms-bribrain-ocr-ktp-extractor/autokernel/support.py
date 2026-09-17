"""
Support-stage helpers for AutoKernel operator families.

These helpers keep profiling, extraction, and reinsertion reporting aligned.
"""

from __future__ import annotations

from collections.abc import Collection


PROFILE_SUPPORTED_OP_TYPES = frozenset(
    {
        "matmul",
        "softmax",
        "layernorm",
        "rmsnorm",
        "batchnorm",
        "flash_attention",
        "fused_mlp",
        "cross_entropy",
        "rotary_embedding",
        "reduce",
        "conv2d",
        "depthwise_conv2d",
        "conv_transpose2d",
        "layout_transform",
        "graph_capture",
        "interpolate",
        "concat",
        "elementwise",
        "pooling",
    }
)

REINSERT_SUPPORTED_OP_TYPES = frozenset(
    {
        "matmul",
        "layernorm",
        "rmsnorm",
        "softmax",
        "conv2d",
        "batchnorm",
        "layout_transform",
        "graph_capture",
        "block_fusion",
    }
)


def build_support_stage(
    op_type: str,
    extract_supported_op_types: Collection[str],
) -> dict[str, bool]:
    """
    Describe how far this operator family is supported in AutoKernel today.

    profile_supported:
        The profiler can classify the kernel into a specific family.
    extract_supported:
        AutoKernel has starter-kernel coverage for extraction/benching.
    reinsert_supported:
        verify.py knows how to patch optimized kernels back into modules of
        this family.
    """
    extract_supported = op_type in extract_supported_op_types
    return {
        "profile_supported": op_type in PROFILE_SUPPORTED_OP_TYPES,
        "extract_supported": extract_supported,
        "reinsert_supported": extract_supported and op_type in REINSERT_SUPPORTED_OP_TYPES,
    }
