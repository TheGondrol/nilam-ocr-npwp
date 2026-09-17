"""
AutoKernel -- Extracted kernel from model profiling.
Op type: softmax
Rank: 70 (0.2% of GPU time)
Model shape: rows=40, cols=18385

This kernel was extracted from profiling models/ppocrv5_server.py.
The agent optimizes this to maximize throughput at the model-specific shapes.
"""

KERNEL_TYPE = "softmax"

# Model-specific shapes (the shapes that matter for THIS model)
MODEL_SHAPES = {'rows': 40, 'cols': 18385}

# Benchmark config (self-describing -- bench.py can load this dynamically)
TEST_SIZES = [
    ("model_primary", {'rows': 40, 'cols': 18385}),
    # Also test nearby sizes for robustness
    ("model_half", {'rows': 20, 'cols': 9192}),
    ("model_double", {'rows': 80, 'cols': 36770}),
]

TOLERANCES = {'float16': {'atol': 0.001, 'rtol': 0.001}, 'bfloat16': {'atol': 0.002, 'rtol': 0.002}, 'float32': {'atol': 1e-05, 'rtol': 1e-05}}


def FLOPS_FN(s):
    return 5 * s["rows"] * s["cols"]


def BYTES_FN(s, dt_bytes):
    return 2 * s["rows"] * s["cols"] * dt_bytes


# ======================================================================
# Triton kernel code (from kernels/softmax.py)
# ======================================================================

import torch
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(
    input_ptr,
    output_ptr,
    n_cols,
    stride_input_row,
    stride_output_row,
    BLOCK_SIZE: tl.constexpr,
):
    """Row-parallel online softmax. One program per row."""
    row_idx = tl.program_id(0)

    row_start_input = input_ptr + row_idx * stride_input_row
    row_start_output = output_ptr + row_idx * stride_output_row

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # Load row
    row = tl.load(row_start_input + col_offsets, mask=mask, other=float("-inf"))

    # Numerically stable softmax: subtract max
    row_max = tl.max(row, axis=0)
    row = row - row_max

    # Exponentiate
    numerator = tl.exp(row)

    # Sum
    denominator = tl.sum(numerator, axis=0)

    # Divide
    result = numerator / denominator

    # Store
    tl.store(row_start_output + col_offsets, result, mask=mask)


@triton.jit
def softmax_tiled_kernel(
    input_ptr,
    output_ptr,
    n_cols,
    stride_input_row,
    stride_output_row,
    BLOCK_SIZE: tl.constexpr,
):
    """Long-row softmax with tiled online reduction to reduce register pressure."""
    row_idx = tl.program_id(0)
    row_start_input = input_ptr + row_idx * stride_input_row
    row_start_output = output_ptr + row_idx * stride_output_row
    col_offsets = tl.arange(0, BLOCK_SIZE)

    row_max = tl.full((), -float("inf"), dtype=tl.float32)
    row_sum = tl.zeros((), dtype=tl.float32)

    for start_col in tl.range(0, n_cols, BLOCK_SIZE):
        cols = start_col + col_offsets
        mask = cols < n_cols
        row = tl.load(row_start_input + cols, mask=mask, other=-float("inf")).to(tl.float32)
        tile_max = tl.max(row, axis=0)
        new_max = tl.maximum(row_max, tile_max)
        row_sum = row_sum * tl.exp(row_max - new_max) + tl.sum(tl.exp(row - new_max), axis=0)
        row_max = new_max

    for start_col in tl.range(0, n_cols, BLOCK_SIZE):
        cols = start_col + col_offsets
        mask = cols < n_cols
        row = tl.load(row_start_input + cols, mask=mask, other=-float("inf")).to(tl.float32)
        numerators = tl.exp(row - row_max)
        tl.store(row_start_output + cols, numerators / row_sum, mask=mask)


def kernel_fn(x: torch.Tensor) -> torch.Tensor:
    """Entry point called by bench.py. Must match reference.softmax_ref signature."""
    assert x.is_cuda

    if x.dtype == torch.float16:
        if x.ndim == 2 and x.shape == (40, 18385):
            output = torch.empty_like(x)
            softmax_tiled_kernel[(40,)](
                x,
                output,
                18385,
                x.stride(0),
                output.stride(0),
                BLOCK_SIZE=2048,
                num_warps=8,
                num_stages=1,
            )
            return output
        if x.ndim == 3 and x.shape == (1, 40, 18385):
            x_2d = x.view(40, 18385)
            output = torch.empty_like(x)
            output_2d = output.view(40, 18385)
            softmax_tiled_kernel[(40,)](
                x_2d,
                output_2d,
                18385,
                x_2d.stride(0),
                output_2d.stride(0),
                BLOCK_SIZE=2048,
                num_warps=8,
                num_stages=1,
            )
            return output

    # Flatten to 2D for row-parallel processing
    orig_shape = x.shape
    if x.ndim == 1:
        x = x.unsqueeze(0)
    elif x.ndim > 2:
        x = x.view(-1, x.shape[-1])

    n_rows, n_cols = x.shape
    output = torch.empty_like(x)

    if n_cols >= 16384:
        block_size = 2048
        num_warps = 8
        num_stages = 1
        kernel = softmax_tiled_kernel
    elif n_cols >= 4096:
        block_size = triton.next_power_of_2(n_cols)
        num_warps = 4
        num_stages = 2
        kernel = softmax_kernel
    else:
        block_size = triton.next_power_of_2(n_cols)
        num_warps = 2
        num_stages = 2
        kernel = softmax_kernel

    grid = (n_rows,)
    kernel[grid](
        x, output,
        n_cols,
        x.stride(0),
        output.stride(0),
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    return output.view(orig_shape)
