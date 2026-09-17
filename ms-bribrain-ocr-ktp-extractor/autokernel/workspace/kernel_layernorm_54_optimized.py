"""
AutoKernel -- Extracted kernel from model profiling.
Op type: layernorm
Rank: 54 (0.4% of GPU time)
Model shape: batch=40, dim=120

This kernel was extracted from profiling models/ppocrv5_server.py.
The agent optimizes this to maximize throughput at the model-specific shapes.
"""

KERNEL_TYPE = "layernorm"

# Model-specific shapes (the shapes that matter for THIS model)
MODEL_SHAPES = {'batch': 40, 'dim': 120}

# Benchmark config (self-describing -- bench.py can load this dynamically)
TEST_SIZES = [
    ("model_primary", {'batch': 40, 'dim': 120}),
    # Also test nearby sizes for robustness
    ("model_half", {'batch': 20, 'dim': 60}),
    ("model_double", {'batch': 80, 'dim': 240}),
]

TOLERANCES = {'float16': {'atol': 0.001, 'rtol': 0.001}, 'bfloat16': {'atol': 0.002, 'rtol': 0.002}, 'float32': {'atol': 1e-05, 'rtol': 1e-05}}


def FLOPS_FN(s):
    return 8 * s["batch"] * s["dim"]


def BYTES_FN(s, dt_bytes):
    return (2 * s["batch"] * s["dim"] + 2 * s["dim"]) * dt_bytes


# ======================================================================
# Triton kernel code (from kernels/layernorm.py)
# ======================================================================

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def layernorm_kernel(
    X_ptr,
    Y_ptr,
    W_ptr,
    B_ptr,
    stride_x_row,
    stride_x_col,
    stride_y_row,
    stride_y_col,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """Row-parallel layer normalization. One program per row."""
    row_idx = tl.program_id(0)

    row_start_x = X_ptr + row_idx * stride_x_row
    row_start_y = Y_ptr + row_idx * stride_y_row

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < N

    # Load row into float32 for numerical stability
    x = tl.load(row_start_x + col_offsets * stride_x_col, mask=mask, other=0.0).to(tl.float32)

    # Pass 1: compute mean
    mean = tl.sum(x, axis=0) / N

    # Pass 2: compute variance
    x_centered = tl.where(mask, x - mean, 0.0)
    variance = tl.sum(x_centered * x_centered, axis=0) / N

    # Normalize
    inv_std = 1.0 / tl.sqrt(variance + eps)
    x_norm = x_centered * inv_std

    # Load weight and bias
    w = tl.load(W_ptr + col_offsets, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(B_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)

    # Apply affine transform
    y = x_norm * w + b

    # Store (cast back to input dtype via the store)
    tl.store(row_start_y + col_offsets * stride_y_col, y, mask=mask)


@triton.jit
def layernorm_small_kernel(
    X_ptr,
    Y_ptr,
    W_ptr,
    B_ptr,
    stride_x_row,
    stride_x_col,
    stride_y_row,
    stride_y_col,
    M,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
    ROWS_PER_PROGRAM: tl.constexpr,
):
    """Process several short rows per program to reduce launch overhead."""
    pid = tl.program_id(0)

    row_offsets = pid * ROWS_PER_PROGRAM + tl.arange(0, ROWS_PER_PROGRAM)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    row_mask = row_offsets < M
    col_mask = col_offsets < N
    mask = row_mask[:, None] & col_mask[None, :]

    x_ptrs = X_ptr + row_offsets[:, None] * stride_x_row + col_offsets[None, :] * stride_x_col
    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)

    mean = tl.sum(x, axis=1) / N
    x_centered = tl.where(mask, x - mean[:, None], 0.0)
    variance = tl.sum(x_centered * x_centered, axis=1) / N
    inv_std = 1.0 / tl.sqrt(variance + eps)

    w = tl.load(W_ptr + col_offsets, mask=col_mask, other=1.0).to(tl.float32)
    b = tl.load(B_ptr + col_offsets, mask=col_mask, other=0.0).to(tl.float32)
    y = x_centered * inv_std[:, None] * w[None, :] + b[None, :]

    y_ptrs = Y_ptr + row_offsets[:, None] * stride_y_row + col_offsets[None, :] * stride_y_col
    tl.store(y_ptrs, y, mask=mask)


def kernel_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Entry point called by bench.py. Must match reference.layernorm_ref signature."""
    assert x.is_cuda

    if x.dtype == torch.bfloat16:
        return F.layer_norm(x, x.shape[-1:], weight, bias, eps)
    if x.stride(-1) != 1:
        return F.layer_norm(x, x.shape[-1:], weight, bias, eps)

    if (
        x.dtype == torch.float16
        and x.ndim == 2
        and x.shape == (40, 120)
        and weight.shape == (120,)
        and bias.shape == (120,)
    ):
        y = torch.empty_like(x)
        layernorm_small_kernel[(20,)](
            x, y,
            weight, bias,
            x.stride(0),
            x.stride(1),
            y.stride(0),
            y.stride(1),
            40,
            120,
            eps,
            BLOCK_SIZE=128,
            ROWS_PER_PROGRAM=2,
            num_warps=1,
            num_stages=1,
        )
        return y

    # Flatten to 2D for row-parallel processing
    orig_shape = x.shape
    if x.ndim == 1:
        x = x.unsqueeze(0)
    elif x.ndim > 2:
        x = x.view(-1, x.shape[-1])

    n_rows, n_cols = x.shape
    assert weight.shape[0] == n_cols
    assert bias.shape[0] == n_cols

    y = torch.empty_like(x)

    if n_rows <= 128 and n_cols <= 128:
        block_size = 128
        rows_per_program = 2
        grid = (triton.cdiv(n_rows, rows_per_program),)
        layernorm_small_kernel[grid](
            x, y,
            weight, bias,
            x.stride(0),
            x.stride(1),
            y.stride(0),
            y.stride(1),
            n_rows,
            n_cols,
            eps,
            BLOCK_SIZE=block_size,
            ROWS_PER_PROGRAM=rows_per_program,
            num_warps=1,
            num_stages=1,
        )
    else:
        block_size = triton.next_power_of_2(n_cols)
        grid = (n_rows,)
        layernorm_kernel[grid](
            x, y,
            weight, bias,
            x.stride(0),
            x.stride(1),
            y.stride(0),
            y.stride(1),
            n_cols,
            eps,
            BLOCK_SIZE=block_size,
        )

    return y.view(orig_shape)
