"""
AutoKernel -- Extracted kernel from model profiling.
Op type: matmul
Rank: 34 (0.7% of GPU time)
Model shape: M=40, N=18385, K=120

This kernel was extracted from profiling models/ppocrv5_server.py.
The agent optimizes this to maximize throughput at the model-specific shapes.
"""

KERNEL_TYPE = "matmul"

# Model-specific shapes (the shapes that matter for THIS model)
MODEL_SHAPES = {'M': 40, 'N': 18385, 'K': 120}

# Benchmark config (self-describing -- bench.py can load this dynamically)
TEST_SIZES = [
    ("model_primary", {'M': 40, 'N': 18385, 'K': 120}),
    # Also test nearby sizes for robustness
    ("model_half", {'M': 20, 'N': 9192, 'K': 60}),
    ("model_double", {'M': 80, 'N': 36770, 'K': 240}),
]

TOLERANCES = {'float16': {'atol': 0.01, 'rtol': 0.01}, 'bfloat16': {'atol': 0.02, 'rtol': 0.02}, 'float32': {'atol': 0.0001, 'rtol': 0.0001}}


def FLOPS_FN(s):
    return 2 * s["M"] * s["N"] * s["K"]


def BYTES_FN(s, dt_bytes):
    return (s["M"] * s["K"] + s["K"] * s["N"] + s["M"] * s["N"]) * dt_bytes


# ======================================================================
# Triton kernel code (from kernels/matmul.py)
# ======================================================================

import torch
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_in_group = pid % num_pid_in_group
    pid_m = first_pid_m + (pid_in_group % group_size_m)
    pid_n = pid_in_group // group_size_m

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for _ in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_mask = offs_k < K
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & k_mask[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=k_mask[:, None] & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
        offs_k += BLOCK_SIZE_K

    c = acc.to(C_ptr.dtype.element_ty)
    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=mask)


def kernel_fn(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Entry point called by bench.py. Must match reference.matmul_ref signature."""
    assert A.is_cuda and B.is_cuda
    M, K = A.shape
    K2, N = B.shape
    assert K == K2

    is_model_shape = (
        M == MODEL_SHAPES["M"]
        and N == MODEL_SHAPES["N"]
        and K == MODEL_SHAPES["K"]
    )
    if A.dtype == torch.float32 or K > 4096:
        return torch.matmul(A, B)
    if not is_model_shape and ((M % 32) or (N % 32) or (K % 32)):
        return torch.matmul(A, B)

    C = torch.empty((M, N), device=A.device, dtype=A.dtype)

    if M <= 64 and K <= 256 and N >= 4096:
        block_size_m = 32
        block_size_n = 128
        block_size_k = 32
        group_size_m = 4
        num_warps = 4
        num_stages = 3
    else:
        block_size_m = 128
        block_size_n = 128
        block_size_k = 32
        group_size_m = 8
        num_warps = 8
        num_stages = 4

    grid = (triton.cdiv(M, block_size_m) * triton.cdiv(N, block_size_n),)

    matmul_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_SIZE_M=block_size_m,
        BLOCK_SIZE_N=block_size_n,
        BLOCK_SIZE_K=block_size_k,
        GROUP_SIZE_M=group_size_m,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return C
