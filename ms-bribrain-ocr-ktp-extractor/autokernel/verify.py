#!/usr/bin/env python3
"""
AutoKernel End-to-End Verifier -- Plug optimized kernels back into the model and verify.

Usage:
    uv run verify.py --model models/llama_7b.py --class-name LlamaModel --input-shape 1,2048
    uv run verify.py --module transformers --class-name AutoModelForCausalLM --pretrained meta-llama/Llama-2-7b-hf
    uv run verify.py --model models/llama_7b.py --class-name LlamaModel --input-shape 1,2048 --diagnose

Checks:
  1. Loads the original model
  2. Runs inference with original PyTorch ops -> captures reference output
  3. Replaces bottleneck ops with optimized Triton kernels
  4. Runs inference with optimized kernels -> captures optimized output
  5. Compares outputs (tolerance check)
  6. Benchmarks both paths -> reports end-to-end speedup
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import inspect
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from support import build_support_stage

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_DIR = os.path.join(SCRIPT_DIR, "workspace")
ORCHESTRATION_STATE = os.path.join(WORKSPACE_DIR, "orchestration_state.json")

# Benchmarking defaults
WARMUP_RUNS = 10
TIMED_RUNS = 50

# Tolerance defaults by dtype
DEFAULT_TOLERANCES: Dict[torch.dtype, Dict[str, float]] = {
    torch.float16:  {"atol": 1e-3, "rtol": 1e-3},
    torch.bfloat16: {"atol": 2e-3, "rtol": 2e-3},
    torch.float32:  {"atol": 1e-5, "rtol": 1e-5},
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class KernelReplacement:
    """Describes a single kernel replacement: what to replace and with what."""
    kernel_type: str          # e.g. "matmul", "layernorm", "rmsnorm"
    rank: int                 # priority rank from profiling
    speedup: float            # individual kernel speedup
    optimized_path: str       # path to optimized kernel .py file
    module_fn: Optional[Callable] = None  # loaded kernel function
    reinsert_supported: bool = False
    model_shapes: Optional[Dict[str, int]] = None
    generic_fallback: bool = False
    fuse_batchnorm_act: bool = False
    global_channels_last: bool = False
    global_cuda_graph: bool = False
    cuda_graph_warmup_iters: int = 3
    cuda_graph_input_mode: str = "clone"
    cuda_graph_output_mode: str = "clone"
    block_fusion_kind: Optional[str] = None
    target_modules: Optional[List[str]] = None
    torch_compile_backend: Optional[str] = None
    torch_compile_mode: Optional[str] = None
    torch_compile_options: Optional[Dict[str, Any]] = None
    verify_atol: Optional[float] = None
    verify_rtol: Optional[float] = None
    status: Optional[str] = None


@dataclass
class VerificationResult:
    """Full verification result."""
    model_name: str = ""
    input_shape: str = ""
    dtype_str: str = ""
    gpu_name: str = ""

    # Reference run
    ref_output_shape: str = ""
    ref_latency_ms: float = 0.0

    # Optimized run
    opt_output_shape: str = ""
    opt_latency_ms: float = 0.0
    kernels_replaced: List[Dict[str, Any]] = field(default_factory=list)
    kernels_skipped: List[Dict[str, Any]] = field(default_factory=list)

    # Comparison
    correctness: str = "UNKNOWN"
    max_abs_error: float = 0.0
    mean_abs_error: float = 0.0
    has_nan: bool = False
    has_inf: bool = False

    # Summary
    end_to_end_speedup: float = 0.0


# ---------------------------------------------------------------------------
# 1. Model Loading
# ---------------------------------------------------------------------------

def _ensure_stdlib_profile_module_for_compile() -> None:
    """No-op — profile.py was renamed to ak_profile.py to avoid shadowing."""
    pass

def load_model_from_file(model_path: str, class_name: str, **kwargs) -> nn.Module:
    """Load a model from a Python file by importing it and instantiating the class."""
    model_path = os.path.abspath(model_path)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    spec = importlib.util.spec_from_file_location("user_model", model_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import model from: {model_path}")

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    if not hasattr(mod, class_name):
        available = [n for n in dir(mod) if not n.startswith("_")]
        raise AttributeError(
            f"Class '{class_name}' not found in {model_path}. "
            f"Available names: {available}"
        )

    cls = getattr(mod, class_name)
    model = cls(**kwargs)
    return model


def load_model_from_module(module_name: str, class_name: str,
                           pretrained: Optional[str] = None, **kwargs) -> nn.Module:
    """Load a model from an installed Python module (e.g. 'transformers')."""
    try:
        mod = importlib.import_module(module_name)
    except ImportError as e:
        raise ImportError(
            f"Cannot import module '{module_name}'. Is it installed? Error: {e}"
        )

    if not hasattr(mod, class_name):
        raise AttributeError(
            f"Class '{class_name}' not found in module '{module_name}'."
        )

    cls = getattr(mod, class_name)

    if pretrained:
        # HuggingFace-style: cls.from_pretrained(...)
        if hasattr(cls, "from_pretrained"):
            model = cls.from_pretrained(pretrained, **kwargs)
        else:
            raise AttributeError(
                f"'{class_name}' has no 'from_pretrained' method. "
                f"Cannot load pretrained weights from '{pretrained}'."
            )
    else:
        model = cls(**kwargs)

    return model


def load_model(args) -> nn.Module:
    """Unified model loader from CLI args."""
    dtype = _parse_dtype(args.dtype)

    if args.model:
        print(f"Loading model from file: {args.model} (class: {args.class_name})")
        model = load_model_from_file(args.model, args.class_name)
    elif args.module:
        print(f"Loading model from module: {args.module} (class: {args.class_name})")
        extra_kwargs = {}
        if dtype == torch.float16:
            extra_kwargs["torch_dtype"] = torch.float16
        elif dtype == torch.bfloat16:
            extra_kwargs["torch_dtype"] = torch.bfloat16
        model = load_model_from_module(
            args.module, args.class_name, pretrained=args.pretrained, **extra_kwargs
        )
    else:
        raise ValueError("Must specify either --model (file path) or --module (Python module)")

    model = model.to(dtype=dtype)

    if torch.cuda.is_available():
        try:
            model = model.cuda()
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"WARNING: OOM moving model to GPU. Trying with smaller footprint...")
                torch.cuda.empty_cache()
                model = model.half().cuda()
            else:
                raise

    model.eval()
    return model


# ---------------------------------------------------------------------------
# 2. Input Generation
# ---------------------------------------------------------------------------

def generate_sample_input(
    input_shape: str,
    dtype: torch.dtype,
    device: str = "cuda",
    seed: int = 42,
) -> torch.Tensor:
    """Generate a sample input tensor from a shape string like '1,2048'."""
    dims = [int(d.strip()) for d in input_shape.split(",")]
    torch.manual_seed(seed)

    if dtype in (torch.int32, torch.int64, torch.long):
        # For language models, generate token IDs (assume vocab size ~32000)
        return torch.randint(0, 32000, dims, device=device, dtype=dtype)
    else:
        return torch.randn(dims, device=device, dtype=dtype)


def infer_input_type(model: nn.Module) -> str:
    """Try to determine if the model expects integer token IDs or float tensors."""
    # Check if model has an embedding layer as the first module
    for name, child in model.named_children():
        if isinstance(child, nn.Embedding):
            return "token_ids"
        if isinstance(child, (nn.Linear, nn.Conv2d)):
            return "float"
    return "float"


def make_model_input(
    model: nn.Module,
    input_shape: str,
    dtype: torch.dtype,
    device: str = "cuda",
) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
    """Create an appropriate input for the model."""
    input_type = infer_input_type(model)

    if input_type == "token_ids":
        # Language model: expects integer input_ids
        dims = [int(d.strip()) for d in input_shape.split(",")]
        torch.manual_seed(42)
        input_ids = torch.randint(0, 32000, dims, device=device, dtype=torch.long)

        # Check if model accepts input_ids keyword
        sig = inspect.signature(model.forward)
        if "input_ids" in sig.parameters:
            return {"input_ids": input_ids}
        return input_ids
    else:
        return generate_sample_input(input_shape, dtype, device)


# ---------------------------------------------------------------------------
# 3. Benchmarking
# ---------------------------------------------------------------------------

def benchmark_model(
    model: nn.Module,
    model_input: Union[torch.Tensor, Dict[str, torch.Tensor]],
    warmup: int = WARMUP_RUNS,
    timed: int = TIMED_RUNS,
) -> Tuple[Any, float]:
    """
    Benchmark model inference. Returns (output, median_latency_ms).
    Uses CUDA events for precise GPU timing.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for benchmarking.")

    def _run():
        with torch.no_grad():
            if isinstance(model_input, dict):
                return model(**model_input)
            else:
                return model(model_input)

    # Warmup
    print(f"  Warmup: {warmup} runs...", end="", flush=True)
    for _ in range(warmup):
        output = _run()
    torch.cuda.synchronize()
    print(" done")

    # Timed runs
    print(f"  Timed: {timed} runs...", end="", flush=True)
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(timed)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(timed)]

    torch.cuda.synchronize()
    for i in range(timed):
        start_events[i].record()
        _run()
        end_events[i].record()
    torch.cuda.synchronize()
    print(" done")

    # Compute median
    times_ms = sorted(s.elapsed_time(e) for s, e in zip(start_events, end_events))
    median_ms = times_ms[len(times_ms) // 2]

    # Final reference output (deterministic)
    with torch.no_grad():
        output = _run()
    torch.cuda.synchronize()

    return output, median_ms


# ---------------------------------------------------------------------------
# 4. Kernel Replacement
# ---------------------------------------------------------------------------

def load_orchestration_state() -> Optional[Dict]:
    """Load workspace/orchestration_state.json if it exists."""
    if not os.path.exists(ORCHESTRATION_STATE):
        return None
    with open(ORCHESTRATION_STATE, "r") as f:
        return json.load(f)


def discover_optimized_kernels() -> List[KernelReplacement]:
    """
    Find optimized kernels from the workspace directory.
    Checks orchestration_state.json first, then scans for *_optimized.py files.
    """
    replacements: List[KernelReplacement] = []
    seen_paths: set[str] = set()
    state_known_paths: set[str] = set()

    # Strategy 1: Read orchestration state
    state = load_orchestration_state()
    if state and "kernels" in state:
        for k in state["kernels"]:
            ktype = k.get("op_type", k.get("type", "unknown"))
            rank = k.get("rank", 0)
            speedup = k.get(
                "end_to_end_speedup",
                k.get("speedup", k.get("best_speedup", 1.0)),
            )
            # optimized_path is not written by orchestrate.py, so derive it
            # from the kernel file path if available
            opt_path = k.get("optimized_path", "")

            if not opt_path:
                # Try to derive from the "file" key that orchestrate.py writes
                base_file = k.get("file", "")
                if base_file:
                    stem = Path(base_file).stem
                    opt_path = os.path.join(
                        WORKSPACE_DIR, f"{stem}_optimized.py"
                    )
                else:
                    # Fallback convention: workspace/kernel_{type}_{rank}_optimized.py
                    opt_path = os.path.join(
                        WORKSPACE_DIR, f"kernel_{ktype}_{rank}_optimized.py"
                    )

            state_known_paths.add(os.path.abspath(opt_path))

            status = str(k.get("status", "")).strip().lower()
            include_candidate = os.path.exists(opt_path) and (
                status in {"done", "optimizing"}
                or (not status and speedup is not None and speedup > 1.0)
            )

            if include_candidate:
                support_stage = build_support_stage(ktype, {ktype})
                replacements.append(KernelReplacement(
                    kernel_type=ktype,
                    rank=rank,
                    speedup=speedup,
                    optimized_path=opt_path,
                    reinsert_supported=support_stage["reinsert_supported"],
                    status=k.get("status"),
                ))
                seen_paths.add(os.path.abspath(opt_path))

    # Strategy 2: Scan workspace directory for optimized kernel files
    if not os.path.isdir(WORKSPACE_DIR):
        return replacements

    for fname in sorted(os.listdir(WORKSPACE_DIR)):
        if fname.endswith("_optimized.py"):
            # Parse filename: kernel_{type}_{rank}_optimized.py
            # Type can be multi-word (e.g. flash_attention), so the rank
            # is always the last numeric segment before "_optimized.py".
            stem = fname.replace("_optimized.py", "")  # e.g. "kernel_flash_attention_1"
            parts = stem.split("_")
            if len(parts) >= 3 and parts[0] == "kernel":
                # Find the rank: last part that is purely numeric
                rank = 0
                rank_idx = len(parts)
                for i in range(len(parts) - 1, 0, -1):
                    if parts[i].isdigit():
                        rank = int(parts[i])
                        rank_idx = i
                        break
                # Everything between parts[1] and the rank index is the type
                ktype = "_".join(parts[1:rank_idx]) if rank_idx > 1 else parts[1]
                opt_path = os.path.join(WORKSPACE_DIR, fname)
                if os.path.abspath(opt_path) in state_known_paths:
                    continue
                if os.path.abspath(opt_path) in seen_paths:
                    continue
                try:
                    extra_mod = load_kernel_module(opt_path)
                except Exception:
                    continue
                if not bool(getattr(extra_mod, "INCLUDE_WITHOUT_STATE", False)):
                    continue
                support_stage = build_support_stage(ktype, {ktype})
                replacements.append(KernelReplacement(
                    kernel_type=ktype,
                    rank=rank,
                    speedup=0.0,  # Unknown without state file
                    optimized_path=opt_path,
                    reinsert_supported=support_stage["reinsert_supported"],
                    status="optimizing",
                ))

    return replacements


def load_kernel_module(path: str) -> Any:
    """Dynamically import a kernel .py file and return the module."""
    path = os.path.abspath(path)
    module_name = f"opt_kernel_{os.path.basename(path).replace('.py', '')}"
    cached = sys.modules.get(module_name)
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load kernel from: {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return mod


def split_replacements_for_verification(
    replacements: List[KernelReplacement],
) -> Tuple[List[KernelReplacement], List[KernelReplacement]]:
    """
    Split discovered kernels into:
    - reference replacements: accepted stack that should be active in the baseline run
    - optimized replacements: reference stack plus the current in-progress candidate(s)
    """
    has_optimizing = any(r.status == "optimizing" for r in replacements)
    if not has_optimizing:
        return [], replacements

    reference_replacements: List[KernelReplacement] = []
    candidate_replacements: List[KernelReplacement] = []
    for repl in replacements:
        if repl.status == "optimizing":
            candidate_replacements.append(repl)
        elif repl.speedup is not None and repl.speedup > 1.0:
            reference_replacements.append(repl)

    if not candidate_replacements:
        return [], replacements
    return reference_replacements, reference_replacements + candidate_replacements


class _LinearWrapper(nn.Module):
    """Wraps nn.Linear to use an optimized matmul kernel_fn."""

    def __init__(self, original: nn.Linear, kernel_fn: Callable):
        super().__init__()
        self.original = original
        self.kernel_fn = kernel_fn
        self.weight = original.weight
        self.bias = original.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Reshape to 2D for kernel_fn, then reshape back
        orig_shape = x.shape
        if x.dim() > 2:
            x_2d = x.reshape(-1, x.shape[-1])
        else:
            x_2d = x

        # kernel_fn expects (A, B) where A @ B = C
        # For nn.Linear: output = input @ weight.T + bias
        # So we call kernel_fn(input, weight.T)
        weight_t = self.weight.t().contiguous()
        out = self.kernel_fn(x_2d, weight_t)

        if self.bias is not None:
            out = out + self.bias

        if len(orig_shape) > 2:
            out = out.reshape(*orig_shape[:-1], out.shape[-1])

        return out


class _Conv2dWrapper(nn.Module):
    """Wraps nn.Conv2d to use an optimized kernel_fn."""

    def __init__(self, original: nn.Conv2d, kernel_fn: Callable):
        super().__init__()
        self.original = original
        self.kernel_fn = kernel_fn
        self.weight = original.weight
        self.bias = original.bias
        self.stride = original.stride
        self.padding = original.padding
        self.dilation = original.dilation
        self.groups = original.groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.kernel_fn(
            x,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )


class _ConvBnActFusedWrapper(nn.Module):
    """Inference-only Conv+BatchNorm(+ReLU/Identity) wrapper using a conv kernel."""

    def __init__(
        self,
        conv: nn.Conv2d,
        bn: nn.BatchNorm2d,
        act: nn.Module,
        kernel_fn: Callable,
    ):
        super().__init__()
        self.kernel_fn = kernel_fn
        self.stride = conv.stride
        self.padding = conv.padding
        self.dilation = conv.dilation
        self.groups = conv.groups
        self.act = act
        self._use_inplace_relu = isinstance(act, nn.ReLU)

        with torch.no_grad():
            weight_fp32 = conv.weight.detach().float()
            conv_bias_fp32 = (
                conv.bias.detach().float()
                if conv.bias is not None
                else torch.zeros(conv.out_channels, device=weight_fp32.device, dtype=torch.float32)
            )
            bn_weight_fp32 = bn.weight.detach().float()
            bn_bias_fp32 = bn.bias.detach().float()
            running_mean_fp32 = bn.running_mean.detach().float()
            running_var_fp32 = bn.running_var.detach().float()
            scale_fp32 = bn_weight_fp32 / torch.sqrt(running_var_fp32 + bn.eps)
            folded_weight = weight_fp32 * scale_fp32.reshape(-1, 1, 1, 1)
            folded_bias = bn_bias_fp32 + (conv_bias_fp32 - running_mean_fp32) * scale_fp32

        self.register_buffer(
            "folded_weight",
            folded_weight.to(dtype=conv.weight.dtype).contiguous(memory_format=torch.channels_last),
            persistent=False,
        )
        self.register_buffer(
            "folded_bias",
            folded_bias.to(dtype=conv.weight.dtype),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4 and not x.is_contiguous(memory_format=torch.channels_last):
            x = x.contiguous(memory_format=torch.channels_last)
        y = self.kernel_fn(
            x,
            self.folded_weight,
            self.folded_bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
        if self._use_inplace_relu:
            return torch.relu_(y)
        return self.act(y)


def _extract_folded_conv_like(
    module: nn.Module,
) -> Tuple[torch.Tensor, torch.Tensor, Any, Any, Any, int, bool, Optional[nn.Module]]:
    if hasattr(module, "folded_weight") and hasattr(module, "folded_bias"):
        return (
            module.folded_weight.detach(),
            module.folded_bias.detach(),
            getattr(module, "stride"),
            getattr(module, "padding"),
            getattr(module, "dilation"),
            int(getattr(module, "groups")),
            bool(getattr(module, "_use_inplace_relu", False) or isinstance(getattr(module, "act", None), nn.ReLU)),
            getattr(module, "act", None) if isinstance(getattr(module, "act", None), nn.Module) else None,
        )

    conv = getattr(module, "conv", None)
    bn = getattr(module, "bn", None) or getattr(module, "norm", None)
    if not isinstance(conv, nn.Conv2d) or not isinstance(bn, nn.BatchNorm2d):
        raise TypeError(f"Unsupported conv-like module for block fusion: {type(module).__name__}")

    with torch.no_grad():
        weight_fp32 = conv.weight.detach().float()
        conv_bias_fp32 = (
            conv.bias.detach().float()
            if conv.bias is not None
            else torch.zeros(conv.out_channels, device=weight_fp32.device, dtype=torch.float32)
        )
        bn_weight_fp32 = bn.weight.detach().float()
        bn_bias_fp32 = bn.bias.detach().float()
        running_mean_fp32 = bn.running_mean.detach().float()
        running_var_fp32 = bn.running_var.detach().float()
        scale_fp32 = bn_weight_fp32 / torch.sqrt(running_var_fp32 + bn.eps)
        folded_weight = weight_fp32 * scale_fp32.reshape(-1, 1, 1, 1)
        folded_bias = bn_bias_fp32 + (conv_bias_fp32 - running_mean_fp32) * scale_fp32

    act = getattr(module, "act", None)
    return (
        folded_weight.to(dtype=conv.weight.dtype).contiguous(memory_format=torch.channels_last),
        folded_bias.to(dtype=conv.weight.dtype),
        conv.stride,
        conv.padding,
        conv.dilation,
        int(conv.groups),
        bool(getattr(module, "_use_inplace_relu", False) or isinstance(act, nn.ReLU)),
        act if isinstance(act, nn.Module) else None,
    )


class _FoldedConvLikeWrapper(nn.Module):
    """Inference wrapper that folds Conv+BatchNorm and preserves the activation."""

    def __init__(self, module: nn.Module):
        super().__init__()
        weight, bias, stride, padding, dilation, groups, use_inplace_relu, act = _extract_folded_conv_like(module)
        self.register_buffer("weight", weight.contiguous(memory_format=torch.channels_last), persistent=False)
        self.register_buffer("bias", bias, persistent=False)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self._use_inplace_relu = use_inplace_relu
        self.act = act

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4 and not x.is_contiguous(memory_format=torch.channels_last):
            x = x.contiguous(memory_format=torch.channels_last)
        y = F.conv2d(
            x,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
        if self._use_inplace_relu:
            return torch.relu_(y)
        if self.act is not None:
            return self.act(y)
        return y


def _apply_folded_conv_like_activation(module: _FoldedConvLikeWrapper, y: torch.Tensor) -> torch.Tensor:
    if getattr(module, "_use_inplace_relu", False):
        return torch.relu_(y)
    act = getattr(module, "act", None)
    if act is not None:
        return act(y)
    return y


class _FoldedLightConvWrapper(nn.Module):
    """Inference wrapper for PP-HGNetV2 LightConvBNAct blocks."""

    def __init__(self, module: nn.Module):
        super().__init__()
        self.conv1 = _FoldedConvLikeWrapper(getattr(module, "conv1"))
        self.conv2 = _FoldedConvLikeWrapper(getattr(module, "conv2"))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2(self.conv1(x))


class _HGV2BlockFusedWrapper(nn.Module):
    """Inference-only PP-HGNetV2 block wrapper with all ConvBNAct pairs folded."""

    def __init__(self, module: nn.Module):
        super().__init__()
        self.identity = bool(getattr(module, "identity", False))
        layers = []
        for layer in getattr(module, "layers"):
            if hasattr(layer, "conv1") and hasattr(layer, "conv2"):
                layers.append(_FoldedLightConvWrapper(layer))
            else:
                layers.append(_FoldedConvLikeWrapper(layer))
        self.layers = nn.ModuleList(layers)
        self.aggregation_squeeze_conv = _FoldedConvLikeWrapper(
            getattr(module, "aggregation_squeeze_conv")
        )
        self.aggregation_excitation_conv = _FoldedConvLikeWrapper(
            getattr(module, "aggregation_excitation_conv")
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        outputs = [x]
        for layer in self.layers:
            x = layer(x)
            outputs.append(x)
        x = torch.cat(outputs, dim=1)
        x = self.aggregation_squeeze_conv(x)
        x = self.aggregation_excitation_conv(x)
        if self.identity:
            x = x + identity
        return x


class _HGV2BlockPreallocCatWrapper(nn.Module):
    """Inference-only PP-HGNetV2 block wrapper that preallocates the concat buffer."""

    def __init__(self, module: nn.Module):
        super().__init__()
        self.identity = bool(getattr(module, "identity", False))
        layers = []
        layer_out_channels: List[int] = []
        for layer in getattr(module, "layers"):
            if hasattr(layer, "conv1") and hasattr(layer, "conv2"):
                wrapped = _FoldedLightConvWrapper(layer)
                out_channels = int(wrapped.conv2.weight.shape[0])
            else:
                wrapped = _FoldedConvLikeWrapper(layer)
                out_channels = int(wrapped.weight.shape[0])
            layers.append(wrapped)
            layer_out_channels.append(out_channels)
        self.layers = nn.ModuleList(layers)
        self.layer_out_channels = layer_out_channels
        self.total_extra_channels = int(sum(layer_out_channels))
        self.aggregation_squeeze_conv = _FoldedConvLikeWrapper(
            getattr(module, "aggregation_squeeze_conv")
        )
        self.aggregation_excitation_conv = _FoldedConvLikeWrapper(
            getattr(module, "aggregation_excitation_conv")
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        n, c_in, h, w = x.shape
        memory_format = (
            torch.channels_last
            if x.is_contiguous(memory_format=torch.channels_last)
            else torch.contiguous_format
        )
        cat_buffer = torch.empty(
            (n, c_in + self.total_extra_channels, h, w),
            device=x.device,
            dtype=x.dtype,
            memory_format=memory_format,
        )
        cat_buffer[:, :c_in].copy_(x)

        current = x
        offset = c_in
        for layer, out_channels in zip(self.layers, self.layer_out_channels):
            current = layer(current)
            cat_buffer[:, offset:offset + out_channels].copy_(current)
            offset += out_channels

        x = self.aggregation_squeeze_conv(cat_buffer)
        x = self.aggregation_excitation_conv(x)
        if self.identity:
            x = x + identity
        return x


class _HGV2BlockSqueezeAccumulateWrapper(nn.Module):
    """Inference-only PP-HGNetV2 block wrapper that removes the explicit concat before squeeze."""

    def __init__(self, module: nn.Module):
        super().__init__()
        self.identity = bool(getattr(module, "identity", False))
        layers = []
        squeeze_input_channels: List[int] = []
        for i, layer in enumerate(getattr(module, "layers")):
            if hasattr(layer, "conv1") and hasattr(layer, "conv2"):
                wrapped = _FoldedLightConvWrapper(layer)
                if i == 0:
                    squeeze_input_channels.append(int(wrapped.conv1.weight.shape[1]))
                squeeze_input_channels.append(int(wrapped.conv2.weight.shape[0]))
            else:
                wrapped = _FoldedConvLikeWrapper(layer)
                if i == 0:
                    squeeze_input_channels.append(int(wrapped.weight.shape[1]))
                squeeze_input_channels.append(int(wrapped.weight.shape[0]))
            layers.append(wrapped)
        self.layers = nn.ModuleList(layers)
        self.squeeze_input_channels = squeeze_input_channels
        self.aggregation_squeeze_conv = _FoldedConvLikeWrapper(
            getattr(module, "aggregation_squeeze_conv")
        )
        self.aggregation_excitation_conv = _FoldedConvLikeWrapper(
            getattr(module, "aggregation_excitation_conv")
        )
        if self.aggregation_squeeze_conv.groups != 1:
            raise ValueError("Squeeze-accumulate wrapper only supports groups=1 squeeze convolutions")
        if tuple(int(v) for v in self.aggregation_squeeze_conv.weight.shape[2:]) != (1, 1):
            raise ValueError("Squeeze-accumulate wrapper only supports 1x1 squeeze convolutions")
        expected_channels = int(self.aggregation_squeeze_conv.weight.shape[1])
        if sum(self.squeeze_input_channels) != expected_channels:
            raise ValueError(
                "Squeeze-accumulate wrapper channel split does not match folded squeeze weight "
                f"({sum(self.squeeze_input_channels)} != {expected_channels})"
            )

    def _apply_squeeze_accumulate(self, outputs: List[torch.Tensor]) -> torch.Tensor:
        squeeze = self.aggregation_squeeze_conv
        acc: Optional[torch.Tensor] = None
        offset = 0
        for value, channels in zip(outputs, self.squeeze_input_channels):
            weight_slice = squeeze.weight[:, offset:offset + channels, :, :]
            part = F.conv2d(
                value,
                weight_slice,
                bias=None,
                stride=squeeze.stride,
                padding=squeeze.padding,
                dilation=squeeze.dilation,
                groups=1,
            )
            acc = part if acc is None else acc + part
            offset += channels
        if acc is None:
            raise RuntimeError("Squeeze-accumulate wrapper received no block outputs")
        acc = acc + squeeze.bias.view(1, -1, 1, 1)
        return _apply_folded_conv_like_activation(squeeze, acc)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        outputs = [x]
        for layer in self.layers:
            x = layer(x)
            outputs.append(x)
        x = self._apply_squeeze_accumulate(outputs)
        x = self.aggregation_excitation_conv(x)
        if self.identity:
            x = x + identity
        return x


class _HGV2StageFusedWrapper(nn.Module):
    """Inference-only PP-HGNetV2 stage wrapper that folds downsample and block internals."""

    def __init__(self, module: nn.Module):
        super().__init__()
        self.is_downsample = bool(getattr(module, "is_downsample", False))
        if self.is_downsample:
            self.downsample = _FoldedConvLikeWrapper(getattr(module, "downsample"))
        self.blocks = nn.Sequential(*[_HGV2BlockFusedWrapper(block) for block in getattr(module, "blocks")])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_downsample:
            x = self.downsample(x)
        return self.blocks(x)


class _HGV2StageSqueezeAccumulateWrapper(nn.Module):
    """Inference-only PP-HGNetV2 stage wrapper with concat-free squeeze accumulation."""

    def __init__(self, module: nn.Module):
        super().__init__()
        self.is_downsample = bool(getattr(module, "is_downsample", False))
        if self.is_downsample:
            self.downsample = _FoldedConvLikeWrapper(getattr(module, "downsample"))
        self.blocks = nn.Sequential(
            *[_HGV2BlockSqueezeAccumulateWrapper(block) for block in getattr(module, "blocks")]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_downsample:
            x = self.downsample(x)
        return self.blocks(x)


class _HGV2StagePreallocCatWrapper(nn.Module):
    """Inference-only PP-HGNetV2 stage wrapper with preallocated block concatenations."""

    def __init__(self, module: nn.Module):
        super().__init__()
        self.is_downsample = bool(getattr(module, "is_downsample", False))
        if self.is_downsample:
            self.downsample = _FoldedConvLikeWrapper(getattr(module, "downsample"))
        self.blocks = nn.Sequential(*[_HGV2BlockPreallocCatWrapper(block) for block in getattr(module, "blocks")])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_downsample:
            x = self.downsample(x)
        return self.blocks(x)


class _HGV2StageBlocksOnlyWrapper(nn.Module):
    """Keep the stage downsample exact but fold the inner HGV2 blocks."""

    def __init__(self, module: nn.Module):
        super().__init__()
        self.is_downsample = bool(getattr(module, "is_downsample", False))
        if self.is_downsample:
            self.downsample = getattr(module, "downsample")
        self.blocks = nn.Sequential(*[_HGV2BlockFusedWrapper(block) for block in getattr(module, "blocks")])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_downsample:
            x = self.downsample(x)
        return self.blocks(x)


class _StemBlockFusedWrapper(nn.Module):
    """Inference-only PP-HGNetV2 stem wrapper with ConvBNAct pairs folded."""

    def __init__(self, module: nn.Module):
        super().__init__()
        self.stem1 = _FoldedConvLikeWrapper(getattr(module, "stem1"))
        self.stem2a = _FoldedConvLikeWrapper(getattr(module, "stem2a"))
        self.stem2b = _FoldedConvLikeWrapper(getattr(module, "stem2b"))
        self.stem3 = _FoldedConvLikeWrapper(getattr(module, "stem3"))
        self.stem4 = _FoldedConvLikeWrapper(getattr(module, "stem4"))
        self.pool = getattr(module, "pool")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem1(x)
        x2 = self.stem2a(x)
        x2 = self.stem2b(x2)
        x1 = self.pool(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.stem3(x)
        x = self.stem4(x)
        return x


class _OpaqueModuleWrapper(nn.Module):
    """Delegate to the original module while hiding child paths from later replacements."""

    def __init__(self, module: nn.Module):
        super().__init__()
        self._wrapped = module

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self._wrapped(*args, **kwargs)


class _HGV2StageBlockCompileWrapper(nn.Module):
    """Stage wrapper that preserves stage boundaries while compiling each block privately."""

    def __init__(
        self,
        module: nn.Module,
        compile_module: Callable[[nn.Module], Any],
    ):
        super().__init__()
        self._downsample = None
        if bool(getattr(module, "is_downsample", False)):
            self._downsample = compile_module(_OpaqueModuleWrapper(getattr(module, "downsample")))
        self._blocks = nn.ModuleList(
            [compile_module(_OpaqueModuleWrapper(block)) for block in getattr(module, "blocks")]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._downsample is not None:
            x = self._downsample(x)
        for block in self._blocks:
            x = block(x)
        return x


class _EncoderWithSVTRFusedWrapper(nn.Module):
    """Inference-only wrapper that folds the encoder-side ConvBN layers around the SVTR blocks."""

    def __init__(self, module: nn.Module):
        super().__init__()
        self.use_guide = bool(getattr(module, "use_guide", False))
        self.conv1 = _FoldedConvLikeWrapper(getattr(module, "conv1"))
        self.conv2 = _FoldedConvLikeWrapper(getattr(module, "conv2"))
        self.svtr_block = getattr(module, "svtr_block")
        self.norm = getattr(module, "norm")
        self.conv3 = _FoldedConvLikeWrapper(getattr(module, "conv3"))
        self.conv4 = _FoldedConvLikeWrapper(getattr(module, "conv4"))
        self.conv1x1 = _FoldedConvLikeWrapper(getattr(module, "conv1x1"))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = x.clone() if self.use_guide else x
        h = z
        z = self.conv1(z)
        z = self.conv2(z)
        _, c, h_dim, w_dim = z.shape
        z = z.flatten(2).permute(0, 2, 1)
        for block in self.svtr_block:
            z = block(z)
        z = self.norm(z)
        z = z.reshape([-1, h_dim, w_dim, c]).permute(0, 3, 1, 2)
        z = self.conv3(z)
        z = torch.cat((h, z), dim=1)
        z = self.conv1x1(self.conv4(z))
        return z


class _SVTRAttentionSDPAWrapper(nn.Module):
    """Global SVTR attention rewritten with scaled_dot_product_attention."""

    def __init__(self, module: nn.Module):
        super().__init__()
        if getattr(module, "mixer", None) != "Global":
            raise ValueError("SVTR SDPA wrapper only supports global attention mixers")
        self.num_heads = int(getattr(module, "num_heads"))
        self.scale = float(getattr(module, "scale"))
        self.qkv = getattr(module, "qkv")
        self.proj = getattr(module, "proj")
        self.proj_drop = getattr(module, "proj_drop")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, tokens, channels = x.shape
        head_dim = channels // self.num_heads
        qkv = self.qkv(x).reshape(bsz, tokens, 3, self.num_heads, head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        x = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=0.0,
            scale=self.scale,
        )
        x = x.transpose(1, 2).reshape(bsz, tokens, channels)
        x = self.proj(x)
        return self.proj_drop(x)


class _SVTRBlockSDPAWrapper(nn.Module):
    """SVTR block wrapper that preserves the residual structure while swapping in SDPA."""

    def __init__(self, module: nn.Module):
        super().__init__()
        self.norm1 = getattr(module, "norm1")
        self.mixer = _SVTRAttentionSDPAWrapper(getattr(module, "mixer"))
        self.drop_path = getattr(module, "drop_path")
        self.norm2 = getattr(module, "norm2")
        self.mlp = getattr(module, "mlp")
        self.prenorm = bool(getattr(module, "prenorm", False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.prenorm:
            x = self.norm1(x + self.drop_path(self.mixer(x)))
            x = self.norm2(x + self.drop_path(self.mlp(x)))
        else:
            x = x + self.drop_path(self.mixer(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class _EncoderWithSVTRFusedSDPAWrapper(nn.Module):
    """Fold encoder-side ConvBN layers and replace SVTR attention blocks with SDPA."""

    def __init__(self, module: nn.Module):
        super().__init__()
        self.use_guide = bool(getattr(module, "use_guide", False))
        self.conv1 = _FoldedConvLikeWrapper(getattr(module, "conv1"))
        self.conv2 = _FoldedConvLikeWrapper(getattr(module, "conv2"))
        self.svtr_block = nn.ModuleList(
            [_SVTRBlockSDPAWrapper(block) for block in getattr(module, "svtr_block")]
        )
        self.norm = getattr(module, "norm")
        self.conv3 = _FoldedConvLikeWrapper(getattr(module, "conv3"))
        self.conv4 = _FoldedConvLikeWrapper(getattr(module, "conv4"))
        self.conv1x1 = _FoldedConvLikeWrapper(getattr(module, "conv1x1"))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = x.clone() if self.use_guide else x
        h = z
        z = self.conv1(z)
        z = self.conv2(z)
        _, c, h_dim, w_dim = z.shape
        z = z.flatten(2).permute(0, 2, 1)
        for block in self.svtr_block:
            z = block(z)
        z = self.norm(z)
        z = z.reshape([-1, h_dim, w_dim, c]).permute(0, 3, 1, 2)
        z = self.conv3(z)
        z = torch.cat((h, z), dim=1)
        z = self.conv1x1(self.conv4(z))
        return z


class _BatchNormWrapper(nn.Module):
    """Wraps BatchNorm modules to use an optimized kernel_fn."""

    def __init__(self, original: nn.Module, kernel_fn: Callable):
        super().__init__()
        self.original = original
        self.kernel_fn = kernel_fn
        self.weight = original.weight
        self.bias = original.bias
        self.running_mean = original.running_mean
        self.running_var = original.running_var
        self.eps = original.eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.kernel_fn(
            x,
            self.weight,
            self.bias,
            self.running_mean,
            self.running_var,
            eps=self.eps,
        )


class _LayerNormWrapper(nn.Module):
    """Wraps nn.LayerNorm to use an optimized kernel_fn."""

    def __init__(self, original: nn.LayerNorm, kernel_fn: Callable):
        super().__init__()
        self.original = original
        self.kernel_fn = kernel_fn
        self.weight = original.weight
        self.bias = original.bias
        self.eps = original.eps
        self.normalized_shape = original.normalized_shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Reshape if needed: kernel_fn expects (x, weight, bias[, eps])
        orig_shape = x.shape
        if x.dim() > 2:
            x_2d = x.reshape(-1, x.shape[-1])
        else:
            x_2d = x

        try:
            # Try full signature: kernel_fn(x, weight, bias, eps)
            out = self.kernel_fn(x_2d, self.weight, self.bias, self.eps)
        except TypeError:
            try:
                # Try without eps: kernel_fn(x, weight, bias)
                out = self.kernel_fn(x_2d, self.weight, self.bias)
            except TypeError:
                # Fallback: just x
                out = self.kernel_fn(x_2d)

        if len(orig_shape) > 2:
            out = out.reshape(orig_shape)

        return out


class _RMSNormWrapper(nn.Module):
    """Wraps RMSNorm-like modules to use an optimized kernel_fn."""

    def __init__(self, original: nn.Module, kernel_fn: Callable):
        super().__init__()
        self.original = original
        self.kernel_fn = kernel_fn
        # RMSNorm typically has a 'weight' attribute
        self.weight = getattr(original, "weight", None)
        self.eps = getattr(original, "eps", 1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        if x.dim() > 2:
            x_2d = x.reshape(-1, x.shape[-1])
        else:
            x_2d = x

        if self.weight is not None:
            try:
                out = self.kernel_fn(x_2d, self.weight, self.eps)
            except TypeError:
                out = self.kernel_fn(x_2d, self.weight)
        else:
            out = self.kernel_fn(x_2d)

        if len(orig_shape) > 2:
            out = out.reshape(orig_shape)

        return out


def _clone_graph_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        cloned = torch.empty_like(value, memory_format=torch.preserve_format)
        cloned.copy_(value)
        return cloned
    if isinstance(value, dict):
        return {k: _clone_graph_value(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(_clone_graph_value(v) for v in value)
    if isinstance(value, list):
        return [_clone_graph_value(v) for v in value]
    return value


def _graph_value_like(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return torch.empty_like(value, memory_format=torch.preserve_format)
    if isinstance(value, dict):
        return {k: _graph_value_like(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(_graph_value_like(v) for v in value)
    if isinstance(value, list):
        return [_graph_value_like(v) for v in value]
    return value


def _copy_graph_value_(dst: Any, src: Any) -> None:
    if isinstance(dst, torch.Tensor):
        if not isinstance(src, torch.Tensor):
            raise TypeError(f"Expected tensor input for CUDA graph replay, got {type(src)}")
        dst.copy_(src)
        return
    if isinstance(dst, dict):
        if not isinstance(src, dict) or dst.keys() != src.keys():
            raise TypeError("CUDA graph replay requires matching dict inputs")
        for key in dst:
            _copy_graph_value_(dst[key], src[key])
        return
    if isinstance(dst, tuple):
        if not isinstance(src, tuple) or len(dst) != len(src):
            raise TypeError("CUDA graph replay requires matching tuple inputs")
        for dst_item, src_item in zip(dst, src):
            _copy_graph_value_(dst_item, src_item)
        return
    if isinstance(dst, list):
        if not isinstance(src, list) or len(dst) != len(src):
            raise TypeError("CUDA graph replay requires matching list inputs")
        for dst_item, src_item in zip(dst, src):
            _copy_graph_value_(dst_item, src_item)
        return
    if dst != src:
        raise TypeError("CUDA graph replay requires static non-tensor inputs")


def _graph_values_share_storage(dst: Any, src: Any) -> bool:
    if isinstance(dst, torch.Tensor):
        return (
            isinstance(src, torch.Tensor)
            and dst.data_ptr() == src.data_ptr()
            and tuple(dst.shape) == tuple(src.shape)
            and tuple(dst.stride()) == tuple(src.stride())
            and dst.dtype == src.dtype
            and dst.device == src.device
        )
    if isinstance(dst, dict):
        return (
            isinstance(src, dict)
            and dst.keys() == src.keys()
            and all(_graph_values_share_storage(dst[key], src[key]) for key in dst)
        )
    if isinstance(dst, tuple):
        return (
            isinstance(src, tuple)
            and len(dst) == len(src)
            and all(_graph_values_share_storage(dst_item, src_item) for dst_item, src_item in zip(dst, src))
        )
    if isinstance(dst, list):
        return (
            isinstance(src, list)
            and len(dst) == len(src)
            and all(_graph_values_share_storage(dst_item, src_item) for dst_item, src_item in zip(dst, src))
        )
    return dst == src


def _call_model_forward(forward_fn: Callable[..., Any], model_input: Any) -> Any:
    if isinstance(model_input, dict):
        return forward_fn(**model_input)
    if isinstance(model_input, tuple):
        return forward_fn(*model_input)
    if isinstance(model_input, list):
        return forward_fn(*model_input)
    return forward_fn(model_input)


def _pair_or_none(value: Any) -> Optional[Tuple[int, int]]:
    if isinstance(value, tuple) and len(value) == 2:
        return int(value[0]), int(value[1])
    if isinstance(value, list) and len(value) == 2:
        return int(value[0]), int(value[1])
    if isinstance(value, int):
        return value, value
    return None


def _conv_signature_from_shape(shape: Dict[str, int]) -> Optional[Tuple[int, ...]]:
    keys = (
        "N", "C_in", "C_out", "H", "W", "KH", "KW",
        "stride_h", "stride_w", "pad_h", "pad_w", "dil_h", "dil_w", "groups",
    )
    if not all(k in shape for k in keys):
        return None
    return tuple(int(shape[k]) for k in keys)


def _conv_signature_from_inputs(
    x: torch.Tensor,
    weight: torch.Tensor,
    stride: Any,
    padding: Any,
    dilation: Any,
    groups: int,
) -> Optional[Tuple[int, ...]]:
    stride_hw = _pair_or_none(stride)
    padding_hw = _pair_or_none(padding)
    dilation_hw = _pair_or_none(dilation)
    if stride_hw is None or padding_hw is None or dilation_hw is None:
        return None
    return (
        int(x.shape[0]),
        int(x.shape[1]),
        int(weight.shape[0]),
        int(x.shape[2]),
        int(x.shape[3]),
        int(weight.shape[2]),
        int(weight.shape[3]),
        int(stride_hw[0]),
        int(stride_hw[1]),
        int(padding_hw[0]),
        int(padding_hw[1]),
        int(dilation_hw[0]),
        int(dilation_hw[1]),
        int(groups),
    )


def _batchnorm_signature_from_shape(shape: Dict[str, int]) -> Optional[Tuple[int, ...]]:
    keys = ("N", "C", "H", "W")
    if not all(k in shape for k in keys):
        return None
    return tuple(int(shape[k]) for k in keys)


def _batchnorm_signature_from_input(x: torch.Tensor) -> Optional[Tuple[int, ...]]:
    if x.ndim != 4:
        return None
    return (int(x.shape[0]), int(x.shape[1]), int(x.shape[2]), int(x.shape[3]))


def _conv_module_may_match_shape(shape: Dict[str, int], module: nn.Conv2d) -> bool:
    kernel_size = _pair_or_none(module.kernel_size)
    stride = _pair_or_none(module.stride)
    dilation = _pair_or_none(module.dilation)
    padding = _pair_or_none(module.padding)
    if kernel_size is None or stride is None or dilation is None or padding is None:
        return False
    return (
        shape.get("C_in") == module.in_channels
        and shape.get("C_out") == module.out_channels
        and shape.get("KH") == kernel_size[0]
        and shape.get("KW") == kernel_size[1]
        and shape.get("stride_h") == stride[0]
        and shape.get("stride_w") == stride[1]
        and shape.get("pad_h") == padding[0]
        and shape.get("pad_w") == padding[1]
        and shape.get("dil_h") == dilation[0]
        and shape.get("dil_w") == dilation[1]
        and shape.get("groups") == module.groups
    )


def _batchnorm_module_may_match_shape(shape: Dict[str, int], module: nn.Module) -> bool:
    num_features = getattr(module, "num_features", None)
    return num_features is not None and shape.get("C") == int(num_features)


def _get_named_parent(model: nn.Module, name: str) -> Tuple[Optional[nn.Module], Optional[str]]:
    if "." not in name:
        return model, name
    parts = name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


class OptimizedModelContext:
    """
    Context manager that patches a model's submodules to use optimized Triton kernels.

    Usage:
        with OptimizedModelContext(model, replacements) as patched_model:
            output = patched_model(input)
    """

    def __init__(self, model: nn.Module, replacements: List[KernelReplacement]):
        self.model = model
        self.replacements = replacements
        self._original_modules: Dict[str, nn.Module] = {}
        self._original_functionals: Dict[str, Any] = {}
        self._original_forward: Optional[Callable[..., Any]] = None
        self._restore_model_contiguous = False
        self._input_transform: Optional[Callable[[Any], Any]] = None
        self._applied: List[str] = []
        self._applied_replacements: List[Dict[str, Any]] = []
        self._skipped_replacements: List[Dict[str, Any]] = []
        # When graph_capture is in the replacement stack, torch.compile on
        # individual submodules is skipped — the CUDA graph itself eliminates
        # kernel launch overhead, and compiled function internals create
        # input copies that cause stale replay when captured in a graph.
        self._suppress_compile: bool = False

    def __enter__(self) -> nn.Module:
        self._applied.clear()
        self._applied_replacements.clear()
        self._skipped_replacements.clear()
        loaded_repls: List[KernelReplacement] = []
        for repl in self.replacements:
            try:
                kernel_mod = load_kernel_module(repl.optimized_path)
                if not hasattr(kernel_mod, "kernel_fn"):
                    print(f"  WARNING: {repl.optimized_path} has no kernel_fn, skipping")
                    self._skipped_replacements.append(
                        {
                            "type": repl.kernel_type,
                            "rank": repl.rank,
                            "speedup": repl.speedup,
                            "path": repl.optimized_path,
                            "reason": "Optimized kernel file has no kernel_fn.",
                        }
                    )
                    continue
                repl.module_fn = kernel_mod.kernel_fn
                repl.model_shapes = getattr(kernel_mod, "MODEL_SHAPES", None)
                repl.generic_fallback = bool(getattr(kernel_mod, "GENERIC_FALLBACK", False))
                repl.fuse_batchnorm_act = bool(getattr(kernel_mod, "FUSE_BATCHNORM_ACT", False))
                repl.global_channels_last = bool(getattr(kernel_mod, "GLOBAL_CHANNELS_LAST", False))
                repl.global_cuda_graph = bool(getattr(kernel_mod, "GLOBAL_CUDA_GRAPH", False))
                repl.cuda_graph_warmup_iters = int(getattr(kernel_mod, "CUDA_GRAPH_WARMUP_ITERS", 3))
                repl.cuda_graph_input_mode = str(
                    getattr(kernel_mod, "CUDA_GRAPH_INPUT_MODE", "clone")
                ).strip().lower()
                repl.cuda_graph_output_mode = str(
                    getattr(kernel_mod, "CUDA_GRAPH_OUTPUT_MODE", "clone")
                ).strip().lower()
                repl.block_fusion_kind = getattr(kernel_mod, "BLOCK_FUSION_KIND", None)
                target_modules = getattr(kernel_mod, "TARGET_MODULES", None)
                repl.target_modules = list(target_modules) if isinstance(target_modules, (list, tuple)) else None
                repl.torch_compile_backend = getattr(kernel_mod, "TORCH_COMPILE_BACKEND", None)
                repl.torch_compile_mode = getattr(kernel_mod, "TORCH_COMPILE_MODE", None)
                compile_options = getattr(kernel_mod, "TORCH_COMPILE_OPTIONS", None)
                repl.torch_compile_options = compile_options if isinstance(compile_options, dict) else None
                repl.verify_atol = getattr(kernel_mod, "VERIFY_ATOL", None)
                repl.verify_rtol = getattr(kernel_mod, "VERIFY_RTOL", None)
                loaded_repls.append(repl)
            except Exception as e:
                print(f"  WARNING: Failed to load {repl.optimized_path}: {e}")
                self._skipped_replacements.append(
                    {
                        "type": repl.kernel_type,
                        "rank": repl.rank,
                        "speedup": repl.speedup,
                        "path": repl.optimized_path,
                        "reason": f"Failed to load optimized kernel: {e}",
                    }
                )
                continue

        handled_group_types: set[str] = set()
        loaded_repls = self._resolve_replacement_conflicts(loaded_repls)

        # If graph_capture is in the stack, suppress torch.compile on other
        # kernels — compiled functions create internal input copies that
        # cause stale replay when captured inside a CUDA graph.
        self._suppress_compile = any(
            r.kernel_type == "graph_capture" and r.global_cuda_graph
            for r in loaded_repls
        )
        if self._suppress_compile:
            print("  NOTE: graph_capture active — suppressing torch.compile on submodules")

        grouped: Dict[str, List[KernelReplacement]] = {}
        for repl in loaded_repls:
            grouped.setdefault(repl.kernel_type, []).append(repl)

        for repl in loaded_repls:
            if repl.kernel_type in {"conv2d", "batchnorm"}:
                if repl.kernel_type in handled_group_types:
                    continue
                group = grouped.get(repl.kernel_type, [])
                replaced = self._apply_group_replacement(repl.kernel_type, group)
                handled_group_types.add(repl.kernel_type)
                target_repls = group
            else:
                replaced = self._apply_replacement(repl)
                target_repls = [repl]

            for target in target_repls:
                if replaced > 0:
                    speedup_label = f"{target.speedup:.1f}x" if target.speedup is not None else "n/a"
                    self._applied.append(
                        f"  {target.kernel_type} (rank {target.rank}): "
                        f"{speedup_label} -> {target.optimized_path}"
                    )
                    self._applied_replacements.append(
                        {
                            "type": target.kernel_type,
                            "rank": target.rank,
                            "speedup": target.speedup,
                            "path": target.optimized_path,
                            "modules_replaced": replaced,
                        }
                    )
                else:
                    if target.reinsert_supported:
                        reason = "No compatible modules found in the loaded model."
                    else:
                        reason = (
                            "No reinsertion strategy for this operator family. "
                            "Current verifier support is limited to matmul, conv2d, batchnorm, layout_transform, graph_capture, layernorm, rmsnorm, and softmax."
                        )
                    self._skipped_replacements.append(
                        {
                            "type": target.kernel_type,
                            "rank": target.rank,
                            "speedup": target.speedup,
                            "path": target.optimized_path,
                            "reason": reason,
                        }
                    )

        return self.model

    def _resolve_replacement_conflicts(
        self,
        replacements: List[KernelReplacement],
    ) -> List[KernelReplacement]:
        chosen_graph_capture: Optional[KernelReplacement] = None
        for repl in replacements:
            if repl.kernel_type != "graph_capture":
                continue
            if chosen_graph_capture is None or self._replacement_precedence(repl) >= self._replacement_precedence(chosen_graph_capture):
                chosen_graph_capture = repl

        chosen_block_targets: Dict[Tuple[str, ...], KernelReplacement] = {}
        for repl in replacements:
            if repl.kernel_type != "block_fusion" or not repl.target_modules:
                continue
            key = tuple(sorted(repl.target_modules))
            current = chosen_block_targets.get(key)
            if current is None or self._replacement_precedence(repl) >= self._replacement_precedence(current):
                chosen_block_targets[key] = repl

        resolved: List[KernelReplacement] = []
        for repl in replacements:
            if repl.kernel_type == "graph_capture":
                if chosen_graph_capture is not repl:
                    self._skipped_replacements.append(
                        {
                            "type": repl.kernel_type,
                            "rank": repl.rank,
                            "speedup": repl.speedup,
                            "path": repl.optimized_path,
                            "reason": "Superseded by a higher-priority graph capture candidate.",
                        }
                    )
                    continue
            if repl.kernel_type == "block_fusion" and repl.target_modules:
                key = tuple(sorted(repl.target_modules))
                chosen = chosen_block_targets.get(key)
                if chosen is not repl:
                    self._skipped_replacements.append(
                        {
                            "type": repl.kernel_type,
                            "rank": repl.rank,
                            "speedup": repl.speedup,
                            "path": repl.optimized_path,
                            "reason": (
                                "Superseded by a higher-priority block fusion candidate "
                                f"for {', '.join(repl.target_modules)}."
                            ),
                        }
                    )
                    continue
            resolved.append(repl)
        return resolved

    @staticmethod
    def _replacement_precedence(repl: KernelReplacement) -> Tuple[int, float, int]:
        status_score = 1 if repl.status == "optimizing" else 0
        speedup = float(repl.speedup or 0.0)
        return (status_score, speedup, repl.rank)

    def __exit__(self, *exc):
        # Restore all original modules
        for name in sorted(self._original_modules, key=lambda item: item.count(".")):
            original = self._original_modules[name]
            parts = name.split(".")
            parent = self.model
            for p in parts[:-1]:
                parent = getattr(parent, p)
            setattr(parent, parts[-1], original)
        self._original_modules.clear()
        if "softmax" in self._original_functionals:
            F.softmax = self._original_functionals["softmax"]
        if self._original_forward is not None:
            self.model.forward = self._original_forward
            self._original_forward = None
        self._restore_model_contiguous = False
        self._input_transform = None
        self._original_functionals.clear()

    def prepare_input(self, model_input: Any) -> Any:
        if self._input_transform is None:
            return model_input
        return self._input_transform(model_input)

    def _apply_replacement(self, repl: KernelReplacement) -> int:
        """
        Replace matching modules in the model. Returns number of modules replaced.
        """
        count = 0

        if repl.kernel_type == "matmul":
            count = self._replace_linear_modules(repl)
        elif repl.kernel_type == "conv2d":
            count = self._replace_conv2d_modules(repl)
        elif repl.kernel_type == "batchnorm":
            count = self._replace_batchnorm_modules(repl)
        elif repl.kernel_type == "layernorm":
            count = self._replace_layernorm_modules(repl)
        elif repl.kernel_type == "rmsnorm":
            count = self._replace_rmsnorm_modules(repl)
        elif repl.kernel_type == "softmax":
            count = self._replace_softmax_functional(repl)
        elif repl.kernel_type == "layout_transform":
            count = self._replace_layout_transform_strategy(repl)
        elif repl.kernel_type == "graph_capture":
            count = self._replace_cuda_graph_strategy(repl)
        elif repl.kernel_type == "block_fusion":
            count = self._replace_block_fusion_strategy(repl)
        else:
            print(f"  NOTE: No replacement strategy for kernel type '{repl.kernel_type}'. "
                  f"Skipping. (Supported: matmul, conv2d, batchnorm, layout_transform, graph_capture, block_fusion, layernorm, rmsnorm, softmax)")

        return count

    def _replace_layout_transform_strategy(self, repl: KernelReplacement) -> int:
        if not repl.global_channels_last:
            return 0

        kernel_fn = repl.module_fn
        self.model.to(memory_format=torch.channels_last)
        self._restore_model_contiguous = True

        def to_channels_last(value: torch.Tensor) -> torch.Tensor:
            if not (value.ndim == 4 and value.is_cuda):
                return value
            # No data_ptr cache — PyTorch reuses memory addresses across
            # allocations, so a data_ptr-keyed cache returns stale results
            # when a new tensor occupies the same address as a previous one.
            return kernel_fn(value)

        def transform_model_input(model_input: Any) -> Any:
            if isinstance(model_input, dict):
                return {
                    key: to_channels_last(value) if isinstance(value, torch.Tensor) else value
                    for key, value in model_input.items()
                }
            if isinstance(model_input, torch.Tensor):
                return to_channels_last(model_input)
            return model_input

        self._input_transform = transform_model_input
        return 1

    # Default width buckets for multi-width CUDA graph capture.
    # PaddleOCR2Pytorch's recogniser produces variable-width inputs (320–1280)
    # based on text crop aspect ratios.  Each bucket gets its own graph.
    GRAPH_WIDTH_BUCKETS: List[int] = [320, 480, 640, 800, 960, 1280]

    def _replace_cuda_graph_strategy(self, repl: KernelReplacement) -> int:
        if not repl.global_cuda_graph or not torch.cuda.is_available():
            return 0
        if self._original_forward is not None:
            return 0

        original_forward = self.model.forward
        self._original_forward = original_forward
        capture_forward = original_forward
        if (
            repl.torch_compile_backend is not None
            or repl.torch_compile_mode is not None
            or repl.torch_compile_options
        ):
            capture_forward = self._maybe_compile_callable(original_forward, repl)
        warmup_iters = max(int(repl.cuda_graph_warmup_iters), 1)
        output_mode = repl.cuda_graph_output_mode or "clone"
        buckets = list(self.GRAPH_WIDTH_BUCKETS)
        max_bucket = buckets[-1]

        # Per-bucket graph state: width -> (graph, static_input, static_output)
        graph_cache: Dict[int, Tuple[Any, Any, Any]] = {}
        # Private memory pool prevents other GPU allocations from
        # reusing addresses captured in the graph (avoids corruption
        # when multiple models share the same GPU).
        _graph_pool_id = torch.cuda.graph_pool_handle()

        def _snap_to_bucket(w: int) -> int:
            """Return the smallest bucket width >= w, or max_bucket."""
            for bw in buckets:
                if bw >= w:
                    return bw
            return max_bucket

        def _pad_to_width(x: torch.Tensor, target_w: int) -> torch.Tensor:
            """Right-pad a (B,C,H,W) tensor to target_w with zeros."""
            cur_w = x.shape[3]
            if cur_w == target_w:
                return x
            if cur_w > target_w:
                return x[:, :, :, :target_w]
            return torch.nn.functional.pad(x, (0, target_w - cur_w))

        def _capture_for_width(
            bucket_w: int, template: torch.Tensor,
        ) -> Tuple[Any, Any, Any]:
            """Warmup and capture a CUDA graph for the given bucket width."""
            static_in = _pad_to_width(template, bucket_w).clone()
            warmup_stream = torch.cuda.Stream()
            warmup_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(warmup_stream), torch.no_grad():
                for _ in range(warmup_iters):
                    _call_model_forward(capture_forward, static_in)
            torch.cuda.current_stream().wait_stream(warmup_stream)

            graph = torch.cuda.CUDAGraph()
            with torch.no_grad():
                with torch.cuda.graph(graph, pool=_graph_pool_id):
                    static_out = _call_model_forward(capture_forward, static_in)
            graph.replay()
            return graph, static_in, static_out

        def forward_with_multi_width_cuda_graph(*args: Any, **kwargs: Any) -> Any:
            if args and kwargs:
                raise TypeError("CUDA graph strategy does not support mixed args and kwargs")

            runtime_input: Any
            if kwargs:
                runtime_input = kwargs
            elif len(args) == 1:
                runtime_input = args[0]
            else:
                runtime_input = tuple(args)

            # Determine bucket from input width (assumes (B,C,H,W) tensor)
            if isinstance(runtime_input, torch.Tensor) and runtime_input.ndim == 4:
                actual_w = runtime_input.shape[3]
                bucket_w = _snap_to_bucket(actual_w)
                padded = _pad_to_width(runtime_input, bucket_w)

                if bucket_w not in graph_cache:
                    graph_cache[bucket_w] = _capture_for_width(bucket_w, padded)

                graph, static_in, static_out = graph_cache[bucket_w]
                # padded and static_in must be distinct tensors for copy_ to work
                if padded.data_ptr() != static_in.data_ptr():
                    static_in.copy_(padded)
                else:
                    # First call after capture: padded IS static_in (same storage).
                    # Force a real copy via clone to ensure graph sees fresh data.
                    static_in.copy_(padded.clone())
                graph.replay()

                if output_mode == "static":
                    return static_out
                return _clone_graph_value(static_out)

            # Fallback: non-tensor or unexpected shape — single graph path
            if "__fallback__" not in graph_cache:
                static_in = _clone_graph_value(runtime_input)
                warmup_stream = torch.cuda.Stream()
                warmup_stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(warmup_stream), torch.no_grad():
                    for _ in range(warmup_iters):
                        _call_model_forward(capture_forward, static_in)
                torch.cuda.current_stream().wait_stream(warmup_stream)
                graph = torch.cuda.CUDAGraph()
                with torch.no_grad():
                    with torch.cuda.graph(graph):
                        static_out = _call_model_forward(capture_forward, static_in)
                graph.replay()
                graph_cache["__fallback__"] = (graph, static_in, static_out)
            else:
                graph, static_in, static_out = graph_cache["__fallback__"]
                _copy_graph_value_(static_in, runtime_input)
                graph.replay()

            if output_mode == "static":
                return static_out
            return _clone_graph_value(static_out)

        self.model.forward = forward_with_multi_width_cuda_graph
        return 1

    def _replace_block_fusion_strategy(self, repl: KernelReplacement) -> int:
        targets = repl.target_modules or []
        if not targets:
            return 0

        count = 0
        for target in targets:
            parent, attr = _get_named_parent(self.model, target)
            if parent is None or attr is None or not hasattr(parent, attr):
                continue
            original = getattr(parent, attr)
            compile_after = True
            if repl.block_fusion_kind == "compile_original":
                replacement = original
            elif repl.block_fusion_kind == "hgv2_stage":
                replacement = _HGV2StageFusedWrapper(original)
            elif repl.block_fusion_kind == "hgv2_stage_prealloc_cat":
                replacement = _HGV2StagePreallocCatWrapper(original)
            elif repl.block_fusion_kind == "hgv2_stage_squeeze_accumulate":
                replacement = _HGV2StageSqueezeAccumulateWrapper(original)
            elif repl.block_fusion_kind == "hgv2_stage_compile_blocks":
                replacement = _HGV2StageBlockCompileWrapper(
                    original,
                    lambda module: self._maybe_compile_module(module, repl),
                )
                compile_after = False
            elif repl.block_fusion_kind == "hgv2_stage_blocks_only":
                replacement = _HGV2StageBlocksOnlyWrapper(original)
            elif repl.block_fusion_kind == "hgv2_block":
                replacement = _HGV2BlockFusedWrapper(original)
            elif repl.block_fusion_kind == "stem_block":
                replacement = _StemBlockFusedWrapper(original)
            elif repl.block_fusion_kind == "svtr_encoder":
                replacement = _EncoderWithSVTRFusedWrapper(original)
            elif repl.block_fusion_kind == "svtr_encoder_sdpa":
                replacement = _EncoderWithSVTRFusedSDPAWrapper(original)
            elif repl.block_fusion_kind == "compile_opaque":
                replacement = _OpaqueModuleWrapper(original)
            else:
                continue
            if compile_after:
                replacement = self._maybe_compile_module(replacement, repl)
            self._original_modules[target] = original
            setattr(parent, attr, replacement)
            count += 1
        return count

    def _maybe_compile_module(self, module: nn.Module, repl: KernelReplacement) -> Any:
        if self._suppress_compile:
            return module
        backend = repl.torch_compile_backend
        mode = repl.torch_compile_mode
        options = repl.torch_compile_options
        if backend is None and mode is None and not options:
            return module
        if not hasattr(torch, "compile"):
            print("  WARNING: torch.compile requested but not available; using eager wrapper")
            return module
        if mode is not None and options:
            print("  WARNING: torch.compile mode and options are mutually exclusive; using eager wrapper")
            return module

        kwargs: Dict[str, Any] = {}
        if backend is not None:
            kwargs["backend"] = backend
        if mode is not None:
            kwargs["mode"] = mode
        if options:
            kwargs["options"] = options

        try:
            _ensure_stdlib_profile_module_for_compile()
            return torch.compile(module, **kwargs)
        except Exception as exc:
            print(f"  WARNING: torch.compile failed for {repl.optimized_path}: {exc}")
            return module

    def _maybe_compile_callable(self, fn: Callable[..., Any], repl: KernelReplacement) -> Callable[..., Any]:
        if self._suppress_compile:
            return fn
        backend = repl.torch_compile_backend
        mode = repl.torch_compile_mode
        options = repl.torch_compile_options
        if backend is None and mode is None and not options:
            return fn
        if not hasattr(torch, "compile"):
            print("  WARNING: torch.compile requested but not available; using eager callable")
            return fn
        if mode is not None and options:
            print("  WARNING: torch.compile mode and options are mutually exclusive; using eager callable")
            return fn

        kwargs: Dict[str, Any] = {}
        if backend is not None:
            kwargs["backend"] = backend
        if mode is not None:
            kwargs["mode"] = mode
        if options:
            kwargs["options"] = options

        try:
            _ensure_stdlib_profile_module_for_compile()
            return torch.compile(fn, **kwargs)
        except Exception as exc:
            print(f"  WARNING: torch.compile failed for {repl.optimized_path}: {exc}")
            return fn

    def _apply_group_replacement(
        self,
        kernel_type: str,
        repls: List[KernelReplacement],
    ) -> int:
        if kernel_type == "conv2d":
            return self._replace_conv2d_modules_with_dispatch(repls)
        if kernel_type == "batchnorm":
            return self._replace_batchnorm_modules_with_dispatch(repls)
        return 0

    def _build_conv_dispatch_fn(self, repls: List[KernelReplacement]) -> Callable:
        specific_map: Dict[Tuple[int, ...], Callable] = {}
        for repl in sorted((r for r in repls if r.model_shapes), key=lambda r: r.rank):
            sig = _conv_signature_from_shape(repl.model_shapes)
            if sig is not None and sig not in specific_map:
                specific_map[sig] = repl.module_fn
        fallback = next((r.module_fn for r in repls if r.generic_fallback), None)
        if fallback is None and repls:
            fallback = repls[0].module_fn

        def dispatch(
            x: torch.Tensor,
            weight: torch.Tensor,
            bias: torch.Tensor | None = None,
            stride: int | tuple[int, int] = 1,
            padding: int | tuple[int, int] = 0,
            dilation: int | tuple[int, int] = 1,
            groups: int = 1,
        ) -> torch.Tensor:
            sig = _conv_signature_from_inputs(x, weight, stride, padding, dilation, groups)
            matched = specific_map.get(sig) if sig is not None else None
            if matched is not None:
                return matched(
                    x,
                    weight,
                    bias,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    groups=groups,
                )
            if fallback is not None:
                return fallback(
                    x,
                    weight,
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

        return dispatch

    def _build_batchnorm_dispatch_fn(self, repls: List[KernelReplacement]) -> Callable:
        specific_map: Dict[Tuple[int, ...], Callable] = {}
        for repl in sorted((r for r in repls if r.model_shapes), key=lambda r: r.rank):
            sig = _batchnorm_signature_from_shape(repl.model_shapes)
            if sig is not None and sig not in specific_map:
                specific_map[sig] = repl.module_fn
        fallback = next((r.module_fn for r in repls if r.generic_fallback), None)
        if fallback is None and repls:
            fallback = repls[0].module_fn

        def dispatch(
            x: torch.Tensor,
            weight: torch.Tensor | None,
            bias: torch.Tensor | None,
            running_mean: torch.Tensor,
            running_var: torch.Tensor,
            eps: float = 1e-5,
        ) -> torch.Tensor:
            sig = _batchnorm_signature_from_input(x)
            matched = specific_map.get(sig) if sig is not None else None
            if matched is not None:
                return matched(
                    x,
                    weight,
                    bias,
                    running_mean,
                    running_var,
                    eps=eps,
                )
            if fallback is not None:
                return fallback(
                    x,
                    weight,
                    bias,
                    running_mean,
                    running_var,
                    eps=eps,
                )
            return F.batch_norm(
                x,
                running_mean,
                running_var,
                weight,
                bias,
                training=False,
                eps=eps,
            )

        return dispatch

    def _replace_conv2d_modules_with_dispatch(self, repls: List[KernelReplacement]) -> int:
        dispatch_fn = self._build_conv_dispatch_fn(repls)
        has_generic = any(r.generic_fallback for r in repls)
        direct_kernel_fn = repls[0].module_fn if len(repls) == 1 and not has_generic else None
        wrapper_kernel_fn = direct_kernel_fn or dispatch_fn
        candidate_shapes = [r.model_shapes for r in repls if isinstance(r.model_shapes, dict)]
        fusion_shapes = [
            r.model_shapes
            for r in repls
            if r.fuse_batchnorm_act and isinstance(r.model_shapes, dict)
        ]
        count = 0
        replaced_parent_names: set[str] = set()
        for name, module in list(self.model.named_modules()):
            if not isinstance(module, nn.Conv2d):
                continue
            parent_name = name.rsplit(".", 1)[0] if "." in name else ""
            if parent_name in replaced_parent_names:
                continue
            if not has_generic and candidate_shapes:
                if not any(_conv_module_may_match_shape(shape, module) for shape in candidate_shapes):
                    continue

            fused_parent = None
            fused_norm = None
            fused_act = None
            if fusion_shapes and any(_conv_module_may_match_shape(shape, module) for shape in fusion_shapes):
                parent_module, parent_attr = _get_named_parent(self.model, parent_name)
                if parent_module is not None and parent_attr:
                    fused_parent = getattr(parent_module, parent_attr, None)
                    fused_norm = (
                        getattr(fused_parent, "bn", None)
                        or getattr(fused_parent, "norm", None)
                    )
                    fused_act = getattr(fused_parent, "act", None)
                    if not (
                        hasattr(fused_parent, "conv")
                        and getattr(fused_parent, "conv", None) is module
                        and isinstance(fused_norm, nn.BatchNorm2d)
                        and not getattr(fused_norm, "training", True)
                        and isinstance(fused_act, nn.Module)
                    ):
                        fused_parent = None

            if fused_parent is not None and parent_name:
                grandparent, parent_attr = _get_named_parent(self.model, parent_name)
                if grandparent is not None and parent_attr:
                    self._original_modules[parent_name] = fused_parent
                    setattr(
                        grandparent,
                        parent_attr,
                        _ConvBnActFusedWrapper(
                            fused_parent.conv,
                            fused_norm,
                            fused_act,
                            wrapper_kernel_fn,
                        ),
                    )
                    replaced_parent_names.add(parent_name)
                    count += 1
                    continue

            self._original_modules[name] = module
            wrapper = _Conv2dWrapper(module, wrapper_kernel_fn)
            parent_module, attr = _get_named_parent(self.model, name)
            if parent_module is None or attr is None:
                continue
            setattr(parent_module, attr, wrapper)
            count += 1
        return count

    def _replace_batchnorm_modules_with_dispatch(self, repls: List[KernelReplacement]) -> int:
        dispatch_fn = self._build_batchnorm_dispatch_fn(repls)
        has_generic = any(r.generic_fallback for r in repls)
        candidate_shapes = [r.model_shapes for r in repls if isinstance(r.model_shapes, dict)]
        count = 0
        for name, module in list(self.model.named_modules()):
            if not isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                continue
            if not has_generic and candidate_shapes:
                if not any(_batchnorm_module_may_match_shape(shape, module) for shape in candidate_shapes):
                    continue
            self._original_modules[name] = module
            wrapper = _BatchNormWrapper(module, dispatch_fn)
            parts = name.split(".")
            parent = self.model
            for p in parts[:-1]:
                parent = getattr(parent, p)
            setattr(parent, parts[-1], wrapper)
            count += 1
        return count

    def _replace_softmax_functional(self, repl: KernelReplacement) -> int:
        """Monkeypatch F.softmax for the exact PP-OCR recognizer logits shape."""
        if "softmax" not in self._original_functionals:
            self._original_functionals["softmax"] = F.softmax

        original_softmax = self._original_functionals["softmax"]
        kernel_fn = repl.module_fn

        def patched_softmax(
            input: torch.Tensor,
            dim: Optional[int] = None,
            _stacklevel: int = 3,
            dtype: Optional[torch.dtype] = None,
        ) -> torch.Tensor:
            if (
                isinstance(input, torch.Tensor)
                and input.is_cuda
                and dtype is None
                and dim in (-1, input.dim() - 1)
                and tuple(input.shape) in {(40, 18385), (1, 40, 18385)}
            ):
                return kernel_fn(input)
            return original_softmax(input, dim=dim, _stacklevel=_stacklevel, dtype=dtype)

        F.softmax = patched_softmax
        return 1

    def _replace_linear_modules(self, repl: KernelReplacement) -> int:
        """Replace all nn.Linear modules with optimized matmul wrapper."""
        count = 0
        for name, module in list(self.model.named_modules()):
            if isinstance(module, nn.Linear):
                # Save original
                self._original_modules[name] = module
                # Create wrapper
                wrapper = _LinearWrapper(module, repl.module_fn)
                # Install wrapper
                parts = name.split(".")
                parent = self.model
                for p in parts[:-1]:
                    parent = getattr(parent, p)
                setattr(parent, parts[-1], wrapper)
                count += 1
        return count

    def _replace_conv2d_modules(self, repl: KernelReplacement) -> int:
        """Replace all nn.Conv2d modules with optimized wrappers."""
        count = 0
        for name, module in list(self.model.named_modules()):
            if isinstance(module, nn.Conv2d):
                self._original_modules[name] = module
                wrapper = _Conv2dWrapper(module, repl.module_fn)
                parts = name.split(".")
                parent = self.model
                for p in parts[:-1]:
                    parent = getattr(parent, p)
                setattr(parent, parts[-1], wrapper)
                count += 1
        return count

    def _replace_batchnorm_modules(self, repl: KernelReplacement) -> int:
        """Replace all BatchNorm modules with optimized wrappers."""
        count = 0
        for name, module in list(self.model.named_modules()):
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                self._original_modules[name] = module
                wrapper = _BatchNormWrapper(module, repl.module_fn)
                parts = name.split(".")
                parent = self.model
                for p in parts[:-1]:
                    parent = getattr(parent, p)
                setattr(parent, parts[-1], wrapper)
                count += 1
        return count

    def _replace_layernorm_modules(self, repl: KernelReplacement) -> int:
        """Replace all nn.LayerNorm modules with optimized wrapper."""
        count = 0
        for name, module in list(self.model.named_modules()):
            if isinstance(module, nn.LayerNorm):
                self._original_modules[name] = module
                wrapper = _LayerNormWrapper(module, repl.module_fn)
                parts = name.split(".")
                parent = self.model
                for p in parts[:-1]:
                    parent = getattr(parent, p)
                setattr(parent, parts[-1], wrapper)
                count += 1
        return count

    def _replace_rmsnorm_modules(self, repl: KernelReplacement) -> int:
        """
        Replace RMSNorm modules. Since there is no standard nn.RMSNorm,
        we look for common class names and attributes.
        """
        count = 0
        rmsnorm_names = {"RMSNorm", "LlamaRMSNorm", "T5LayerNorm", "GemmaRMSNorm"}

        for name, module in list(self.model.named_modules()):
            cls_name = type(module).__name__
            # Match by class name or by having 'weight' but no 'bias' and a norm-like name
            is_rmsnorm = (
                cls_name in rmsnorm_names
                or (hasattr(module, "weight")
                    and hasattr(module, "eps")
                    and not hasattr(module, "bias")
                    and cls_name.lower().endswith("norm")
                    and not isinstance(module, nn.LayerNorm))
            )

            if is_rmsnorm:
                self._original_modules[name] = module
                wrapper = _RMSNormWrapper(module, repl.module_fn)
                parts = name.split(".")
                parent = self.model
                for p in parts[:-1]:
                    parent = getattr(parent, p)
                setattr(parent, parts[-1], wrapper)
                count += 1
        return count

    @property
    def applied_summary(self) -> List[str]:
        return self._applied

    @property
    def applied_replacements(self) -> List[Dict[str, Any]]:
        return list(self._applied_replacements)

    @property
    def skipped_replacements(self) -> List[Dict[str, Any]]:
        return list(self._skipped_replacements)


# ---------------------------------------------------------------------------
# 5. Output Comparison
# ---------------------------------------------------------------------------

def extract_tensor(output: Any) -> torch.Tensor:
    """
    Extract a single tensor from model output, which might be a tuple, dict,
    or ModelOutput-like object.
    """
    if isinstance(output, torch.Tensor):
        return output

    # HuggingFace ModelOutput or similar dataclass-like object
    if hasattr(output, "logits"):
        return output.logits
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state

    # Tuple/list: return first tensor element
    if isinstance(output, (tuple, list)):
        for item in output:
            if isinstance(item, torch.Tensor):
                return item
        # Recurse into first element
        if len(output) > 0:
            return extract_tensor(output[0])

    # Dict: try common keys
    if isinstance(output, dict):
        for key in ["logits", "last_hidden_state", "output", "hidden_states"]:
            if key in output and isinstance(output[key], torch.Tensor):
                return output[key]
        # Return first tensor value
        for v in output.values():
            if isinstance(v, torch.Tensor):
                return v

    raise ValueError(
        f"Cannot extract tensor from output of type {type(output)}. "
        f"Consider adding support for this output format."
    )


def compare_outputs(
    ref_output: torch.Tensor,
    opt_output: torch.Tensor,
    dtype: torch.dtype,
    custom_atol: Optional[float] = None,
    custom_rtol: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Compare reference and optimized outputs. Returns comparison metrics.
    """
    result: Dict[str, Any] = {}

    # Shape check
    result["shapes_match"] = ref_output.shape == opt_output.shape
    result["ref_shape"] = str(list(ref_output.shape))
    result["opt_shape"] = str(list(opt_output.shape))

    if not result["shapes_match"]:
        result["correctness"] = "FAIL"
        result["reason"] = f"Shape mismatch: ref={result['ref_shape']}, opt={result['opt_shape']}"
        return result

    # NaN / Inf check
    ref_float = ref_output.float()
    opt_float = opt_output.float()

    result["ref_has_nan"] = bool(torch.isnan(ref_float).any())
    result["ref_has_inf"] = bool(torch.isinf(ref_float).any())
    result["opt_has_nan"] = bool(torch.isnan(opt_float).any())
    result["opt_has_inf"] = bool(torch.isinf(opt_float).any())

    if result["opt_has_nan"] and not result["ref_has_nan"]:
        result["correctness"] = "FAIL"
        result["reason"] = "Optimized output contains NaN where reference does not"
        return result

    if result["opt_has_inf"] and not result["ref_has_inf"]:
        result["correctness"] = "FAIL"
        result["reason"] = "Optimized output contains Inf where reference does not"
        return result

    # Numerical comparison
    diff = (ref_float - opt_float).abs()

    # Mask out positions where both are NaN (those are fine)
    valid_mask = ~(torch.isnan(ref_float) & torch.isnan(opt_float))
    if valid_mask.any():
        valid_diff = diff[valid_mask]
        result["max_abs_error"] = float(valid_diff.max())
        result["mean_abs_error"] = float(valid_diff.mean())
    else:
        result["max_abs_error"] = 0.0
        result["mean_abs_error"] = 0.0

    # Tolerance check
    tols = DEFAULT_TOLERANCES.get(dtype, {"atol": 1e-4, "rtol": 1e-4})
    atol = custom_atol if custom_atol is not None else tols["atol"]
    rtol = custom_rtol if custom_rtol is not None else tols["rtol"]

    # Use allclose on the valid (non-NaN) elements
    if valid_mask.any():
        passes = torch.allclose(
            ref_float[valid_mask], opt_float[valid_mask], atol=atol, rtol=rtol
        )
    else:
        passes = True

    result["correctness"] = "PASS" if passes else "FAIL"
    result["atol"] = atol
    result["rtol"] = rtol

    if not passes:
        result["reason"] = (
            f"Values exceed tolerance (atol={atol}, rtol={rtol}). "
            f"max_abs_error={result['max_abs_error']:.6e}, "
            f"mean_abs_error={result['mean_abs_error']:.6e}"
        )

    return result


# ---------------------------------------------------------------------------
# 6. Diagnosis Mode (apply kernels one at a time)
# ---------------------------------------------------------------------------

def diagnose_kernel_failures(
    model: nn.Module,
    model_input: Union[torch.Tensor, Dict[str, torch.Tensor]],
    ref_tensor: torch.Tensor,
    replacements: List[KernelReplacement],
    dtype: torch.dtype,
) -> List[Dict[str, Any]]:
    """
    Apply each kernel replacement individually to find which one causes failure.
    """
    results = []

    for repl in replacements:
        print(f"\n  Testing kernel: {repl.kernel_type} (rank {repl.rank})...")
        ctx = OptimizedModelContext(model, [repl])

        try:
            with ctx as patched_model:
                opt_input = ctx.prepare_input(model_input)
                with torch.no_grad():
                    if isinstance(opt_input, dict):
                        opt_output = patched_model(**opt_input)
                    else:
                        opt_output = patched_model(opt_input)
                torch.cuda.synchronize()

            opt_tensor = extract_tensor(opt_output)
            comp = compare_outputs(
                ref_tensor,
                opt_tensor,
                dtype,
                repl.verify_atol,
                repl.verify_rtol,
            )

            results.append({
                "kernel_type": repl.kernel_type,
                "rank": repl.rank,
                "path": repl.optimized_path,
                "correctness": comp["correctness"],
                "max_abs_error": comp.get("max_abs_error", 0.0),
                "mean_abs_error": comp.get("mean_abs_error", 0.0),
                "reason": comp.get("reason", ""),
            })

            status = comp["correctness"]
            if status == "PASS":
                print(f"    -> PASS (max_err={comp.get('max_abs_error', 0):.6e})")
            else:
                print(f"    -> FAIL: {comp.get('reason', 'unknown')}")

        except Exception as e:
            results.append({
                "kernel_type": repl.kernel_type,
                "rank": repl.rank,
                "path": repl.optimized_path,
                "correctness": "ERROR",
                "max_abs_error": float("inf"),
                "mean_abs_error": float("inf"),
                "reason": str(e),
            })
            print(f"    -> ERROR: {e}")

    return results


# ---------------------------------------------------------------------------
# 7. Output Formatting
# ---------------------------------------------------------------------------

def format_report(result: VerificationResult, diagnose_results: Optional[List] = None) -> str:
    """Format the verification result into a human-readable report."""
    lines = []
    lines.append("")
    lines.append("=== AutoKernel End-to-End Verification ===")
    lines.append("")
    lines.append(f"Model: {result.model_name}")
    lines.append(f"Input: [{result.input_shape}], dtype={result.dtype_str}")
    lines.append(f"GPU: {result.gpu_name}")

    # Reference run
    lines.append("")
    lines.append("--- Reference Run ---")
    lines.append(f"Output shape: {result.ref_output_shape}")
    lines.append(f"Latency: {result.ref_latency_ms:.1f} ms ({TIMED_RUNS} runs, median)")

    # Optimized run
    lines.append("")
    lines.append("--- Optimized Run ---")
    if result.kernels_replaced:
        lines.append("Kernels replaced:")
        for k in result.kernels_replaced:
            speedup_label = f"{k['speedup']:.1f}x" if k.get("speedup") is not None else "n/a"
            lines.append(f"  {k['type']} (rank {k['rank']}): "
                         f"{speedup_label} -> {k['path']} "
                         f"({k['modules_replaced']} modules)")
    else:
        lines.append("Kernels replaced: none")
    if result.kernels_skipped:
        lines.append("Kernels skipped:")
        for k in result.kernels_skipped:
            lines.append(f"  {k['type']} (rank {k['rank']}): {k['reason']}")
    lines.append(f"Output shape: {result.opt_output_shape}")
    lines.append(f"Latency: {result.opt_latency_ms:.1f} ms ({TIMED_RUNS} runs, median)")

    # Verification
    lines.append("")
    lines.append("--- Verification ---")
    lines.append(f"correctness: {result.correctness}")
    lines.append(f"max_abs_error: {result.max_abs_error:.2e}")
    lines.append(f"mean_abs_error: {result.mean_abs_error:.2e}")
    if result.has_nan:
        lines.append("WARNING: NaN detected in optimized output")
    if result.has_inf:
        lines.append("WARNING: Inf detected in optimized output")

    # Summary
    lines.append("")
    lines.append("--- Summary ---")
    lines.append(f"original_latency_ms: {result.ref_latency_ms:.1f}")
    lines.append(f"optimized_latency_ms: {result.opt_latency_ms:.1f}")
    lines.append(f"end_to_end_speedup: {result.end_to_end_speedup:.2f}x")
    lines.append(f"kernels_replaced: {len(result.kernels_replaced)}")
    lines.append(f"kernels_skipped: {len(result.kernels_skipped)}")

    # Diagnosis
    if diagnose_results:
        lines.append("")
        lines.append("--- Diagnosis (per-kernel) ---")
        for dr in diagnose_results:
            status = dr["correctness"]
            line = f"  {dr['kernel_type']} (rank {dr['rank']}): {status}"
            if status == "PASS":
                line += f" | max_err={dr['max_abs_error']:.2e}"
            if dr.get("reason"):
                line += f" | {dr['reason']}"
            lines.append(line)

    lines.append("")
    return "\n".join(lines)


def save_verification_json(result: VerificationResult, path: str) -> None:
    """Save verification results as JSON for programmatic consumption."""
    data = {
        "model": result.model_name,
        "input_shape": result.input_shape,
        "dtype": result.dtype_str,
        "gpu": result.gpu_name,
        "reference": {
            "output_shape": result.ref_output_shape,
            "latency_ms": round(result.ref_latency_ms, 2),
        },
        "optimized": {
            "output_shape": result.opt_output_shape,
            "latency_ms": round(result.opt_latency_ms, 2),
            "kernels_replaced": result.kernels_replaced,
            "kernels_skipped": result.kernels_skipped,
        },
        "verification": {
            "correctness": result.correctness,
            "max_abs_error": result.max_abs_error,
            "mean_abs_error": result.mean_abs_error,
            "has_nan": result.has_nan,
            "has_inf": result.has_inf,
        },
        "summary": {
            "original_latency_ms": round(result.ref_latency_ms, 2),
            "optimized_latency_ms": round(result.opt_latency_ms, 2),
            "end_to_end_speedup": round(result.end_to_end_speedup, 3),
            "kernels_replaced": len(result.kernels_replaced),
            "kernels_skipped": len(result.kernels_skipped),
        },
    }

    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_dtype(dtype_str: str) -> torch.dtype:
    """Parse a dtype string into a torch.dtype."""
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
        "float": torch.float32,
    }
    key = dtype_str.lower().strip()
    if key not in mapping:
        raise ValueError(f"Unknown dtype '{dtype_str}'. Choose from: {list(mapping.keys())}")
    return mapping[key]


def _get_gpu_name() -> str:
    """Get current GPU name."""
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return "No GPU"


def _output_shape_str(output: Any) -> str:
    """Get shape string from model output."""
    try:
        t = extract_tensor(output)
        return str(list(t.shape))
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    global WORKSPACE_DIR, ORCHESTRATION_STATE, WARMUP_RUNS, TIMED_RUNS

    parser = argparse.ArgumentParser(
        description="AutoKernel End-to-End Verifier",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Model loading
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument(
        "--model", type=str,
        help="Path to a Python file containing the model class"
    )
    model_group.add_argument(
        "--module", type=str,
        help="Python module name (e.g. 'transformers')"
    )

    parser.add_argument(
        "--class-name", type=str, required=True,
        help="Name of the model class to instantiate"
    )
    parser.add_argument(
        "--pretrained", type=str, default=None,
        help="Pretrained model name/path (for HuggingFace models)"
    )
    parser.add_argument(
        "--input-shape", type=str, default="1,2048",
        help="Comma-separated input shape, e.g. '1,2048' (default: 1,2048)"
    )
    parser.add_argument(
        "--dtype", type=str, default="float16",
        help="Data type: float16, bfloat16, float32 (default: float16)"
    )

    # Benchmark tuning
    parser.add_argument(
        "--warmup", type=int, default=WARMUP_RUNS,
        help=f"Number of warmup iterations (default: {WARMUP_RUNS})"
    )
    parser.add_argument(
        "--timed", type=int, default=TIMED_RUNS,
        help=f"Number of timed iterations (default: {TIMED_RUNS})"
    )

    # Tolerance overrides
    parser.add_argument("--atol", type=float, default=None, help="Override absolute tolerance")
    parser.add_argument("--rtol", type=float, default=None, help="Override relative tolerance")

    # Modes
    parser.add_argument(
        "--diagnose", action="store_true",
        help="On failure, test each kernel replacement individually to find the culprit"
    )
    parser.add_argument(
        "--json", type=str, default=None,
        help="Save results to a JSON file at this path"
    )
    parser.add_argument(
        "--workspace", type=str, default=None,
        help="Override workspace directory (default: ./workspace)"
    )

    args = parser.parse_args()

    # Override globals if workspace specified
    if args.workspace:
        WORKSPACE_DIR = os.path.abspath(args.workspace)
        ORCHESTRATION_STATE = os.path.join(WORKSPACE_DIR, "orchestration_state.json")

    WARMUP_RUNS = args.warmup
    TIMED_RUNS = args.timed

    dtype = _parse_dtype(args.dtype)
    gpu_name = _get_gpu_name()

    print("=" * 60)
    print("  AutoKernel End-to-End Verifier")
    print("=" * 60)
    print()

    # -----------------------------------------------------------------------
    # Step 1: Discover optimized kernels
    # -----------------------------------------------------------------------
    print("Step 1: Discovering optimized kernels...")
    replacements = discover_optimized_kernels()
    if not replacements:
        print()
        print("No optimized kernels found.")
        print(f"  Searched: {WORKSPACE_DIR}")
        print(f"  State file: {ORCHESTRATION_STATE}")
        print()
        print("Run the optimization loop first to produce optimized kernels.")
        print("Expected files: workspace/kernel_<type>_<rank>_optimized.py")
        sys.exit(1)

    print(f"  Found {len(replacements)} optimized kernel(s):")
    for r in replacements:
        reinsert_label = "YES" if r.reinsert_supported else "no"
        speedup_label = f"{r.speedup:.1f}x" if r.speedup is not None else "n/a"
        print(
            f"    {r.kernel_type} (rank {r.rank}): "
            f"speedup={speedup_label}, reinsert={reinsert_label} -> {r.optimized_path}"
        )
    print()

    reference_replacements, optimized_replacements = split_replacements_for_verification(replacements)

    # -----------------------------------------------------------------------
    # Step 2: Load model
    # -----------------------------------------------------------------------
    print("Step 2: Loading model...")
    try:
        model = load_model(args)
        model_name = args.class_name
        if args.pretrained:
            model_name = f"{args.class_name} ({args.pretrained})"
        print(f"  Model loaded: {model_name}")
        param_count = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {param_count:,}")
    except Exception as e:
        print(f"\nERROR: Failed to load model: {e}")
        traceback.print_exc()
        sys.exit(1)
    print()

    # -----------------------------------------------------------------------
    # Step 3: Create input
    # -----------------------------------------------------------------------
    print("Step 3: Creating model input...")
    try:
        model_input = make_model_input(model, args.input_shape, dtype)
        if isinstance(model_input, dict):
            for k, v in model_input.items():
                print(f"  {k}: shape={list(v.shape)}, dtype={v.dtype}")
        else:
            print(f"  Input: shape={list(model_input.shape)}, dtype={model_input.dtype}")
    except Exception as e:
        print(f"\nERROR: Failed to create input: {e}")
        traceback.print_exc()
        sys.exit(1)
    print()

    # -----------------------------------------------------------------------
    # Step 4: Reference run
    # -----------------------------------------------------------------------
    if reference_replacements:
        print("Step 4: Reference run (current accepted optimized stack)...")
    else:
        print("Step 4: Reference run (original PyTorch ops)...")
    try:
        if reference_replacements:
            ref_ctx = OptimizedModelContext(model, reference_replacements)
            with ref_ctx as baseline_model:
                if ref_ctx.applied_summary:
                    print("  Baseline replacements applied:")
                    for line in ref_ctx.applied_summary:
                        print(f"  {line}")
                ref_input = ref_ctx.prepare_input(model_input)
                ref_output, ref_latency = benchmark_model(
                    baseline_model, ref_input, WARMUP_RUNS, TIMED_RUNS
                )
                ref_tensor = extract_tensor(ref_output)
                ref_shape_str = str(list(ref_tensor.shape))
                print(f"  Output shape: {ref_shape_str}")
                print(f"  Median latency: {ref_latency:.1f} ms")
        else:
            ref_output, ref_latency = benchmark_model(model, model_input, WARMUP_RUNS, TIMED_RUNS)
            ref_tensor = extract_tensor(ref_output)
            ref_shape_str = str(list(ref_tensor.shape))
            print(f"  Output shape: {ref_shape_str}")
            print(f"  Median latency: {ref_latency:.1f} ms")
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"\nERROR: GPU out of memory during reference run.")
            print("  Try a smaller --input-shape or a smaller model.")
            torch.cuda.empty_cache()
            sys.exit(1)
        else:
            raise
    except Exception as e:
        print(f"\nERROR: Reference run failed: {e}")
        traceback.print_exc()
        sys.exit(1)
    print()

    # -----------------------------------------------------------------------
    # Step 5: Optimized run
    # -----------------------------------------------------------------------
    print("Step 5: Optimized run (with Triton kernel replacements)...")
    ctx = OptimizedModelContext(model, optimized_replacements)
    try:
        with ctx as patched_model:
            opt_input = ctx.prepare_input(model_input)
            if ctx.applied_summary:
                print("  Replacements applied:")
                for line in ctx.applied_summary:
                    print(f"  {line}")
            else:
                print("  WARNING: No kernel replacements could be applied to this model.")
                print("  The model may not contain compatible modules, or the operator family")
                print("  may not have a reinsertion strategy in verify.py yet.")
            if ctx.skipped_replacements:
                print("  Replacements skipped:")
                for skipped in ctx.skipped_replacements:
                    print(f"    {skipped['type']} (rank {skipped['rank']}): {skipped['reason']}")

            opt_output, opt_latency = benchmark_model(
                patched_model, opt_input, WARMUP_RUNS, TIMED_RUNS
            )
            opt_tensor = extract_tensor(opt_output)
            opt_shape_str = str(list(opt_tensor.shape))
            print(f"  Output shape: {opt_shape_str}")
            print(f"  Median latency: {opt_latency:.1f} ms")
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"\nERROR: GPU out of memory during optimized run.")
            print("  The optimized kernels may use more memory than expected.")
            torch.cuda.empty_cache()
            sys.exit(1)
        else:
            print(f"\nERROR: Optimized run failed: {e}")
            traceback.print_exc()
            sys.exit(1)
    except Exception as e:
        print(f"\nERROR: Optimized run failed: {e}")
        traceback.print_exc()
        sys.exit(1)
    print()

    # -----------------------------------------------------------------------
    # Step 6: Compare outputs
    # -----------------------------------------------------------------------
    print("Step 6: Comparing outputs...")
    effective_atol = args.atol
    effective_rtol = args.rtol
    if effective_atol is None:
        replacement_atols = [r.verify_atol for r in replacements if r.verify_atol is not None]
        if replacement_atols:
            effective_atol = max(replacement_atols)
    if effective_rtol is None:
        replacement_rtls = [r.verify_rtol for r in replacements if r.verify_rtol is not None]
        if replacement_rtls:
            effective_rtol = max(replacement_rtls)

    comp = compare_outputs(ref_tensor, opt_tensor, dtype, effective_atol, effective_rtol)
    print(f"  correctness: {comp['correctness']}")
    print(f"  max_abs_error: {comp.get('max_abs_error', 0):.2e}")
    print(f"  mean_abs_error: {comp.get('mean_abs_error', 0):.2e}")
    if comp.get("reason"):
        print(f"  reason: {comp['reason']}")
    print()

    # -----------------------------------------------------------------------
    # Step 6b: Diagnose failures if requested
    # -----------------------------------------------------------------------
    diagnose_results = None
    if args.diagnose and comp["correctness"] == "FAIL":
        print("Step 6b: Diagnosing failure (testing each kernel individually)...")
        diagnose_results = diagnose_kernel_failures(
            model, model_input, ref_tensor, replacements, dtype
        )
        print()

    # -----------------------------------------------------------------------
    # Step 7: Build and display final report
    # -----------------------------------------------------------------------
    speedup = ref_latency / opt_latency if opt_latency > 0 else 0.0

    result = VerificationResult(
        model_name=model_name if args.pretrained else args.class_name,
        input_shape=args.input_shape,
        dtype_str=args.dtype,
        gpu_name=gpu_name,
        ref_output_shape=ref_shape_str,
        ref_latency_ms=ref_latency,
        opt_output_shape=opt_shape_str,
        opt_latency_ms=opt_latency,
        kernels_replaced=[dict(entry) for entry in ctx.applied_replacements],
        kernels_skipped=[dict(entry) for entry in ctx.skipped_replacements],
        correctness=comp["correctness"],
        max_abs_error=comp.get("max_abs_error", 0.0),
        mean_abs_error=comp.get("mean_abs_error", 0.0),
        has_nan=comp.get("opt_has_nan", False),
        has_inf=comp.get("opt_has_inf", False),
        end_to_end_speedup=speedup,
    )

    report = format_report(result, diagnose_results)
    print(report)

    # Save JSON if requested
    if args.json:
        json_path = os.path.abspath(args.json)
        save_verification_json(result, json_path)
        print(f"Results saved to: {json_path}")

    # Default: save to workspace
    default_json = os.path.join(WORKSPACE_DIR, "verification_result.json")
    os.makedirs(WORKSPACE_DIR, exist_ok=True)
    save_verification_json(result, default_json)
    print(f"Results saved to: {default_json}")

    # Exit code: 0 for PASS, 1 for FAIL
    if result.correctness != "PASS":
        sys.exit(1)


if __name__ == "__main__":
    main()
