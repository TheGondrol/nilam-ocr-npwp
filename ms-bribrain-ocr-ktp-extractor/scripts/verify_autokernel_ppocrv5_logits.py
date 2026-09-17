#!/usr/bin/env python3
"""Compare PP-OCRv5 recognizer logits before and after AutoKernel replacement."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT
SRC_ROOT = PROJECT_ROOT / "src"
os.chdir(PROJECT_ROOT)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from src.core.config import settings  # noqa: E402
from src.services.ocr_backends import load_verified_replacement_specs  # noqa: E402


def _resolve_path(path_value: str | Path, base_dir: Path) -> Path:
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _import_module_from_path(module_name: str, module_path: Path) -> Any:
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {module_name} from {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


def _parse_shape(shape: str) -> tuple[int, int, int]:
    dims = tuple(int(part.strip()) for part in shape.split(","))
    if len(dims) != 3:
        raise ValueError(f"Expected recognizer image shape C,H,W, got {shape!r}")
    channels, height, width = dims
    if channels != 3:
        raise ValueError(f"Expected 3-channel recognizer input, got {channels}")
    return channels, height, width


def _make_synthetic_recognizer_input(
    index: int,
    rng: np.random.Generator,
    image_shape: str,
) -> np.ndarray:
    channels, height, width = _parse_shape(image_shape)
    yy, xx = np.mgrid[0:height, 0:width]
    image = np.zeros((height, width, channels), dtype=np.float32)

    image[..., 0] = (xx * (index + 3) + yy * 7) % 256
    image[..., 1] = (yy * (index + 5) + 31) % 256
    image[..., 2] = ((xx // 3 + yy // 2) * (index + 11)) % 256

    for _ in range(12):
        x1 = int(rng.integers(0, max(width - 2, 1)))
        y1 = int(rng.integers(0, max(height - 2, 1)))
        x2 = int(rng.integers(x1 + 1, width + 1))
        y2 = int(rng.integers(y1 + 1, height + 1))
        color = rng.integers(0, 256, size=(channels,), dtype=np.uint8).astype(np.float32)
        image[y1:y2, x1:x2, :] = color

    noise = rng.normal(loc=0.0, scale=18.0, size=image.shape).astype(np.float32)
    image = np.clip(image + noise, 0.0, 255.0)
    image = (image / 255.0 - 0.5) / 0.5
    return image.transpose(2, 0, 1).copy()


def _prepare_imports(autokernel_root: Path, ppocr_root: Path) -> None:
    for path in (autokernel_root, ppocr_root):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

    os.environ["AUTOKERNEL_PPOCR_ROOT"] = str(ppocr_root)


def _make_replacements(autokernel_root: Path, workspace_path: Path) -> list[Any]:
    specs = load_verified_replacement_specs(workspace_path, autokernel_root)
    if not specs:
        raise RuntimeError(f"No verified AutoKernel replacements found in {workspace_path}")

    verify_mod = _import_module_from_path(
        "dgc_ext_autokernel_verify_logits",
        autokernel_root / "verify.py",
    )
    support_mod = _import_module_from_path(
        "dgc_ext_autokernel_support_logits",
        autokernel_root / "support.py",
    )

    replacements = []
    for spec in specs:
        support_stage = support_mod.build_support_stage(spec.kernel_type, {spec.kernel_type})
        replacements.append(
            verify_mod.KernelReplacement(
                kernel_type=spec.kernel_type,
                rank=spec.rank,
                speedup=spec.speedup,
                optimized_path=str(spec.optimized_path),
                reinsert_supported=support_stage["reinsert_supported"],
                status=spec.status,
            )
        )
    return replacements


def _required_paths(args: argparse.Namespace) -> dict[str, Path]:
    base_dir = settings.config_path.resolve().parent
    autokernel_root = _resolve_path(args.autokernel_root or settings.ocr_autokernel_root, base_dir)
    ppocr_root = _resolve_path(args.ppocr_root or settings.ocr_autokernel_ppocr_root, base_dir)
    det_weights = _resolve_path(
        args.det_weights or settings.ocr_autokernel_det_weights_path,
        base_dir,
    )
    rec_weights = _resolve_path(
        args.rec_weights or settings.ocr_autokernel_rec_weights_path,
        base_dir,
    )
    workspace = _resolve_path(args.workspace or settings.ocr_autokernel_workspace_path, base_dir)
    return {
        "AutoKernel root": autokernel_root,
        "PaddleOCR2Pytorch root": ppocr_root,
        "detector weights": det_weights,
        "recognizer weights": rec_weights,
        "AutoKernel workspace": workspace,
    }


def verify_logits(args: argparse.Namespace) -> dict[str, Any]:
    paths = _required_paths(args)
    missing = [f"{label}: {path}" for label, path in paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required artifacts:\n" + "\n".join(missing))

    import torch

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    if device != "cuda":
        raise RuntimeError("AutoKernel logits parity requires CUDA for optimized kernels")

    dtype = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }[args.dtype.lower()]

    autokernel_root = paths["AutoKernel root"]
    ppocr_root = paths["PaddleOCR2Pytorch root"]
    rec_weights = paths["recognizer weights"]
    workspace = paths["AutoKernel workspace"]

    _prepare_imports(autokernel_root, ppocr_root)
    os.environ["AUTOKERNEL_PPOCRV5_SERVER_REC_PTH"] = str(rec_weights)

    from models.ppocrv5_server import PPOCRv5ServerRecModel

    baseline = PPOCRv5ServerRecModel().to(device=device, dtype=dtype).eval()
    optimized = PPOCRv5ServerRecModel().to(device=device, dtype=dtype).eval()

    replacements = _make_replacements(autokernel_root, workspace)
    verify_mod = _import_module_from_path(
        "dgc_ext_autokernel_verify_logits",
        autokernel_root / "verify.py",
    )
    context = verify_mod.OptimizedModelContext(optimized, replacements)

    failures = []
    max_abs_error = 0.0
    mean_abs_error_sum = 0.0
    max_rel_error = 0.0
    worst_sample = -1
    rng = np.random.default_rng(args.seed)

    try:
        optimized_model = context.__enter__()
        with torch.inference_mode():
            for sample_idx in range(args.samples):
                synthetic = _make_synthetic_recognizer_input(
                    sample_idx,
                    rng,
                    args.image_shape,
                )
                sample = torch.from_numpy(synthetic).unsqueeze(0).to(
                    device=device,
                    dtype=dtype,
                )

                reference_logits = baseline(sample.clone())
                optimized_input = context.prepare_input(sample.clone())
                optimized_logits = optimized_model(optimized_input)
                torch.cuda.synchronize()

                diff = (reference_logits.float() - optimized_logits.float()).abs()
                sample_max_abs = float(diff.max().item())
                sample_mean_abs = float(diff.mean().item())
                denom = torch.maximum(
                    reference_logits.float().abs(),
                    optimized_logits.float().abs(),
                ).clamp_min(1e-12)
                sample_max_rel = float((diff / denom).max().item())
                is_close = torch.allclose(
                    reference_logits.float(),
                    optimized_logits.float(),
                    atol=args.atol,
                    rtol=args.rtol,
                )

                mean_abs_error_sum += sample_mean_abs
                if sample_max_abs > max_abs_error:
                    max_abs_error = sample_max_abs
                    worst_sample = sample_idx
                max_rel_error = max(max_rel_error, sample_max_rel)
                if not is_close:
                    failures.append(
                        {
                            "sample": sample_idx,
                            "max_abs_error": sample_max_abs,
                            "mean_abs_error": sample_mean_abs,
                            "max_rel_error": sample_max_rel,
                        }
                    )
    finally:
        context.__exit__(None, None, None)

    return {
        "samples": args.samples,
        "input_shape": f"1,{args.image_shape}",
        "dtype": args.dtype,
        "device": device,
        "seed": args.seed,
        "atol": args.atol,
        "rtol": args.rtol,
        "replacements_total": len(replacements),
        "replacements_applied": len(getattr(context, "_applied_replacements", [])),
        "replacements_skipped": len(getattr(context, "_skipped_replacements", [])),
        "max_abs_error": max_abs_error,
        "mean_abs_error": mean_abs_error_sum / max(args.samples, 1),
        "max_rel_error": max_rel_error,
        "worst_sample": worst_sample,
        "failures": failures,
        "passed": not failures,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260410)
    parser.add_argument("--image-shape", default=settings.ocr_autokernel_rec_image_shape)
    parser.add_argument("--dtype", default=settings.ocr_autokernel_dtype)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--atol", type=float, default=5e-3)
    parser.add_argument("--rtol", type=float, default=5e-3)
    parser.add_argument("--autokernel-root")
    parser.add_argument("--ppocr-root")
    parser.add_argument("--det-weights")
    parser.add_argument("--rec-weights")
    parser.add_argument("--workspace")
    parser.add_argument("--json-output")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = verify_logits(args)
    except Exception as exc:
        print(f"AutoKernel PP-OCRv5 logits verification failed: {exc}", file=sys.stderr)
        return 2

    output = json.dumps(result, indent=2, sort_keys=True)
    print(output)
    if args.json_output:
        Path(args.json_output).write_text(output + "\n", encoding="utf-8")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
