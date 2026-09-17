#!/usr/bin/env python3
"""Compare PP-OCRv5 server detector output from Paddle inference vs converted PyTorch.

Feeds the same synthetic input tensors through (a) the original Paddle PIR
inference model and (b) the PyTorch detector built from the converted `.pth`,
and reports the max/mean absolute and relative error between their DB
probability maps.

Prints JSON. Exits 0 on parity, 1 on mismatch, 2 on setup failure.
"""

from __future__ import annotations

import argparse
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
for _p in (PROJECT_ROOT, SRC_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


DEFAULT_PPOCR_ROOT = PROJECT_ROOT / "PaddleOCR2Pytorch"
DEFAULT_AUTOKERNEL_ROOT = REPO_ROOT / "autokernel"
DEFAULT_DET_SOURCE = (
    PROJECT_ROOT / "src/models/server_models/ppocrv5_server_det_source/inference.pdiparams"
)
DEFAULT_DET_PTH = DEFAULT_AUTOKERNEL_ROOT / "workspace/ppocrv5/server_det.pth"


def _parse_shape(shape: str) -> tuple[int, int, int]:
    parts = tuple(int(p.strip()) for p in shape.split(","))
    if len(parts) != 3:
        raise ValueError(f"Expected C,H,W, got {shape!r}")
    c, h, w = parts
    if c != 3:
        raise ValueError(f"Expected 3 channels, got {c}")
    if h % 32 or w % 32:
        raise ValueError(
            f"Det input H and W must be multiples of 32 (stride-32 backbone); got {h}x{w}"
        )
    return c, h, w


def _make_synthetic_det_input(
    index: int,
    rng: np.random.Generator,
    shape: tuple[int, int, int],
) -> np.ndarray:
    """Generate a deterministic, reasonably diverse 3xHxW tensor in [-1, 1].

    Mirrors the recognizer verifier's approach: periodic channel patterns plus
    random rectangles plus gaussian noise, normalised with ImageNet-like mean/std.
    The detector was trained with mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]
    but what matters here is that baseline and converted see the SAME input —
    any valid in-range float32 tensor works.
    """
    channels, height, width = shape
    yy, xx = np.mgrid[0:height, 0:width]
    image = np.zeros((height, width, channels), dtype=np.float32)
    image[..., 0] = (xx * (index + 3) + yy * 7) % 256
    image[..., 1] = (yy * (index + 5) + 31) % 256
    image[..., 2] = ((xx // 3 + yy // 2) * (index + 11)) % 256

    for _ in range(24):
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


def _prepare_paths(args: argparse.Namespace) -> dict[str, Path]:
    autokernel_root = Path(args.autokernel_root or DEFAULT_AUTOKERNEL_ROOT).resolve()
    ppocr_root = Path(args.ppocr_root or DEFAULT_PPOCR_ROOT).resolve()
    det_source = Path(args.det_source or DEFAULT_DET_SOURCE).resolve()
    det_pth = Path(args.det_pth or DEFAULT_DET_PTH).resolve()

    paths = {
        "autokernel_root": autokernel_root,
        "ppocr_root": ppocr_root,
        "det_source (pdiparams)": det_source,
        "det_pth": det_pth,
    }
    missing = [f"{k}: {v}" for k, v in paths.items() if not v.exists()]
    if missing:
        raise FileNotFoundError("Missing required artifacts:\n" + "\n".join(missing))
    return paths


def _build_paddle_model(source: Path) -> Any:
    """Load the Paddle PIR inference model from a directory or .pdiparams file."""
    import paddle

    prefix = source.with_suffix("") if source.suffix == ".pdiparams" else source / "inference"
    layer = paddle.jit.load(str(prefix))
    layer.eval()
    return layer


def _build_pytorch_model(autokernel_root: Path, ppocr_root: Path, det_pth: Path) -> Any:
    """Load the converted PyTorch detector via AutoKernel's wrapper."""
    for p in (autokernel_root, ppocr_root):
        p_str = str(p)
        if p_str not in sys.path:
            sys.path.insert(0, p_str)
    os.environ["AUTOKERNEL_PPOCR_ROOT"] = str(ppocr_root)
    os.environ["AUTOKERNEL_PPOCRV5_SERVER_DET_PTH"] = str(det_pth)

    import importlib

    ppocr_model = importlib.import_module("models.ppocrv5_server")
    model = ppocr_model.PPOCRv5ServerDetModel()
    return model


def verify_logits(args: argparse.Namespace) -> dict[str, Any]:
    paths = _prepare_paths(args)

    import torch

    device = "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    torch_dtype = {"float32": torch.float32, "float16": torch.float16}[args.dtype.lower()]

    paddle_model = _build_paddle_model(paths["det_source (pdiparams)"])
    torch_model = _build_pytorch_model(
        paths["autokernel_root"], paths["ppocr_root"], paths["det_pth"]
    ).to(device=device, dtype=torch_dtype).eval()

    shapes = [_parse_shape(s) for s in args.shapes.split(";")]
    rng = np.random.default_rng(args.seed)

    failures = []
    summary = {
        "max_abs_error": 0.0,
        "mean_abs_error": 0.0,
        "max_rel_error": 0.0,
        "worst_sample": -1,
        "per_shape": [],
    }
    total_samples = 0
    mean_abs_error_accumulator = 0.0

    import paddle

    for shape_idx, shape in enumerate(shapes):
        shape_summary = {
            "shape": f"3,{shape[1]},{shape[2]}",
            "max_abs_error": 0.0,
            "mean_abs_error": 0.0,
            "max_rel_error": 0.0,
            "samples": args.samples_per_shape,
        }
        shape_mae = 0.0

        for sample_idx in range(args.samples_per_shape):
            global_idx = shape_idx * args.samples_per_shape + sample_idx
            tensor = _make_synthetic_det_input(global_idx, rng, shape)

            with paddle.no_grad():
                paddle_input = paddle.to_tensor(tensor[np.newaxis, ...])
                paddle_output = paddle_model(paddle_input)
                if isinstance(paddle_output, (tuple, list)):
                    paddle_output = paddle_output[0]
                if isinstance(paddle_output, dict):
                    paddle_output = paddle_output.get("maps", next(iter(paddle_output.values())))
                paddle_np = paddle_output.numpy().astype(np.float32)

            with torch.inference_mode():
                torch_input = torch.from_numpy(tensor).unsqueeze(0).to(device=device, dtype=torch_dtype)
                torch_output = torch_model(torch_input)
                torch_np = torch_output.float().cpu().numpy()

            if paddle_np.shape != torch_np.shape:
                raise RuntimeError(
                    f"Shape mismatch sample={global_idx} shape={shape}: "
                    f"paddle={paddle_np.shape} torch={torch_np.shape}"
                )

            diff = np.abs(paddle_np - torch_np)
            sample_max_abs = float(diff.max())
            sample_mean_abs = float(diff.mean())
            denom = np.maximum(np.abs(paddle_np), np.abs(torch_np))
            denom = np.where(denom < 1e-12, 1e-12, denom)
            sample_max_rel = float((diff / denom).max())

            mean_abs_error_accumulator += sample_mean_abs
            total_samples += 1
            shape_mae += sample_mean_abs
            shape_summary["max_abs_error"] = max(shape_summary["max_abs_error"], sample_max_abs)
            shape_summary["max_rel_error"] = max(shape_summary["max_rel_error"], sample_max_rel)
            if sample_max_abs > summary["max_abs_error"]:
                summary["max_abs_error"] = sample_max_abs
                summary["worst_sample"] = global_idx
            summary["max_rel_error"] = max(summary["max_rel_error"], sample_max_rel)

            close = sample_max_abs <= args.atol + args.rtol * float(np.abs(paddle_np).max())
            if not close:
                failures.append(
                    {
                        "sample": global_idx,
                        "shape": shape_summary["shape"],
                        "max_abs_error": sample_max_abs,
                        "mean_abs_error": sample_mean_abs,
                        "max_rel_error": sample_max_rel,
                    }
                )

        shape_summary["mean_abs_error"] = shape_mae / max(args.samples_per_shape, 1)
        summary["per_shape"].append(shape_summary)

    summary["mean_abs_error"] = mean_abs_error_accumulator / max(total_samples, 1)

    return {
        "samples": total_samples,
        "shapes": [f"3,{s[1]},{s[2]}" for s in shapes],
        "device": device,
        "dtype": args.dtype,
        "seed": args.seed,
        "atol": args.atol,
        "rtol": args.rtol,
        "paddle_source": str(paths["det_source (pdiparams)"]),
        "pytorch_weights": str(paths["det_pth"]),
        **summary,
        "failures": failures,
        "passed": not failures,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shapes",
        default="3,640,640;3,960,640;3,1280,960",
        help="Semicolon-separated list of C,H,W shapes to test.",
    )
    parser.add_argument("--samples-per-shape", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260417)
    parser.add_argument(
        "--dtype",
        choices=("float32", "float16"),
        default="float32",
        help="Torch-side dtype. Paddle inference always runs fp32.",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--autokernel-root", default=None)
    parser.add_argument("--ppocr-root", default=None)
    parser.add_argument("--det-source", default=None)
    parser.add_argument("--det-pth", default=None)
    parser.add_argument("--json-output", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = verify_logits(args)
    except Exception as exc:
        print(f"Det logits verification failed: {exc}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 2

    output = json.dumps(result, indent=2, sort_keys=True)
    print(output)
    if args.json_output:
        Path(args.json_output).write_text(output + "\n", encoding="utf-8")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
