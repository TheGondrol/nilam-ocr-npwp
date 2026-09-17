#!/usr/bin/env python3
"""Diagnose why PP-OCRv5 detector output degrades catastrophically in fp16.

Loads the converted PyTorch detector twice (fp32 + fp16) on CUDA, feeds the
same preprocessed test image, and reports:

  * DB probability map stats (min/mean/max, fraction>thresh, NaN/Inf counts)
  * Per-submodule divergence (max |fp32 - fp16|) via forward hooks
  * The first submodule whose output contains NaN/Inf in fp16

Exits 0 when fp16 matches fp32 closely, 1 when catastrophe is detected.
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
DEFAULT_DET_PTH = DEFAULT_AUTOKERNEL_ROOT / "workspace/ppocrv5/server_det.pth"
DEFAULT_IMAGE = PROJECT_ROOT / "tests/data/good_data.png"


def _build_pytorch_model(autokernel_root: Path, ppocr_root: Path, det_pth: Path) -> Any:
    for p in (autokernel_root, ppocr_root):
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)
    os.environ["AUTOKERNEL_PPOCR_ROOT"] = str(ppocr_root)
    os.environ["AUTOKERNEL_PPOCRV5_SERVER_DET_PTH"] = str(det_pth)
    import importlib
    ppocr_model = importlib.import_module("models.ppocrv5_server")
    return ppocr_model.PPOCRv5ServerDetModel()


def _preprocess(image_path: Path, limit_side_len: int, limit_type: str) -> np.ndarray:
    """Same preprocessing as TextDetector: resize to multiple of 32, normalize."""
    import cv2
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(image_path)
    h, w = img.shape[:2]
    if limit_type == "max":
        ratio = float(limit_side_len) / max(h, w) if max(h, w) > limit_side_len else 1.0
    else:
        ratio = float(limit_side_len) / min(h, w) if min(h, w) < limit_side_len else 1.0
    resize_h = max(int(round(h * ratio / 32)) * 32, 32)
    resize_w = max(int(round(w * ratio / 32)) * 32, 32)
    resized = cv2.resize(img, (resize_w, resize_h))
    resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    resized = (resized - mean) / std
    return resized.transpose(2, 0, 1)[np.newaxis, ...]  # 1,3,H,W


def _tensor_stats(t: Any) -> dict:
    import torch
    x = t.detach().float()
    finite = torch.isfinite(x)
    nan = torch.isnan(x).sum().item()
    inf = torch.isinf(x).sum().item()
    xf = x[finite]
    return {
        "shape": list(t.shape),
        "dtype": str(t.dtype),
        "nan": int(nan),
        "inf": int(inf),
        "min": float(xf.min().item()) if xf.numel() else None,
        "max": float(xf.max().item()) if xf.numel() else None,
        "mean": float(xf.mean().item()) if xf.numel() else None,
        "abs_max": float(xf.abs().max().item()) if xf.numel() else None,
    }


def _hook_all(module: Any, store: dict) -> list:
    import torch
    handles = []
    def mk(name):
        def hook(mod, inp, out):
            if isinstance(out, torch.Tensor):
                store[name] = out.detach().float().cpu()
            elif isinstance(out, dict):
                for k, v in out.items():
                    if isinstance(v, torch.Tensor):
                        store[f"{name}[{k}]"] = v.detach().float().cpu()
        return hook
    for name, m in module.named_modules():
        if name == "":
            continue
        handles.append(m.register_forward_hook(mk(name)))
    return handles


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--image", default=str(DEFAULT_IMAGE))
    p.add_argument("--limit-side-len", type=int, default=1072)
    p.add_argument("--limit-type", choices=("max", "min"), default="max")
    p.add_argument("--thresh", type=float, default=0.3)
    p.add_argument("--box-thresh", type=float, default=0.6)
    p.add_argument("--top-k-layers", type=int, default=15)
    p.add_argument("--autokernel-root", default=str(DEFAULT_AUTOKERNEL_ROOT))
    p.add_argument("--ppocr-root", default=str(DEFAULT_PPOCR_ROOT))
    p.add_argument("--det-pth", default=str(DEFAULT_DET_PTH))
    args = p.parse_args()

    import torch
    assert torch.cuda.is_available(), "CUDA required"

    ak_root = Path(args.autokernel_root).resolve()
    ppocr_root = Path(args.ppocr_root).resolve()
    det_pth = Path(args.det_pth).resolve()

    np_input = _preprocess(Path(args.image), args.limit_side_len, args.limit_type)
    x_fp32 = torch.from_numpy(np_input).cuda().float()
    x_fp16 = x_fp32.half()

    print(f"input shape: {list(x_fp32.shape)}  range: [{x_fp32.min().item():.3f}, {x_fp32.max().item():.3f}]")

    m_fp32 = _build_pytorch_model(ak_root, ppocr_root, det_pth).cuda().float().eval()
    m_fp16 = _build_pytorch_model(ak_root, ppocr_root, det_pth).cuda().half().eval()

    store32, store16 = {}, {}
    h32 = _hook_all(m_fp32, store32)
    h16 = _hook_all(m_fp16, store16)

    with torch.inference_mode():
        out32 = m_fp32(x_fp32)
        out16 = m_fp16(x_fp16)

    for h in h32 + h16:
        h.remove()

    def pick_maps(o):
        if isinstance(o, dict) and "maps" in o:
            return o["maps"]
        return o

    maps32 = pick_maps(out32).detach().float().cpu()
    maps16 = pick_maps(out16).detach().float().cpu()

    print("\n=== Final DB 'maps' output ===")
    print(f"fp32 stats: {_tensor_stats(maps32)}")
    print(f"fp16 stats: {_tensor_stats(maps16)}")

    # DB: channel 0 is probability map after sigmoid
    prob32 = maps32[:, 0]
    prob16 = maps16[:, 0]
    def frac_above(t, thr):
        return float((t > thr).float().mean().item())
    print(f"fp32 fraction prob>thresh({args.thresh}) = {frac_above(prob32, args.thresh):.4f} "
          f"| >box_thresh({args.box_thresh}) = {frac_above(prob32, args.box_thresh):.4f}")
    print(f"fp16 fraction prob>thresh({args.thresh}) = {frac_above(prob16, args.thresh):.4f} "
          f"| >box_thresh({args.box_thresh}) = {frac_above(prob16, args.box_thresh):.4f}")

    diff = (prob32 - prob16).abs()
    print(f"prob-map abs diff: max={diff.max().item():.4e}  mean={diff.mean().item():.4e}")

    # Per-layer divergence
    common = sorted(set(store32.keys()) & set(store16.keys()))
    rows = []
    for name in common:
        a, b = store32[name], store16[name]
        if a.shape != b.shape:
            continue
        nan16 = int(torch.isnan(b).sum().item())
        inf16 = int(torch.isinf(b).sum().item())
        d = (a - b).abs()
        d_max = float(d.max().item()) if d.numel() else 0.0
        abs_max32 = float(a.abs().max().item()) if a.numel() else 0.0
        rel = d_max / abs_max32 if abs_max32 > 0 else 0.0
        rows.append((name, d_max, rel, nan16, inf16, abs_max32))

    # Find first submodule where fp16 gains NaN/Inf
    first_bad = None
    for row in rows:
        if row[3] > 0 or row[4] > 0:
            first_bad = row
            break
    if first_bad:
        print(f"\nFIRST NaN/Inf in fp16: layer={first_bad[0]}  nan={first_bad[3]} inf={first_bad[4]} "
              f"fp32_abs_max={first_bad[5]:.4e}")
    else:
        print("\nNo NaN/Inf detected in fp16 intermediate outputs.")

    rows.sort(key=lambda r: r[1], reverse=True)
    print(f"\nTop-{args.top_k_layers} layers by max |fp32-fp16|:")
    print(f"{'layer':60s} {'abs_diff':>10s} {'rel':>8s} {'fp32_max':>10s}")
    for row in rows[:args.top_k_layers]:
        print(f"{row[0][:60]:60s} {row[1]:10.4e} {row[2]:8.2%} {row[5]:10.4e}")

    return 0 if diff.max().item() < 0.05 and not first_bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
