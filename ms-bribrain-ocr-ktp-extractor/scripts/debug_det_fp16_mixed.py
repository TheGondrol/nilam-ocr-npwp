#!/usr/bin/env python3
"""Verify that a mixed-precision detector (fp16 backbone+neck, fp32 head) fixes
the fp16 catastrophe diagnosed by debug_det_fp16.py.

Compares:
  (a) full fp32 — reference
  (b) full fp16 — known catastrophic
  (c) mixed: backbone+neck fp16, head fp32
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

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


def _build(ak_root: Path, ppocr_root: Path, det_pth: Path):
    for p in (ak_root, ppocr_root):
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)
    os.environ["AUTOKERNEL_PPOCR_ROOT"] = str(ppocr_root)
    os.environ["AUTOKERNEL_PPOCRV5_SERVER_DET_PTH"] = str(det_pth)
    import importlib
    return importlib.import_module("models.ppocrv5_server").PPOCRv5ServerDetModel()


def _preprocess(image_path: Path, limit_side_len: int = 1072) -> np.ndarray:
    import cv2
    img = cv2.imread(str(image_path))
    h, w = img.shape[:2]
    ratio = float(limit_side_len) / max(h, w) if max(h, w) > limit_side_len else 1.0
    rh = max(int(round(h * ratio / 32)) * 32, 32)
    rw = max(int(round(w * ratio / 32)) * 32, 32)
    r = cv2.resize(img, (rw, rh))
    r = cv2.cvtColor(r, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    r = (r - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
    return r.transpose(2, 0, 1)[None, ...]


def _stats(t, thr: float):
    import torch
    x = t.float().cpu()
    nan = int(torch.isnan(x).sum().item())
    finite = x[torch.isfinite(x)]
    return {
        "nan": nan,
        "min": float(finite.min()) if finite.numel() else None,
        "max": float(finite.max()) if finite.numel() else None,
        "mean": float(finite.mean()) if finite.numel() else None,
        "frac_above_thr": float((x > thr).float().mean()),
    }


def main() -> int:
    import torch

    ak = DEFAULT_AUTOKERNEL_ROOT.resolve()
    pp = DEFAULT_PPOCR_ROOT.resolve()
    pth = DEFAULT_DET_PTH.resolve()
    np_in = _preprocess(DEFAULT_IMAGE)
    x_fp32 = torch.from_numpy(np_in).cuda().float()
    x_fp16 = x_fp32.half()

    # (a) fp32
    m32 = _build(ak, pp, pth).cuda().float().eval()
    with torch.inference_mode():
        maps32 = m32(x_fp32)

    # (b) fp16
    m16 = _build(ak, pp, pth).cuda().half().eval()
    with torch.inference_mode():
        maps16 = m16(x_fp16)

    # (c) mixed: backbone fp16, neck + head fp32
    m_mix = _build(ak, pp, pth).cuda().eval()
    m_mix.net.backbone.half()
    m_mix.net.neck.float()
    m_mix.net.head.float()
    with torch.inference_mode():
        xb = m_mix.net.backbone(x_fp16)
        if isinstance(xb, (list, tuple)):
            xb = type(xb)(t.float() if isinstance(t, torch.Tensor) else t for t in xb)
        elif isinstance(xb, torch.Tensor):
            xb = xb.float()
        xn = m_mix.net.neck(xb)
        maps_mix = m_mix.net.head(xn)
        if isinstance(maps_mix, dict):
            maps_mix = maps_mix["maps"]

    print("=== fp32 (reference) ===")
    print(_stats(maps32, 0.3))
    print("=== full fp16 ===")
    print(_stats(maps16, 0.3))
    print("=== mixed (backbone+neck fp16, head fp32) ===")
    print(_stats(maps_mix, 0.3))

    diff = (maps32.float().cpu() - maps_mix.float().cpu()).abs()
    print(f"\nmixed-vs-fp32 prob-map diff: max={diff.max().item():.4e} mean={diff.mean().item():.4e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
