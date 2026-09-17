#!/usr/bin/env python3
"""Test bfloat16 detector — same dynamic range as fp32, fewer mantissa bits.

Should handle the PP-OCRv5 server neck's large activations (~92k) without
overflow, unlike fp16 which maxes out at 65504.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT
os.chdir(PROJECT_ROOT)
for _p in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def main() -> int:
    for p in (REPO_ROOT / "autokernel", PROJECT_ROOT / "PaddleOCR2Pytorch"):
        sys.path.insert(0, str(p))
    os.environ["AUTOKERNEL_PPOCR_ROOT"] = str(PROJECT_ROOT / "PaddleOCR2Pytorch")
    os.environ["AUTOKERNEL_PPOCRV5_SERVER_DET_PTH"] = str(
        REPO_ROOT / "autokernel/workspace/ppocrv5/server_det.pth"
    )

    import cv2
    import importlib
    import torch

    ppocr_model = importlib.import_module("models.ppocrv5_server")

    img = cv2.imread(str(PROJECT_ROOT / "tests/data/good_data.png"))
    h, w = img.shape[:2]
    r = 1072.0 / max(h, w) if max(h, w) > 1072 else 1.0
    rh = int(round(h * r / 32)) * 32
    rw = int(round(w * r / 32)) * 32
    img = cv2.resize(img, (rw, rh))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = (img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    x32 = torch.from_numpy(img.transpose(2, 0, 1)[None, ...]).cuda().float()

    def run(dtype_name):
        dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[dtype_name]
        m = ppocr_model.PPOCRv5ServerDetModel().cuda().to(dtype).eval()
        with torch.inference_mode():
            maps = m(x32.to(dtype))
        maps = maps.float().cpu()
        return maps

    maps_fp32 = run("fp32")
    maps_fp16 = run("fp16")
    maps_bf16 = run("bf16")

    def stats(m, name):
        nan = int(torch.isnan(m).sum())
        frac = float((m > 0.3).float().mean())
        mx = float(m[torch.isfinite(m)].max()) if torch.isfinite(m).any() else float("nan")
        print(f"{name:6s}: nan={nan:>7d}  max={mx:8.4f}  frac>0.3={frac:.4f}")

    stats(maps_fp32, "fp32")
    stats(maps_fp16, "fp16")
    stats(maps_bf16, "bf16")

    d = (maps_fp32 - maps_bf16).abs()
    finite = d[torch.isfinite(d)]
    print(f"\nbf16 vs fp32: max_abs_diff={finite.max().item():.4e}  mean_abs_diff={finite.mean().item():.4e}")

    # How much do the box-threshold decisions agree?
    p32 = maps_fp32[:, 0]
    pbf = maps_bf16[:, 0]
    for thr in (0.3, 0.6):
        a = (p32 > thr)
        b = (pbf > thr)
        iou = (a & b).sum().float() / max((a | b).sum().item(), 1)
        print(f"  thresh>{thr}: IoU of positive pixels = {iou.item():.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
