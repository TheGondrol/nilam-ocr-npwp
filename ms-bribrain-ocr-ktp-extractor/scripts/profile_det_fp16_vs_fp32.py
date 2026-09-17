#!/usr/bin/env python3
"""Profile detector latency fp32 vs fp16 mixed, and compare recognized texts
line-by-line for all three test images.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT
os.chdir(PROJECT_ROOT)
for _p in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


IMAGES = [
    PROJECT_ROOT / "tests/data/good_data.png",
    PROJECT_ROOT / "tests/data/good_data_2.png",
    PROJECT_ROOT / "tests/data/good_data_3.png",
]

WARMUP_ITERS = 3
MEASURE_ITERS = 20


def _time_det(backend: Any, image: np.ndarray) -> float:
    """Return median detector latency in ms over MEASURE_ITERS."""
    import torch

    for _ in range(WARMUP_ITERS):
        backend._detect_boxes(image)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    samples = []
    for _ in range(MEASURE_ITERS):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        backend._detect_boxes(image)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        samples.append((time.perf_counter() - t0) * 1000.0)
    samples.sort()
    return samples[len(samples) // 2]


def main() -> int:
    import cv2
    from src.services.ocr_backends import FullPyTorchBackend

    ak = REPO_ROOT / "autokernel"
    common = dict(
        autokernel_root=ak,
        ppocr_root=PROJECT_ROOT / "PaddleOCR2Pytorch",
        det_weights_path=ak / "workspace/ppocrv5/server_det.pth",
        rec_weights_path=ak / "workspace/ppocrv5/server_rec.pth",
        det_source_path=None,
        rec_source_path=None,
        use_gpu=True,
        auto_convert_weights=False,
        rec_batch_size=8,
        dtype="float16",
    )

    print("Loading fp32 detector backend ...")
    b32 = FullPyTorchBackend(**common, det_dtype="float32")
    print("Loading fp16-mixed detector backend ...")
    b16 = FullPyTorchBackend(**common, det_dtype="float16")

    images = []
    for p in IMAGES:
        img = cv2.imread(str(p))
        if img is None:
            raise FileNotFoundError(p)
        images.append((p.name, cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))

    # --- Latency profile ---
    print("\n" + "=" * 72)
    print(f"{'image':22s} {'shape':>14s} {'fp32 det ms':>12s} {'fp16 det ms':>12s} {'speedup':>10s}")
    print("-" * 72)
    for name, img in images:
        t32 = _time_det(b32, img)
        t16 = _time_det(b16, img)
        speedup = t32 / t16 if t16 > 0 else 0.0
        print(f"{name:22s} {str(img.shape):>14s} {t32:12.2f} {t16:12.2f} {speedup:9.2f}x")

    # --- Line-by-line text comparison ---
    print("\n" + "=" * 72)
    print("Line-by-line text diff (fp32 reference vs fp16 mixed)")
    print("=" * 72)

    total_diff = 0
    total_lines = 0
    for name, img in images:
        r32 = b32.predict(img)[0]
        r16 = b16.predict(img)[0]
        t32 = r32["rec_texts"]
        t16 = r16["rec_texts"]
        n = max(len(t32), len(t16))
        same = sum(1 for i in range(n) if i < len(t32) and i < len(t16) and t32[i] == t16[i])
        diff = n - same
        total_diff += diff
        total_lines += n
        print(f"\n--- {name}  (fp32 boxes={len(t32)}, fp16 boxes={len(t16)}, "
              f"matches={same}/{n}) ---")
        print(f"  {'idx':>3s}  {'match':5s}  fp32  |  fp16")
        for i in range(n):
            a = t32[i] if i < len(t32) else "<missing>"
            b = t16[i] if i < len(t16) else "<missing>"
            mark = "==" if a == b else "!="
            print(f"  {i:3d}  {mark:5s}  {a!r}  |  {b!r}")

    print(f"\nOverall: {total_lines - total_diff}/{total_lines} lines identical "
          f"({(total_lines - total_diff) / max(total_lines, 1) * 100:.1f}%)")

    b32.close()
    b16.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
