#!/usr/bin/env python3
"""End-to-end validation: does FullPyTorchBackend with det_dtype=fp16 detect boxes?"""

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

    img = cv2.cvtColor(cv2.imread(str(PROJECT_ROOT / "tests/data/good_data.png")),
                       cv2.COLOR_BGR2RGB)

    print("--- det_dtype=float32 (reference) ---")
    b_fp32 = FullPyTorchBackend(**common, det_dtype="float32")
    r_fp32 = b_fp32.predict(img)[0]
    print(f"  boxes={len(r_fp32['rec_polys'])}  sample_texts={r_fp32['rec_texts'][:5]}")
    b_fp32.close()

    print("--- det_dtype=float16 (mixed precision) ---")
    b_fp16 = FullPyTorchBackend(**common, det_dtype="float16")
    r_fp16 = b_fp16.predict(img)[0]
    print(f"  boxes={len(r_fp16['rec_polys'])}  sample_texts={r_fp16['rec_texts'][:5]}")
    b_fp16.close()

    if len(r_fp16["rec_polys"]) == 0:
        print("FAIL: fp16 detector produced zero boxes")
        return 1
    print("OK: fp16 mixed-precision detector produced boxes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
