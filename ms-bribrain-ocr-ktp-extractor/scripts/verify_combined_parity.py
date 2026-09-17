#!/usr/bin/env python3
"""Verify text output parity: baseline fullpytorch vs (autocast-islands + compile-rec)."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.chdir(PROJECT_ROOT)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.compare_ocr_lines import _build_fullpytorch_backend  # noqa: E402

IMAGES = [
    PROJECT_ROOT / "tests/data/good_data.png",
    PROJECT_ROOT / "tests/data/good_data_2.png",
    PROJECT_ROOT / "tests/data/good_data_3.png",
]


def _apply_det_islands(backend):
    import torch
    import torch.nn as nn

    class Fp32Island(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, x, *a, **kw):
            def to32(v):
                if isinstance(v, torch.Tensor):
                    return v.float()
                if isinstance(v, list):
                    return [to32(t) for t in v]
                if isinstance(v, tuple):
                    return tuple(to32(t) for t in v)
                return v
            with torch.autocast(device_type="cuda", enabled=False):
                return self.inner(to32(x), *to32(a), **kw)

    det = backend.system.text_detector.net
    for leaf in ("neck", "head"):
        setattr(det, leaf, Fp32Island(getattr(det, leaf)))
    orig = det.forward

    def fwd(x):
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            return orig(x)

    det.forward = fwd


def _compile_rec(backend):
    import torch
    backend.system.text_recognizer.net = torch.compile(
        backend.system.text_recognizer.net, mode="reduce-overhead"
    )


def main():
    import torch

    ref = _build_fullpytorch_backend()
    opt = _build_fullpytorch_backend()
    _apply_det_islands(opt)
    _compile_rec(opt)

    def run(backend, img):
        # Use backend.predict to get the final lines
        return backend.predict(img)

    for img_path in IMAGES:
        img = np.asarray(Image.open(img_path).convert("RGB"))
        # warmup
        for _ in range(2):
            run(ref, img)
            run(opt, img)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        ref_out = run(ref, img)
        opt_out = run(opt, img)

        def flatten(out):
            # predict() returns [dict(rec_texts=[...], rec_scores=[...])]
            d = out[0]
            texts = d.get("rec_texts", [])
            scores = d.get("rec_scores", [])
            return [(t, round(float(s), 4)) for t, s in zip(texts, scores)]
        ref_lines = flatten(ref_out)
        opt_lines = flatten(opt_out)

        print(f"\n{img_path.name}  ref={len(ref_lines)}  opt={len(opt_lines)}")
        n = max(len(ref_lines), len(opt_lines))
        diffs = 0
        for i in range(n):
            r = ref_lines[i] if i < len(ref_lines) else ("<missing>", 0.0)
            o = opt_lines[i] if i < len(opt_lines) else ("<missing>", 0.0)
            marker = "  " if r == o else "≠ "
            if r != o:
                diffs += 1
            print(f"  {marker}{i:3d}  ref={r}  opt={o}")
        print(f"  → {n - diffs}/{n} lines match")


if __name__ == "__main__":
    main()
