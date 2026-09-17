#!/usr/bin/env python3
"""Break down det stage into preprocess/H2D/forward/D2H/postprocess timings."""
from __future__ import annotations

import argparse
import gc
import statistics
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
_PPOCR = PROJECT_ROOT / "PaddleOCR2Pytorch"
if str(_PPOCR) not in sys.path:
    sys.path.insert(0, str(_PPOCR))

import torch  # noqa: E402

from scripts.compare_ocr_lines import _build_fullpytorch_backend  # noqa: E402
from pytorchocr.data import transform  # noqa: E402


def _stats(xs: list[float]) -> str:
    if not xs:
        return "n/a"
    ordered = sorted(xs)
    p95 = ordered[max(0, int(round(0.95 * (len(ordered) - 1))))]
    return (
        f"min {min(xs):6.2f}  med {statistics.median(xs):6.2f}  "
        f"mean {statistics.fmean(xs):6.2f}  p95 {p95:6.2f}"
    )


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def profile_det(backend, image_rgb, iterations, warmup):
    det = backend.system.text_detector

    pre_ms, h2d_ms, fwd_ms, d2h_ms, post_ms, total_ms = [], [], [], [], [], []
    for it in range(iterations):
        # preprocess
        _sync()
        t0 = time.perf_counter()
        data = {"image": image_rgb.copy()}
        data = transform(data, det.preprocess_op)
        img_np, shape_list = data
        img_np = np.expand_dims(img_np, axis=0).copy()
        shape_list = np.expand_dims(shape_list, axis=0)
        pre_t = time.perf_counter()

        # H2D
        inp = torch.from_numpy(img_np)
        if det.use_gpu:
            inp = inp.cuda()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        h2d_t = time.perf_counter()

        # forward
        with torch.no_grad():
            outputs = det.net(inp)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        fwd_t = time.perf_counter()

        # D2H
        maps_cpu = outputs["maps"].cpu().numpy()
        d2h_t = time.perf_counter()

        # postprocess
        preds = {"maps": maps_cpu}
        post_result = det.postprocess_op(preds, shape_list)
        dt_boxes = post_result[0]["points"]
        dt_boxes = det.filter_tag_det_res(dt_boxes, image_rgb.shape)
        post_t = time.perf_counter()

        if it >= warmup:
            pre_ms.append((pre_t - t0) * 1000)
            h2d_ms.append((h2d_t - pre_t) * 1000)
            fwd_ms.append((fwd_t - h2d_t) * 1000)
            d2h_ms.append((d2h_t - fwd_t) * 1000)
            post_ms.append((post_t - d2h_t) * 1000)
            total_ms.append((post_t - t0) * 1000)

    print(f"  input shape after pre: {inp.shape}  output maps shape: {outputs['maps'].shape}")
    print(f"  num boxes: {len(dt_boxes)}")
    print(f"  preprocess (CPU):   {_stats(pre_ms)}")
    print(f"  H2D:                {_stats(h2d_ms)}")
    print(f"  forward (GPU):      {_stats(fwd_ms)}")
    print(f"  D2H (maps->numpy):  {_stats(d2h_ms)}")
    print(f"  postprocess (CPU):  {_stats(post_ms)}")
    print(f"  total:              {_stats(total_ms)}")
    med = statistics.median(total_ms)
    print("  share of total (median):")
    for name, xs in (
        ("pre", pre_ms),
        ("h2d", h2d_ms),
        ("fwd", fwd_ms),
        ("d2h", d2h_ms),
        ("post", post_ms),
    ):
        print(f"    {name:<4} {statistics.median(xs):6.2f} ms  ({100 * statistics.median(xs) / med:5.1f}%)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=Path, default=PROJECT_ROOT / "tests/data/good_data_3.png")
    ap.add_argument("--iterations", type=int, default=12)
    ap.add_argument("--warmup", type=int, default=4)
    args = ap.parse_args()

    image_rgb = np.asarray(Image.open(args.image).convert("RGB"))
    print(f"Image: {args.image}  shape: {image_rgb.shape}")

    backend = _build_fullpytorch_backend()
    print("Backend built.")
    profile_det(backend, image_rgb, args.iterations, args.warmup)

    closer = getattr(backend, "close", None)
    if callable(closer):
        closer()
    del backend
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
