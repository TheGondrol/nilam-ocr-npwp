#!/usr/bin/env python3
"""Per-stage latency profiler for HybridOCRBackend.predict().

Breaks a predict() call into:
    det        → PaddleOCR detection pass
    crop       → cv2 crop extraction (CPU)
    rec_pre    → upload + resize + normalize (GPU, in _fast_recognize loop)
    rec_fwd    → recognizer forward pass (GPU)
    rec_post   → argmax + D2H + CTC decode
    total      → wall-clock end-to-end

Uses ``torch.cuda.Event`` for GPU-accurate timing of GPU stages.

Usage:
    .venv/bin/python -u scripts/profile_hybrid_stages.py
    .venv/bin/python -u scripts/profile_hybrid_stages.py --iterations 30 --warmup 3
    .venv/bin/python -u scripts/profile_hybrid_stages.py --image tests/data/good_data_2.png
"""

from __future__ import annotations

import argparse
import gc
import logging
import math
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
for name in ("ppocr", "paddle", "paddleocr", "torch", "triton"):
    logging.getLogger(name).setLevel(logging.ERROR)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from scripts.compare_ocr_lines import _build_hybrid_backend  # noqa: E402
import scripts.compare_ocr_lines as _cmp  # noqa: E402


STAGE_KEYS = ["det", "crop", "rec_pre", "rec_fwd", "rec_post", "total"]


@dataclass
class StageRun:
    timings_ms: dict[str, list[float]] = field(
        default_factory=lambda: {k: [] for k in STAGE_KEYS}
    )
    n_crops: list[int] = field(default_factory=list)
    n_batches: list[int] = field(default_factory=list)

    def record(self, stage: str, ms: float) -> None:
        self.timings_ms[stage].append(ms)


def _patched_fast_recognize(backend: Any, run: StageRun, iter_idx: int):
    """Return an instrumented ``_fast_recognize`` bound to ``backend``.

    Breaks the per-batch loop into three stages using CUDA events:
    ``rec_pre`` (upload + resize + norm), ``rec_fwd`` (model forward),
    ``rec_post`` (argmax + D2H + CTC decode).
    """

    def _fast_recognize(crops: list[np.ndarray]) -> list[tuple[str, float]]:
        imgC, imgH = backend._rec_imgC, backend._rec_imgH
        device = "cuda"
        model_dtype = backend._torch_dtype(torch)
        limited_max = getattr(backend.recognizer, "limited_max_width", 4000)
        limited_min = getattr(backend.recognizer, "limited_min_width", 16)

        width_list = [c.shape[1] / float(c.shape[0]) for c in crops]
        indices = np.argsort(np.asarray(width_list))
        batch_num = max(int(backend.rec_batch_size), 1)
        rec_res: list[tuple[str, float]] = [("", 0.0)] * len(crops)

        pre_ms_total = 0.0
        fwd_ms_total = 0.0
        post_ms_total = 0.0
        n_batches = 0

        for beg in range(0, len(crops), batch_num):
            end = min(len(crops), beg + batch_num)
            batch_indices = indices[beg:end]
            n_batches += 1

            pre_start = torch.cuda.Event(enable_timing=True)
            pre_end = torch.cuda.Event(enable_timing=True)
            fwd_end = torch.cuda.Event(enable_timing=True)
            post_end = torch.cuda.Event(enable_timing=True)

            pre_start.record()

            max_wh_ratio = max(width_list[int(i)] for i in batch_indices)
            max_wh_ratio = max(max_wh_ratio, backend._rec_imgW / imgH)
            target_w = max(min(int(imgH * max_wh_ratio), limited_max), limited_min)

            batch = torch.zeros(
                len(batch_indices), imgC, imgH, target_w,
                dtype=model_dtype, device=device,
            )
            for batch_pos, crop_index in enumerate(batch_indices):
                crop = crops[int(crop_index)]
                h, w = crop.shape[:2]
                resized_w = max(
                    min(int(math.ceil(imgH * w / float(h))), target_w), limited_min
                )
                t = torch.from_numpy(crop.copy()).to(device=device, dtype=torch.float32)
                t = t.permute(2, 0, 1).unsqueeze(0)
                t = F.interpolate(
                    t, size=(imgH, resized_w), mode="bilinear", align_corners=False
                )
                t = t.squeeze(0).to(dtype=model_dtype)
                t = (t / 255.0 - 0.5) / 0.5
                batch[batch_pos, :, :, :resized_w] = t

            pre_end.record()

            with torch.no_grad():
                preds = backend.recognizer.net(batch)

            fwd_end.record()

            preds_f = preds.float()
            preds_idx = preds_f.argmax(dim=2).cpu().numpy()
            preds_prob = preds_f.max(dim=2).values.cpu().numpy()

            batch_results = backend.recognizer.postprocess_op.decode(
                preds_idx, preds_prob, is_remove_duplicate=True
            )
            for batch_pos, result in enumerate(batch_results):
                rec_res[int(batch_indices[batch_pos])] = result

            post_end.record()
            torch.cuda.synchronize()

            pre_ms_total += pre_start.elapsed_time(pre_end)
            fwd_ms_total += pre_end.elapsed_time(fwd_end)
            post_ms_total += fwd_end.elapsed_time(post_end)

        run.record("rec_pre", pre_ms_total)
        run.record("rec_fwd", fwd_ms_total)
        run.record("rec_post", post_ms_total)
        if iter_idx == 0:
            run.n_batches.append(n_batches)
        return rec_res

    return _fast_recognize


def _profile_one(backend: Any, img_rgb: np.ndarray, run: StageRun, iter_idx: int) -> int:
    """Run one instrumented predict(), append per-stage timings to run."""
    from src.services.ocr_backends import _crop_text_region

    orig_fast = backend._fast_recognize
    backend._fast_recognize = _patched_fast_recognize(backend, run, iter_idx)
    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        wall_start = time.perf_counter()

        t0 = time.perf_counter()
        boxes = backend._detect_boxes(img_rgb)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        run.record("det", (time.perf_counter() - t0) * 1000)

        if not boxes:
            run.record("crop", 0.0)
            run.record("rec_pre", 0.0); run.record("rec_fwd", 0.0); run.record("rec_post", 0.0)
            run.record("total", (time.perf_counter() - wall_start) * 1000)
            return 0

        t0 = time.perf_counter()
        image_rgb = np.ascontiguousarray(img_rgb)
        crops = [_crop_text_region(image_rgb, box) for box in boxes]
        run.record("crop", (time.perf_counter() - t0) * 1000)

        backend._fast_recognize(crops)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        run.record("total", (time.perf_counter() - wall_start) * 1000)
        return len(crops)
    finally:
        backend._fast_recognize = orig_fast


def _fmt_stats(xs: list[float]) -> str:
    if not xs:
        return "   (none)"
    mn = min(xs); mx = max(xs); med = statistics.median(xs); avg = statistics.fmean(xs)
    xs_s = sorted(xs)
    p95 = xs_s[max(0, int(round(0.95 * (len(xs_s) - 1))))]
    std = statistics.pstdev(xs) if len(xs) > 1 else 0.0
    return f"min {mn:6.1f}  med {med:6.1f}  mean {avg:6.1f}  p95 {p95:6.1f}  max {mx:6.1f}  std {std:5.1f}"


def _report(run: StageRun, img_name: str, n_crops: int) -> str:
    out = []
    out.append(f"\n── {img_name} ({n_crops} crops, {run.n_batches[0] if run.n_batches else '?'} batches) ──")
    out.append(f"  {'stage':<10} {_fmt_stats([])}".rstrip())
    for k in STAGE_KEYS:
        xs = run.timings_ms[k]
        out.append(f"  {k:<10} {_fmt_stats(xs)}")

    # Quick breakdown — median % of total
    med_total = statistics.median(run.timings_ms["total"])
    if med_total > 0:
        out.append("  ── median share of total ──")
        for k in ["det", "crop", "rec_pre", "rec_fwd", "rec_post"]:
            xs = run.timings_ms[k]
            if not xs:
                continue
            med = statistics.median(xs)
            out.append(f"    {k:<10} {med:6.1f} ms  ({100*med/med_total:5.1f}%)")
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=None)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--paddle-rec-dir", type=str,
        default="/home/jupyter/gisa_playground/OCR/deploy/rec_model/051025_data_additional_7900_v2_latest",
    )
    args = parser.parse_args()
    if args.warmup >= args.iterations:
        sys.exit("--warmup must be < --iterations")

    _cmp.PADDLE_REC_MODEL_DIR = args.paddle_rec_dir

    IMAGE_EXTS = {".png", ".jpg", ".jpeg"}
    src = args.image or PROJECT_ROOT / "tests" / "data"
    if src.is_file(): images = [src]
    elif src.is_dir(): images = sorted(p for p in src.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    else: sys.exit(f"No image at {src}")
    if not images: sys.exit(f"No image at {src}")

    print(f"Images: {[p.name for p in images]}")
    print(f"Iterations: {args.iterations} (first {args.warmup} discarded)")

    print("\nBuilding hybrid backend...")
    t_build = time.perf_counter()
    backend = _build_hybrid_backend()
    print(f"  built in {time.perf_counter() - t_build:.1f}s")

    lines_out: list[str] = []
    for img_path in images:
        img_rgb = np.asarray(Image.open(img_path).convert("RGB"))
        run = StageRun()
        n_crops = 0
        for i in range(args.iterations):
            tmp = StageRun() if i < args.warmup else run
            nc = _profile_one(backend, img_rgb, tmp, i)
            if i >= args.warmup:
                n_crops = nc
                row = [
                    f"{tmp.timings_ms[k][-1]:6.1f}"
                    for k in STAGE_KEYS
                ]
                print(f"  iter {i+1:2d}  " + "  ".join(f"{k}:{v}" for k, v in zip(STAGE_KEYS, row)))
            else:
                print(f"  iter {i+1:2d}  [warmup, discarded]  total:{tmp.timings_ms['total'][-1]:.1f}ms")
        lines_out.append(_report(run, img_path.name, n_crops))

    closer = getattr(backend, "close", None)
    if callable(closer):
        try: closer()
        except Exception: pass
    del backend
    gc.collect(); gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    report = "\n".join(lines_out)
    print("\n" + "=" * 82)
    print("  Hybrid per-stage latency summary")
    print("=" * 82)
    print(report)
    if args.output:
        args.output.write_text(report, encoding="utf-8")
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
