#!/usr/bin/env python3
"""Profile fullpytorch stages across all tests/data images with a single backend build."""

from __future__ import annotations

import argparse
import gc
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from scripts.compare_ocr_lines import _build_fullpytorch_backend  # noqa: E402
from src.services.ocr_backends import _crop_text_region  # noqa: E402


STAGES = ("det", "crop", "rec_pre", "rec_fwd", "rec_post", "rec", "total")


@dataclass
class StageStats:
    timings_ms: dict[str, list[float]] = field(
        default_factory=lambda: {stage: [] for stage in STAGES}
    )
    n_boxes: list[int] = field(default_factory=list)
    n_batches: list[int] = field(default_factory=list)

    def add(self, stage: str, ms: float) -> None:
        self.timings_ms[stage].append(ms)


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def _fmt(xs: list[float]) -> str:
    if not xs:
        return "n/a"
    ordered = sorted(xs)
    p95 = ordered[max(0, int(round(0.95 * (len(ordered) - 1))))]
    return (
        f"min {min(xs):7.1f}  med {statistics.median(xs):7.1f}  "
        f"mean {statistics.fmean(xs):7.1f}  p95 {p95:7.1f}  max {max(xs):7.1f}"
    )


def _profile_optimized(backend: Any, crops: list[np.ndarray], stats: StageStats) -> None:
    if not crops:
        stats.add("rec_pre", 0.0)
        stats.add("rec_fwd", 0.0)
        stats.add("rec_post", 0.0)
        stats.add("rec", 0.0)
        stats.n_batches.append(0)
        return

    imgC, imgH = backend._rec_imgC, backend._rec_imgH
    device = "cuda" if backend.use_gpu else "cpu"
    model_dtype = torch.float32 if device == "cpu" else backend._torch_dtype(torch)
    recognizer = backend.system.text_recognizer
    batches, resized_widths = backend._recognition_batch_plan(crops)

    rec_start = time.perf_counter()
    pre_total = fwd_total = post_total = 0.0

    for target_w, batch_indices in batches:
        if torch.cuda.is_available():
            e_pre = torch.cuda.Event(enable_timing=True)
            e_pre_end = torch.cuda.Event(enable_timing=True)
            e_fwd_end = torch.cuda.Event(enable_timing=True)
            e_post_end = torch.cuda.Event(enable_timing=True)
            e_pre.record()
        else:
            wall = time.perf_counter()

        batch = torch.zeros(
            len(batch_indices), imgC, imgH, target_w, dtype=model_dtype, device=device
        )
        for batch_pos, crop_index in enumerate(batch_indices):
            crop = crops[int(crop_index)]
            resized_w = resized_widths[int(crop_index)]
            t = torch.from_numpy(crop.copy()).to(device=device, dtype=torch.float32)
            t = t.permute(2, 0, 1).unsqueeze(0)
            t = F.interpolate(t, size=(imgH, resized_w), mode="bilinear", align_corners=False)
            t = t.squeeze(0).to(dtype=model_dtype)
            t = (t / 255.0 - 0.5) / 0.5
            batch[batch_pos, :, :, :resized_w] = t

        if torch.cuda.is_available():
            e_pre_end.record()
        else:
            pre_total += (time.perf_counter() - wall) * 1000
            wall = time.perf_counter()

        with torch.no_grad():
            preds = recognizer.net(batch)

        if torch.cuda.is_available():
            e_fwd_end.record()
        else:
            fwd_total += (time.perf_counter() - wall) * 1000
            wall = time.perf_counter()

        preds_f = preds.float()
        preds_idx = preds_f.argmax(dim=2).cpu().numpy()
        preds_prob = preds_f.max(dim=2).values.cpu().numpy()
        recognizer.postprocess_op.decode(preds_idx, preds_prob, is_remove_duplicate=True)

        if torch.cuda.is_available():
            e_post_end.record()
            torch.cuda.synchronize()
            pre_total += e_pre.elapsed_time(e_pre_end)
            fwd_total += e_pre_end.elapsed_time(e_fwd_end)
            post_total += e_fwd_end.elapsed_time(e_post_end)
        else:
            post_total += (time.perf_counter() - wall) * 1000

    stats.add("rec_pre", pre_total)
    stats.add("rec_fwd", fwd_total)
    stats.add("rec_post", post_total)
    stats.add("rec", (time.perf_counter() - rec_start) * 1000)
    stats.n_batches.append(len(batches))


def _profile_one(backend: Any, image_rgb: np.ndarray, stats: StageStats) -> None:
    _sync()
    total_start = time.perf_counter()

    t0 = time.perf_counter()
    boxes = backend._detect_boxes(image_rgb)
    _sync()
    stats.add("det", (time.perf_counter() - t0) * 1000)
    stats.n_boxes.append(len(boxes))

    t0 = time.perf_counter()
    crops = [_crop_text_region(image_rgb, box) for box in boxes]
    stats.add("crop", (time.perf_counter() - t0) * 1000)

    _profile_optimized(backend, crops, stats)

    _sync()
    stats.add("total", (time.perf_counter() - total_start) * 1000)


def _report(name: str, stats: StageStats) -> str:
    lines = [f"\n=== {name} ==="]
    if stats.n_boxes:
        lines.append(f"  boxes   : {sorted(set(stats.n_boxes))}")
    if stats.n_batches:
        vs = sorted(set(v for v in stats.n_batches if v >= 0))
        if vs:
            lines.append(f"  batches : {vs}")
    lines.append("  stage      latency (ms)")
    lines.append("  --------   " + "-" * 62)
    for stage in STAGES:
        if stage.startswith("rec_") and not any(stats.timings_ms[stage]):
            continue
        lines.append(f"  {stage:<8}   {_fmt(stats.timings_ms[stage])}")

    total = statistics.median(stats.timings_ms["total"]) if stats.timings_ms["total"] else 0
    if total:
        lines.append("  median share:")
        for stage in ("det", "crop", "rec_pre", "rec_fwd", "rec_post", "rec"):
            xs = stats.timings_ms[stage]
            if not xs or not any(xs):
                continue
            med = statistics.median(xs)
            lines.append(f"    {stage:<8} {med:7.1f} ms  ({100 * med / total:5.1f}%)")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--images",
        nargs="+",
        type=Path,
        default=[
            PROJECT_ROOT / "tests/data/good_data.png",
            PROJECT_ROOT / "tests/data/good_data_2.png",
            PROJECT_ROOT / "tests/data/good_data_3.png",
            PROJECT_ROOT / "tests/data/good_data_4.png",
        ],
    )
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--det-autocast-islands",
        action="store_true",
        help="Wrap det neck+head as fp32 islands under autocast fp16 (experiment)",
    )
    return parser.parse_args()


def _apply_det_autocast_islands(backend: Any) -> None:
    import torch.nn as nn

    class Fp32Island(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, x, *args, **kwargs):
            def to_fp32(v):
                if isinstance(v, torch.Tensor):
                    return v.float()
                if isinstance(v, list):
                    return [to_fp32(t) for t in v]
                if isinstance(v, tuple):
                    return tuple(to_fp32(t) for t in v)
                return v
            with torch.autocast(device_type="cuda", enabled=False):
                return self.inner(to_fp32(x), *to_fp32(args), **kwargs)

    det_net = backend.system.text_detector.net
    # The prod detector is wrapped in _MixedPrecisionDetNet; its .wrapped
    # is the real BaseModel with .backbone/.neck/.head. Walk down.
    target = det_net
    if hasattr(det_net, "wrapped"):
        target = det_net.wrapped
    for leaf in ("neck", "head"):
        setattr(target, leaf, Fp32Island(getattr(target, leaf)))

    original_forward = target.forward

    def forward(x):
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            return original_forward(x)

    target.forward = forward  # type: ignore[method-assign]
    # Replace the outer wrapped net entirely — bypass _MixedPrecisionDetNet so
    # autocast sees the whole backbone+neck+head forward.
    backend.system.text_detector.net = target
    print("  installed det autocast+islands (neck, head in fp32, rest autocast fp16)")


def main() -> int:
    args = parse_args()
    for p in args.images:
        if not p.exists():
            sys.exit(f"Image not found: {p}")

    print("Building fullpytorch backend once...")
    backend = _build_fullpytorch_backend()
    if args.det_autocast_islands:
        _apply_det_autocast_islands(backend)

    per_image: dict[str, StageStats] = {}
    overall = StageStats()

    # Warm once on the first image (covers compile + shape warmup the service
    # would do at startup).
    first_img = _load_rgb(args.images[0])
    for _ in range(args.warmup):
        tmp = StageStats()
        _profile_one(backend, first_img, tmp)

    for image_path in args.images:
        name = image_path.name
        print(f"\nProfiling {name} ({args.iterations} iters)...")
        img_rgb = _load_rgb(image_path)
        stats = StageStats()
        # small per-image warmup to stabilise shape-specific paths
        for i in range(args.warmup):
            tmp = StageStats()
            _profile_one(backend, img_rgb, tmp)
        for i in range(args.iterations):
            _profile_one(backend, img_rgb, stats)
            total = stats.timings_ms["total"][-1]
            det = stats.timings_ms["det"][-1]
            rec = stats.timings_ms["rec"][-1]
            print(
                f"  iter {i + 1:2d} boxes={stats.n_boxes[-1]:2d} "
                f"det={det:6.1f} rec={rec:6.1f} total={total:6.1f} ms"
            )
        per_image[name] = stats
        # accumulate into overall
        for stage in STAGES:
            overall.timings_ms[stage].extend(stats.timings_ms[stage])
        overall.n_boxes.extend(stats.n_boxes)
        overall.n_batches.extend(stats.n_batches)

    reports = [_report(name, s) for name, s in per_image.items()]
    reports.append(_report("AGGREGATE (all 4 images)", overall))
    report = "\n".join(reports)

    print("\n" + "=" * 88)
    print("FullPyTorch Per-Image Stage Profile")
    print("=" * 88)
    print(report)

    if args.output:
        args.output.write_text(report + "\n", encoding="utf-8")
        print(f"\nSaved to {args.output}")

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
