#!/usr/bin/env python3
"""Profile fullpytorch OCR stages and compare recognizer transfer paths."""

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


def _stats(xs: list[float]) -> str:
    if not xs:
        return "     n/a"
    ordered = sorted(xs)
    p95 = ordered[max(0, int(round(0.95 * (len(ordered) - 1))))]
    return (
        f"min {min(xs):7.1f}  med {statistics.median(xs):7.1f}  "
        f"mean {statistics.fmean(xs):7.1f}  p95 {p95:7.1f}  max {max(xs):7.1f}"
    )


def _profile_fast_recognize(
    backend: Any, crops: list[np.ndarray], stats: StageStats
) -> list[tuple[str, float]]:
    if not crops:
        stats.add("rec_pre", 0.0)
        stats.add("rec_fwd", 0.0)
        stats.add("rec_post", 0.0)
        stats.add("rec", 0.0)
        stats.n_batches.append(0)
        return []

    imgC, imgH = backend._rec_imgC, backend._rec_imgH
    device = "cuda" if backend.use_gpu else "cpu"
    model_dtype = torch.float32 if device == "cpu" else backend._torch_dtype(torch)
    recognizer = backend.system.text_recognizer
    rec_res: list[tuple[str, float]] = [("", 0.0)] * len(crops)
    batches, resized_widths = backend._recognition_batch_plan(crops)

    rec_start = time.perf_counter()
    pre_total = 0.0
    fwd_total = 0.0
    post_total = 0.0

    for target_w, batch_indices in batches:
        if torch.cuda.is_available():
            pre_start = torch.cuda.Event(enable_timing=True)
            pre_end = torch.cuda.Event(enable_timing=True)
            fwd_end = torch.cuda.Event(enable_timing=True)
            post_end = torch.cuda.Event(enable_timing=True)
            pre_start.record()
        else:
            pre_wall = time.perf_counter()

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
            pre_end.record()
        else:
            pre_total += (time.perf_counter() - pre_wall) * 1000
            fwd_wall = time.perf_counter()

        with torch.no_grad():
            preds = recognizer.net(batch)

        if torch.cuda.is_available():
            fwd_end.record()
        else:
            fwd_total += (time.perf_counter() - fwd_wall) * 1000
            post_wall = time.perf_counter()

        preds_f = preds.float()
        preds_idx = preds_f.argmax(dim=2).cpu().numpy()
        preds_prob = preds_f.max(dim=2).values.cpu().numpy()
        batch_results = recognizer.postprocess_op.decode(
            preds_idx, preds_prob, is_remove_duplicate=True
        )
        for batch_pos, result in enumerate(batch_results):
            rec_res[int(batch_indices[batch_pos])] = result

        if torch.cuda.is_available():
            post_end.record()
            torch.cuda.synchronize()
            pre_total += pre_start.elapsed_time(pre_end)
            fwd_total += pre_end.elapsed_time(fwd_end)
            post_total += fwd_end.elapsed_time(post_end)
        else:
            post_total += (time.perf_counter() - post_wall) * 1000

    stats.add("rec_pre", pre_total)
    stats.add("rec_fwd", fwd_total)
    stats.add("rec_post", post_total)
    stats.add("rec", (time.perf_counter() - rec_start) * 1000)
    stats.n_batches.append(len(batches))
    return rec_res


def _profile_one(backend: Any, image_rgb: np.ndarray, mode: str, stats: StageStats) -> None:
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

    if mode == "legacy":
        t0 = time.perf_counter()
        backend.system.text_recognizer(crops)
        _sync()
        stats.add("rec", (time.perf_counter() - t0) * 1000)
        stats.add("rec_pre", 0.0)
        stats.add("rec_fwd", 0.0)
        stats.add("rec_post", 0.0)
        stats.n_batches.append(-1)
    else:
        _profile_fast_recognize(backend, crops, stats)

    _sync()
    stats.add("total", (time.perf_counter() - total_start) * 1000)


def _report(name: str, stats: StageStats) -> str:
    lines = [f"\n{name}"]
    if stats.n_boxes:
        lines.append(f"  boxes  : {sorted(set(stats.n_boxes))}")
    if stats.n_batches:
        values = sorted(set(v for v in stats.n_batches if v >= 0))
        if values:
            lines.append(f"  batches: {values}")
    lines.append("  stage      latency")
    lines.append("  --------   " + "-" * 58)
    for stage in STAGES:
        if stage.startswith("rec_") and not any(stats.timings_ms[stage]):
            continue
        lines.append(f"  {stage:<8}   {_stats(stats.timings_ms[stage])}")

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
    parser.add_argument("--image", type=Path, default=PROJECT_ROOT / "tests/data/good_data_3.png")
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--mode", choices=("legacy", "optimized", "both"), default="both")
    parser.add_argument("--compile-det", action="store_true")
    parser.add_argument("--compile-rec", action="store_true")
    parser.add_argument("--compile-mode", default="reduce-overhead")
    parser.add_argument("--compile-dynamic", action="store_true")
    parser.add_argument("--det-autocast-islands", action="store_true",
                        help="Wrap det neck+head as fp32 islands under autocast fp16")
    parser.add_argument("--output", type=Path, default=None)
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
    for leaf in ("neck", "head"):
        setattr(det_net, leaf, Fp32Island(getattr(det_net, leaf)))

    original_forward = det_net.forward

    def forward(x):
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            return original_forward(x)

    det_net.forward = forward  # type: ignore[method-assign]
    print("  installed det autocast+islands (neck, head in fp32)")


def _maybe_compile_backend(backend: Any, args: argparse.Namespace) -> None:
    if not (args.compile_det or args.compile_rec):
        return
    if not hasattr(torch, "compile"):
        raise RuntimeError("This PyTorch build does not expose torch.compile")
    compile_kwargs = {
        "mode": args.compile_mode,
        "dynamic": bool(args.compile_dynamic),
    }
    if args.compile_det:
        backend.system.text_detector.net = torch.compile(
            backend.system.text_detector.net, **compile_kwargs
        )
        print(
            "  compiled detector "
            f"(mode={args.compile_mode}, dynamic={args.compile_dynamic})"
        )
    if args.compile_rec:
        backend.system.text_recognizer.net = torch.compile(
            backend.system.text_recognizer.net, **compile_kwargs
        )
        print(
            "  compiled recognizer "
            f"(mode={args.compile_mode}, dynamic={args.compile_dynamic})"
        )


def main() -> int:
    args = parse_args()
    if args.warmup >= args.iterations:
        sys.exit("--warmup must be less than --iterations")
    if not args.image.exists():
        sys.exit(f"Image not found: {args.image}")

    image_rgb = _load_rgb(args.image)
    modes = ["legacy", "optimized"] if args.mode == "both" else [args.mode]
    reports: list[str] = []

    for mode in modes:
        print(f"\nBuilding fullpytorch backend for {mode} profile...")
        backend = _build_fullpytorch_backend()
        if args.det_autocast_islands:
            _apply_det_autocast_islands(backend)
        _maybe_compile_backend(backend, args)
        stats = StageStats()
        for i in range(args.iterations):
            current = StageStats() if i < args.warmup else stats
            _profile_one(backend, image_rgb, mode, current)
            tag = "warm" if i < args.warmup else "meas"
            print(
                f"  {mode:<9} iter {i + 1:2d} [{tag}] "
                f"boxes={current.n_boxes[-1]:2d} total={current.timings_ms['total'][-1]:7.1f} ms"
            )
        reports.append(_report(mode, stats))
        closer = getattr(backend, "close", None)
        if callable(closer):
            closer()
        del backend
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    report = "\n".join(reports)
    print("\n" + "=" * 88)
    print("FullPyTorch Stage Profile")
    print("=" * 88)
    print(report)
    if args.output:
        args.output.write_text(report + "\n", encoding="utf-8")
        print(f"\nSaved to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
