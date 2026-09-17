#!/usr/bin/env python3
"""Throughput benchmark: paddle vs hybrid (full predict loop).

Each backend is built once, warmed, then the same image is fed through
``.predict()`` N times sequentially. Reports steady-state latency after
warmup (min/median/mean/p95/max) so the first-request compile spike
doesn't skew the number.

Usage:
    .venv/bin/python -u scripts/bench_throughput.py
    .venv/bin/python -u scripts/bench_throughput.py --iterations 20
    .venv/bin/python -u scripts/bench_throughput.py --only hybrid --image tests/data/good_data.png
"""

from __future__ import annotations

import argparse
import gc
import logging
import statistics
import sys
import time
from dataclasses import dataclass
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

# Reuse the exact backend builders the comparison script uses so the
# weights, batch sizes, and kernel replacements are identical to what
# runs in production tests.
from scripts.compare_ocr_lines import (  # noqa: E402
    BACKEND_BUILDERS,
    _build_paddle_backend,
    _build_hybrid_backend,
)
import scripts.compare_ocr_lines as _cmp  # noqa: E402


@dataclass
class IterStats:
    label: str
    n: int
    warmup: int
    measured: list[float]  # milliseconds

    @property
    def min(self) -> float: return min(self.measured)
    @property
    def max(self) -> float: return max(self.measured)
    @property
    def median(self) -> float: return statistics.median(self.measured)
    @property
    def mean(self) -> float: return statistics.fmean(self.measured)
    @property
    def p95(self) -> float:
        xs = sorted(self.measured)
        idx = max(0, int(round(0.95 * (len(xs) - 1))))
        return xs[idx]
    @property
    def stdev(self) -> float:
        return statistics.pstdev(self.measured) if len(self.measured) > 1 else 0.0


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def _bench_backend(
    backend_name: str,
    image_paths: list[Path],
    iterations: int,
    warmup: int,
) -> dict[Path, IterStats]:
    print(f"\n=== Backend: {backend_name} ===")
    t_build = time.perf_counter()
    backend = BACKEND_BUILDERS[backend_name]()
    print(f"  built in {time.perf_counter() - t_build:.1f}s")

    per_image: dict[Path, IterStats] = {}
    for img_path in image_paths:
        img_rgb = _load_rgb(img_path)
        timings: list[float] = []
        print(f"  {img_path.name}:")
        for i in range(iterations):
            _sync()
            t0 = time.perf_counter()
            backend.predict(img_rgb)
            _sync()
            ms = (time.perf_counter() - t0) * 1000
            tag = "warm" if i < warmup else "meas"
            print(f"    iter {i+1:2d} [{tag}] {ms:8.1f} ms")
            if i >= warmup:
                timings.append(ms)
        per_image[img_path] = IterStats(
            label=backend_name, n=iterations, warmup=warmup, measured=timings
        )

    closer = getattr(backend, "close", None)
    if callable(closer):
        try: closer()
        except Exception as exc:
            print(f"  close() raised: {exc}")
    del backend
    gc.collect(); gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache(); torch.cuda.synchronize()
    try: torch._dynamo.reset()
    except Exception: pass
    return per_image


def _format_report(
    results: dict[str, dict[Path, IterStats]],
    image_paths: list[Path],
    backends: list[str],
) -> str:
    lines: list[str] = []
    w = lines.append
    w("=" * 86)
    w("  Throughput benchmark — full predict() loop, steady-state after warmup")
    w(f"  Backends: {'  vs  '.join(backends)}")
    w("=" * 86)
    w("")

    for img_path in image_paths:
        w(f"Image: {img_path.name}")
        w(f"  {'backend':<11} {'min':>8}  {'median':>8}  {'mean':>8}  {'p95':>8}  {'max':>8}  {'stdev':>7}  {'n':>3}")
        w(f"  {'─'*11} {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*7}  {'─'*3}")
        for b in backends:
            s = results[b].get(img_path)
            if s is None or not s.measured:
                w(f"  {b:<11} <no data>")
                continue
            w(f"  {b:<11} {s.min:7.1f}  {s.median:7.1f}  {s.mean:7.1f}  {s.p95:7.1f}  {s.max:7.1f}  {s.stdev:6.1f}  {len(s.measured):3d}")

        # speedup / slowdown relative to paddle, if both present
        if "paddle" in backends and len(backends) > 1:
            p = results["paddle"].get(img_path)
            if p and p.measured:
                for b in backends:
                    if b == "paddle":
                        continue
                    s = results[b].get(img_path)
                    if s and s.measured:
                        ratio = p.median / s.median if s.median else float("inf")
                        delta = p.median - s.median
                        verb = "faster" if ratio > 1 else "slower"
                        w(f"  → {b} vs paddle (median): {ratio:.2f}× {verb} ({delta:+.1f} ms)")
        w("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=None,
                        help="Single image path or directory (defaults to tests/data)")
    parser.add_argument("--iterations", type=int, default=12,
                        help="Total iterations per image (warmup + measured)")
    parser.add_argument("--warmup", type=int, default=2,
                        help="Warmup iterations to discard")
    parser.add_argument("--only", type=str, default="paddle,hybrid",
                        help="Comma-separated backends (paddle,hybrid,autokernel)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Write report to file")
    parser.add_argument(
        "--paddle-rec-dir", type=str,
        default="/home/jupyter/gisa_playground/OCR/deploy/rec_model/051025_data_additional_7900_v2_latest",
        help="Finetuned rec model dir — passed to PaddleOCR so both backends "
             "run identical weights.",
    )
    args = parser.parse_args()

    if args.warmup >= args.iterations:
        sys.exit("--warmup must be less than --iterations")

    _cmp.PADDLE_REC_MODEL_DIR = args.paddle_rec_dir

    backends = [b.strip().lower() for b in args.only.split(",") if b.strip()]
    for b in backends:
        if b not in BACKEND_BUILDERS:
            sys.exit(f"Unknown backend: {b}")

    # Collect images
    IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}
    src = args.image or PROJECT_ROOT / "tests" / "data"
    if src.is_file():
        images = [src]
    elif src.is_dir():
        images = sorted(p for p in src.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    else:
        sys.exit(f"No images at {src}")
    if not images:
        sys.exit(f"No images at {src}")

    print(f"Images ({len(images)}): {[p.name for p in images]}")
    print(f"Backends: {backends}")
    print(f"Iterations per image: {args.iterations} (first {args.warmup} discarded as warmup)")

    # Order passes: paddle first (cheapest cleanup), hybrid/autokernel later.
    order = {"paddle": 0, "hybrid": 1, "fullpytorch": 2, "autokernel": 3}
    passes = sorted(backends, key=lambda b: order[b])

    results: dict[str, dict[Path, IterStats]] = {}
    for b in passes:
        results[b] = _bench_backend(b, images, args.iterations, args.warmup)

    print()
    report = _format_report(results, images, backends)
    print(report)

    if args.output:
        args.output.write_text(report, encoding="utf-8")
        print(f"\nReport saved to {args.output}")


if __name__ == "__main__":
    main()
