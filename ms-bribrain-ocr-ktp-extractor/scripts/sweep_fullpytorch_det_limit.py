#!/usr/bin/env python3
"""Sweep fullpytorch detector resize limits against OCR quality and latency."""

from __future__ import annotations

import argparse
import gc
import os
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402

import scripts.compare_ocr_lines as compare  # noqa: E402
from scripts.compare_ocr_lines import Line, _match_by_centroid, _result_to_lines  # noqa: E402
from src.services.ocr_backends import AutoKernelPPOCRv5Backend, _crop_text_region  # noqa: E402


@dataclass(frozen=True)
class SweepResult:
    limit_type: str
    limit_side_len: int
    input_shape: tuple[int, int]
    boxes: int
    exact: int
    total_ref: int
    missing: int
    orphans: int
    det_ms: float
    crop_ms: float
    rec_ms: float
    total_ms: float
    diffs: tuple[str, ...]

    @property
    def pct(self) -> float:
        return 100.0 * self.exact / self.total_ref if self.total_ref else 0.0

    @property
    def pixels(self) -> int:
        return self.input_shape[0] * self.input_shape[1]


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def _median(xs: list[float]) -> float:
    return statistics.median(xs) if xs else 0.0


def _set_detector_resize(detector: Any, limit_side_len: int, limit_type: str) -> None:
    from pytorchocr.data import create_operators

    detector.args.det_limit_side_len = limit_side_len
    detector.args.det_limit_type = limit_type
    detector.preprocess_op = create_operators(
        [
            {
                "DetResizeForTest": {
                    "limit_side_len": float(limit_side_len),
                    "limit_type": limit_type,
                }
            },
            {
                "NormalizeImage": {
                    "std": [0.229, 0.224, 0.225],
                    "mean": [0.485, 0.456, 0.406],
                    "scale": "1./255.",
                    "order": "hwc",
                }
            },
            {"ToCHWImage": None},
            {"KeepKeys": {"keep_keys": ["image", "shape"]}},
        ]
    )


def _det_input_shape(detector: Any, image_rgb: np.ndarray) -> tuple[int, int]:
    from pytorchocr.data import transform

    data = transform({"image": image_rgb}, detector.preprocess_op)
    img, _shape_list = data
    return int(img.shape[1]), int(img.shape[2])


def _lines_from_boxes_rec(boxes: list[np.ndarray], rec_res: list[tuple[str, float]]) -> list[Line]:
    lines: list[Line] = []
    for box, rec in zip(boxes, rec_res):
        if rec is None:
            continue
        text, score = rec
        lines.append(Line(text=str(text), score=float(score), poly=np.asarray(box)))
    return lines


def _score_lines(ref: list[Line], candidate: list[Line]) -> tuple[int, int, int, tuple[str, ...]]:
    matches = _match_by_centroid(ref, candidate)
    exact = 0
    missing = 0
    used = {j for j in matches if j is not None}
    diffs: list[str] = []
    for idx, ref_line in enumerate(ref):
        match_idx = matches[idx]
        if match_idx is None:
            missing += 1
            diffs.append(f"{idx}:<missing> expected={ref_line.text!r}")
            continue
        got = candidate[match_idx].text
        if got == ref_line.text:
            exact += 1
        else:
            diffs.append(f"{idx}:{got!r} expected={ref_line.text!r}")
    orphans = len([idx for idx in range(len(candidate)) if idx not in used])
    return exact, missing, orphans, tuple(diffs[:8])


def _build_paddle_reference(args: argparse.Namespace, image_rgb: np.ndarray) -> list[Line]:
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    compare.PADDLE_REC_MODEL_DIR = args.paddle_rec_dir
    compare.PADDLE_CONFIG_PATH = args.paddle_config
    backend = compare._build_paddle_backend()
    try:
        # Warm once to avoid reporting first-run setup behavior.
        backend.predict(image_rgb)
        raw = backend.predict(image_rgb)
        return _result_to_lines(raw)
    finally:
        close = getattr(backend, "close", None)
        if callable(close):
            close()


def _sweep_one(
    backend: Any,
    image_rgb: np.ndarray,
    ref_lines: list[Line],
    limit_side_len: int,
    limit_type: str,
    iterations: int,
    warmup: int,
) -> SweepResult:
    _set_detector_resize(backend.system.text_detector, limit_side_len, limit_type)
    input_shape = _det_input_shape(backend.system.text_detector, image_rgb)
    measured_det: list[float] = []
    measured_crop: list[float] = []
    measured_rec: list[float] = []
    measured_total: list[float] = []
    last_boxes: list[np.ndarray] = []
    last_rec: list[tuple[str, float]] = []

    for idx in range(iterations):
        record = idx >= warmup
        _sync()
        t_total = time.perf_counter()

        t0 = time.perf_counter()
        boxes = backend._detect_boxes(image_rgb)
        _sync()
        det_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        crops = [_crop_text_region(image_rgb, box) for box in boxes]
        crop_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        rec_res = backend._fast_recognize(crops)
        _sync()
        rec_ms = (time.perf_counter() - t0) * 1000

        total_ms = (time.perf_counter() - t_total) * 1000
        if record:
            measured_det.append(det_ms)
            measured_crop.append(crop_ms)
            measured_rec.append(rec_ms)
            measured_total.append(total_ms)
            last_boxes = boxes
            last_rec = rec_res

    candidate_lines = _lines_from_boxes_rec(last_boxes, last_rec)
    exact, missing, orphans, diffs = _score_lines(ref_lines, candidate_lines)
    return SweepResult(
        limit_type=limit_type,
        limit_side_len=limit_side_len,
        input_shape=input_shape,
        boxes=len(candidate_lines),
        exact=exact,
        total_ref=len(ref_lines),
        missing=missing,
        orphans=orphans,
        det_ms=_median(measured_det),
        crop_ms=_median(measured_crop),
        rec_ms=_median(measured_rec),
        total_ms=_median(measured_total),
        diffs=diffs,
    )


def _format_report(results: list[SweepResult], ref_count: int) -> str:
    lines: list[str] = []
    add = lines.append
    add("=" * 120)
    add("  FullPyTorch detector limit sweep")
    add(f"  Reference lines: {ref_count}")
    add("=" * 120)
    add(
        "  "
        f"{'limit':>7} {'type':>5} {'input':>11} {'Mpixels':>8} "
        f"{'boxes':>5} {'exact':>9} {'miss':>4} {'orph':>4} "
        f"{'det':>7} {'rec':>7} {'total':>7} {'ms/Mpix':>8}"
    )
    add(
        "  "
        f"{'-'*7} {'-'*5} {'-'*11} {'-'*8} "
        f"{'-'*5} {'-'*9} {'-'*4} {'-'*4} "
        f"{'-'*7} {'-'*7} {'-'*7} {'-'*8}"
    )
    for result in results:
        mpixels = result.pixels / 1_000_000
        ms_per_mpix = result.total_ms / mpixels if mpixels else 0.0
        add(
            "  "
            f"{result.limit_side_len:7d} {result.limit_type:>5} "
            f"{result.input_shape[1]}x{result.input_shape[0]:<5} "
            f"{mpixels:8.3f} {result.boxes:5d} "
            f"{result.exact:2d}/{result.total_ref:<3d} {result.pct:5.1f}% "
            f"{result.missing:4d} {result.orphans:4d} "
            f"{result.det_ms:7.1f} {result.rec_ms:7.1f} "
            f"{result.total_ms:7.1f} {ms_per_mpix:8.1f}"
        )
    add("")
    add("Diff samples:")
    for result in results:
        diff_preview = "; ".join(result.diffs) if result.diffs else "none"
        add(f"  {result.limit_side_len:>4}/{result.limit_type:<3}: {diff_preview}")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=PROJECT_ROOT / "tests/data/good_data_3.png")
    parser.add_argument("--limits", default="512,576,640,704,768,832,896,960,1024,1080")
    parser.add_argument("--limit-type", choices=("max", "min", "resize_long"), default="max")
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--paddle-config",
        default=str(PROJECT_ROOT / "PaddleOCR_server_nohpi.yaml"),
    )
    parser.add_argument(
        "--paddle-rec-dir",
        default=str(PROJECT_ROOT / "src/models/server_models/ppocrv5_server_rec_source"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.warmup >= args.iterations:
        sys.exit("--warmup must be less than --iterations")
    limits = [int(part.strip()) for part in args.limits.split(",") if part.strip()]
    image_rgb = _load_rgb(args.image)

    print(f"image: {args.image}  size={image_rgb.shape[1]}x{image_rgb.shape[0]}")
    print("Building Paddle reference...")
    ref_lines = _build_paddle_reference(args, image_rgb)
    print(f"Reference lines: {len(ref_lines)}")

    print("Building fullpytorch backend...")
    backend = compare._build_fullpytorch_backend()
    results: list[SweepResult] = []
    try:
        for limit in limits:
            print(f"  sweep {args.limit_type}:{limit}...", flush=True)
            result = _sweep_one(
                backend=backend,
                image_rgb=image_rgb,
                ref_lines=ref_lines,
                limit_side_len=limit,
                limit_type=args.limit_type,
                iterations=args.iterations,
                warmup=args.warmup,
            )
            results.append(result)
            print(
                f"    input={result.input_shape[1]}x{result.input_shape[0]} "
                f"boxes={result.boxes} exact={result.exact}/{result.total_ref} "
                f"total={result.total_ms:.1f}ms"
            )
    finally:
        close = getattr(backend, "close", None)
        if callable(close):
            close()
        del backend
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    report = _format_report(results, len(ref_lines))
    print()
    print(report)
    if args.output:
        args.output.write_text(report + "\n", encoding="utf-8")
        print(f"\nSaved to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
