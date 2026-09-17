#!/usr/bin/env python3
"""Break the production FullPyTorch detector path into cv2_resize / h2d / norm / fwd / d2h / post across all images."""
from __future__ import annotations

import argparse
import gc
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402

from scripts.compare_ocr_lines import _build_fullpytorch_backend  # noqa: E402
from src.services.ocr_backends import _compute_det_resize_shape, _sort_text_boxes  # noqa: E402


STAGES = ("cv2_resize", "h2d", "norm", "fwd", "d2h", "post", "filter", "total")


def _stats(xs: list[float]) -> str:
    if not xs:
        return "n/a"
    ordered = sorted(xs)
    p95 = ordered[max(0, int(round(0.95 * (len(ordered) - 1))))]
    return (
        f"min {min(xs):6.2f}  med {statistics.median(xs):6.2f}  "
        f"mean {statistics.fmean(xs):6.2f}  p95 {p95:6.2f}  max {max(xs):6.2f}"
    )


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def profile(backend, image_rgb: np.ndarray, iterations: int, warmup: int) -> dict[str, list[float]]:
    detector = backend.system.text_detector
    preprocess = backend._get_detector_preprocess_state(torch)
    src_h, src_w = image_rgb.shape[:2]
    resize_h, resize_w, ratio_h, ratio_w = _compute_det_resize_shape(
        src_h, src_w,
        limit_side_len=float(getattr(preprocess.resize_op, "limit_side_len")),
        limit_type=str(getattr(preprocess.resize_op, "limit_type", "min")),
    )
    shape_list = np.asarray([[src_h, src_w, ratio_h, ratio_w]], dtype=np.float32)

    collect = {s: [] for s in STAGES}
    box_counts: list[int] = []
    input_shape = None
    map_shape = None

    for it in range(iterations):
        _sync()
        t0 = time.perf_counter()
        resized = cv2.resize(image_rgb, (int(resize_w), int(resize_h)))
        t_cv = time.perf_counter()

        with torch.inference_mode():
            inp = torch.from_numpy(np.ascontiguousarray(resized)).to(
                device="cuda", dtype=torch.uint8
            )
            _sync()
            t_h2d = time.perf_counter()

            inp = inp.permute(2, 0, 1).unsqueeze(0).to(dtype=torch.float32)
            inp.mul_(preprocess.scale)
            inp.sub_(preprocess.mean)
            inp.div_(preprocess.std)
            _sync()
            t_norm = time.perf_counter()
            input_shape = tuple(inp.shape)

            outputs = detector.net(inp)
            _sync()
            t_fwd = time.perf_counter()

            maps_cpu = outputs["maps"].float().cpu().numpy()
            t_d2h = time.perf_counter()
            map_shape = tuple(maps_cpu.shape)

        preds = {"maps": maps_cpu}
        post_result = detector.postprocess_op(preds, shape_list)
        dt_boxes = post_result[0]["points"]
        t_post = time.perf_counter()

        if dt_boxes is None:
            dt_boxes = []
        dt_boxes = detector.filter_tag_det_res(dt_boxes, image_rgb.shape)
        dt_boxes = _sort_text_boxes([np.asarray(b) for b in dt_boxes])
        t_filter = time.perf_counter()

        if it >= warmup:
            collect["cv2_resize"].append((t_cv - t0) * 1000)
            collect["h2d"].append((t_h2d - t_cv) * 1000)
            collect["norm"].append((t_norm - t_h2d) * 1000)
            collect["fwd"].append((t_fwd - t_norm) * 1000)
            collect["d2h"].append((t_d2h - t_fwd) * 1000)
            collect["post"].append((t_post - t_d2h) * 1000)
            collect["filter"].append((t_filter - t_post) * 1000)
            collect["total"].append((t_filter - t0) * 1000)
            box_counts.append(len(dt_boxes))

    collect["_input_shape"] = input_shape  # type: ignore[assignment]
    collect["_map_shape"] = map_shape  # type: ignore[assignment]
    collect["_boxes"] = box_counts  # type: ignore[assignment]
    return collect


def _print_report(name: str, data: dict[str, list[float]]) -> str:
    lines = [f"\n=== {name} ==="]
    lines.append(f"  input: {data.get('_input_shape')}  maps: {data.get('_map_shape')}  boxes: {sorted(set(data.get('_boxes', [])))}")
    for stage in STAGES:
        lines.append(f"  {stage:<11} {_stats(data[stage])}")
    total_med = statistics.median(data["total"]) if data["total"] else 0.0
    if total_med:
        lines.append("  median share:")
        for stage in ("cv2_resize", "h2d", "norm", "fwd", "d2h", "post", "filter"):
            xs = data[stage]
            if xs:
                med = statistics.median(xs)
                lines.append(f"    {stage:<11} {med:6.2f} ms  ({100 * med / total_med:5.1f}%)")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
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
    ap.add_argument("--iterations", type=int, default=12)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    print("Building fullpytorch backend once...")
    backend = _build_fullpytorch_backend()
    reports: list[str] = []
    agg: dict[str, list[float]] = {s: [] for s in STAGES}
    for image in args.images:
        print(f"\nProfiling {image.name}...")
        img_rgb = np.asarray(Image.open(image).convert("RGB"))
        data = profile(backend, img_rgb, args.iterations, args.warmup)
        reports.append(_print_report(image.name, data))
        for s in STAGES:
            agg[s].extend(data[s])
        print(reports[-1])

    agg["_input_shape"] = "various"  # type: ignore[assignment]
    agg["_map_shape"] = "various"  # type: ignore[assignment]
    agg["_boxes"] = []  # type: ignore[assignment]
    reports.append(_print_report("AGGREGATE (all 4 images)", agg))

    print("\n" + "=" * 88)
    print("FullPyTorch Detector Substage Profile")
    print("=" * 88)
    full = "\n".join(reports)
    print(full)
    if args.output:
        args.output.write_text(full + "\n", encoding="utf-8")
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
