#!/usr/bin/env python3
"""Stage profiler for the full-PyTorch PP-OCRv5 detector path.

Breaks PaddleOCR2Pytorch ``TextDetector.predict`` into:

    pre       CPU resize/normalize/CHW transform
    h2d       torch tensor creation and device upload
    fwd       PyTorch detector forward
    d2h       DB map transfer back to CPU numpy
    post      DB postprocess plus box clipping/filtering
    total     end-to-end detector wall time

The default run compares the current fullpytorch detector settings
(``960/max`` from PaddleOCR2Pytorch defaults) against the project's PaddleOCR
YAML detector settings (for example ``64/min`` in ``PaddleOCR_server_nohpi``).
"""

from __future__ import annotations

import argparse
import gc
import importlib
import os
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT
AUTOKERNEL_ROOT = REPO_ROOT / "autokernel"
PPOCR_ROOT = PROJECT_ROOT / "PaddleOCR2Pytorch"
DET_PTH = AUTOKERNEL_ROOT / "workspace/ppocrv5/server_det.pth"

for path in (PROJECT_ROOT, AUTOKERNEL_ROOT, PPOCR_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

import torch  # noqa: E402


STAGES = ("pre", "h2d", "fwd", "d2h", "post", "total")


@dataclass
class DetVariant:
    name: str
    limit_side_len: float
    limit_type: str
    thresh: float = 0.3
    box_thresh: float = 0.6
    unclip_ratio: float = 1.5


@dataclass
class RunStats:
    timings_ms: dict[str, list[float]] = field(
        default_factory=lambda: {stage: [] for stage in STAGES}
    )
    input_shapes: list[tuple[int, ...]] = field(default_factory=list)
    map_shapes: list[tuple[int, ...]] = field(default_factory=list)
    box_counts: list[int] = field(default_factory=list)

    def add(self, stage: str, value_ms: float) -> None:
        self.timings_ms[stage].append(value_ms)


def _load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _fmt_stats(xs: list[float]) -> str:
    if not xs:
        return "no data"
    ordered = sorted(xs)
    p95 = ordered[max(0, int(round(0.95 * (len(ordered) - 1))))]
    return (
        f"min {min(xs):7.2f}  med {statistics.median(xs):7.2f}  "
        f"mean {statistics.fmean(xs):7.2f}  p95 {p95:7.2f}  "
        f"max {max(xs):7.2f}"
    )


def _load_project_variant(config_path: Path) -> DetVariant:
    try:
        import yaml
    except Exception as exc:  # pragma: no cover - script setup failure
        raise RuntimeError("PyYAML is required to read PaddleOCR detector config") from exc

    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    det = ((data.get("SubModules") or {}).get("TextDetection") or {})
    return DetVariant(
        name=f"project:{config_path.name}",
        limit_side_len=float(det.get("limit_side_len", 64)),
        limit_type=str(det.get("limit_type", "min")),
        thresh=float(det.get("thresh", 0.3)),
        box_thresh=float(det.get("box_thresh", 0.6)),
        unclip_ratio=float(det.get("unclip_ratio", 1.5)),
    )


def _build_detector(variant: DetVariant, use_gpu: bool) -> Any:
    os.environ["AUTOKERNEL_PPOCR_ROOT"] = str(PPOCR_ROOT)
    os.environ["AUTOKERNEL_PPOCRV5_SERVER_DET_PTH"] = str(DET_PTH)

    ppocr_model = importlib.import_module("models.ppocrv5_server")
    pytorchocr_utility = importlib.import_module("tools.infer.pytorchocr_utility")
    predict_det = importlib.import_module("tools.infer.predict_det")

    parser = pytorchocr_utility.init_args()
    args = parser.parse_args([])
    args.use_gpu = use_gpu
    args.det_algorithm = "DB"
    args.det_yaml_path = str(PPOCR_ROOT / ppocr_model.DET_YAML_RELATIVE)
    args.det_model_path = str(DET_PTH)
    args.det_limit_side_len = variant.limit_side_len
    args.det_limit_type = variant.limit_type
    args.det_db_thresh = variant.thresh
    args.det_db_box_thresh = variant.box_thresh
    args.det_db_unclip_ratio = variant.unclip_ratio
    args.image_dir = ""

    detector = predict_det.TextDetector(args)
    device = "cuda" if use_gpu else "cpu"
    det_wrapper = ppocr_model.PPOCRv5ServerDetModel()
    detector.net = det_wrapper.net.to(device).float().eval()
    return detector


def _profile_one(detector: Any, image_rgb: np.ndarray, stats: RunStats) -> int:
    from pytorchocr.data import transform

    _sync()
    total_start = time.perf_counter()

    pre_start = time.perf_counter()
    ori_im = image_rgb.copy()
    data = transform({"image": image_rgb}, detector.preprocess_op)
    img, shape_list = data
    if img is None:
        for stage in STAGES:
            stats.add(stage, 0.0)
        return 0
    img = np.expand_dims(img, axis=0).copy()
    shape_list = np.expand_dims(shape_list, axis=0)
    stats.input_shapes.append(tuple(img.shape))
    stats.add("pre", (time.perf_counter() - pre_start) * 1000)

    h2d_start = time.perf_counter()
    inp = torch.from_numpy(img)
    if detector.use_gpu:
        inp = inp.cuda()
    _sync()
    stats.add("h2d", (time.perf_counter() - h2d_start) * 1000)

    fwd_start = time.perf_counter()
    with torch.no_grad():
        outputs = detector.net(inp)
    _sync()
    stats.add("fwd", (time.perf_counter() - fwd_start) * 1000)

    d2h_start = time.perf_counter()
    if detector.det_algorithm in ["DB", "PSE", "DB++"]:
        maps = outputs["maps"].detach().cpu().numpy()
        preds = {"maps": maps}
        stats.map_shapes.append(tuple(maps.shape))
    else:  # pragma: no cover - this profiler is currently DB-only.
        raise NotImplementedError(detector.det_algorithm)
    _sync()
    stats.add("d2h", (time.perf_counter() - d2h_start) * 1000)

    post_start = time.perf_counter()
    post_result = detector.postprocess_op(preds, shape_list)
    dt_boxes = post_result[0]["points"]
    dt_boxes = detector.filter_tag_det_res(dt_boxes, ori_im.shape)
    stats.add("post", (time.perf_counter() - post_start) * 1000)

    stats.box_counts.append(0 if dt_boxes is None else len(dt_boxes))
    stats.add("total", (time.perf_counter() - total_start) * 1000)
    return stats.box_counts[-1]


def _report(variant: DetVariant, image_name: str, stats: RunStats) -> str:
    lines: list[str] = []
    add = lines.append
    add(
        f"\n{variant.name} on {image_name} "
        f"(limit_side_len={variant.limit_side_len:g}, limit_type={variant.limit_type})"
    )
    if stats.input_shapes:
        add(f"  input_shapes: {sorted(set(stats.input_shapes))}")
    if stats.map_shapes:
        add(f"  map_shapes  : {sorted(set(stats.map_shapes))}")
    if stats.box_counts:
        add(f"  boxes       : {sorted(set(stats.box_counts))}")
    add(f"  {'stage':<8} {'min':>7}  {'median':>7}  {'mean':>7}  {'p95':>7}  {'max':>7}")
    add(f"  {'-' * 8} {'-' * 7}  {'-' * 7}  {'-' * 7}  {'-' * 7}  {'-' * 7}")
    for stage in STAGES:
        add(f"  {stage:<8} {_fmt_stats(stats.timings_ms[stage])}")

    total = statistics.median(stats.timings_ms["total"]) if stats.timings_ms["total"] else 0.0
    if total:
        add("  median share:")
        for stage in ("pre", "h2d", "fwd", "d2h", "post"):
            xs = stats.timings_ms[stage]
            if xs:
                med = statistics.median(xs)
                add(f"    {stage:<6} {med:7.2f} ms  ({100 * med / total:5.1f}%)")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=PROJECT_ROOT / "tests/data/good_data_3.png")
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--variant", choices=("stock", "project", "both"), default="both")
    parser.add_argument("--paddle-config", type=Path, default=PROJECT_ROOT / "PaddleOCR_server_nohpi.yaml")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.warmup >= args.iterations:
        sys.exit("--warmup must be less than --iterations")
    if not args.image.exists():
        sys.exit(f"Image not found: {args.image}")
    for label, path in {
        "AutoKernel root": AUTOKERNEL_ROOT,
        "PaddleOCR2Pytorch root": PPOCR_ROOT,
        "detector weights": DET_PTH,
    }.items():
        if not path.exists():
            sys.exit(f"{label} not found: {path}")

    use_gpu = (not args.cpu) and torch.cuda.is_available()
    image_rgb = _load_rgb(args.image)
    variants: list[DetVariant] = []
    if args.variant in {"stock", "both"}:
        variants.append(DetVariant(name="stock:PaddleOCR2Pytorch", limit_side_len=960, limit_type="max"))
    if args.variant in {"project", "both"}:
        variants.append(_load_project_variant(args.paddle_config.resolve()))

    print(f"image      : {args.image}")
    print(f"device     : {'cuda' if use_gpu else 'cpu'}")
    print(f"iterations : {args.iterations} (discarding first {args.warmup})")

    reports: list[str] = []
    for variant in variants:
        print(f"\nBuilding detector variant: {variant.name}")
        detector = _build_detector(variant, use_gpu=use_gpu)
        measured = RunStats()
        for i in range(args.iterations):
            current = RunStats() if i < args.warmup else measured
            boxes = _profile_one(detector, image_rgb, current)
            total = current.timings_ms["total"][-1]
            tag = "warm" if i < args.warmup else "meas"
            shape = current.input_shapes[-1] if current.input_shapes else "n/a"
            print(
                f"  iter {i + 1:2d} [{tag}] boxes={boxes:2d} "
                f"input={shape} total={total:7.2f} ms"
            )
        reports.append(_report(variant, args.image.name, measured))
        del detector
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    report = "\n".join(reports)
    print("\n" + "=" * 88)
    print("Full-PyTorch Detector Stage Profile")
    print("=" * 88)
    print(report)
    if args.output:
        args.output.write_text(report + "\n", encoding="utf-8")
        print(f"\nSaved to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
