#!/usr/bin/env python3
"""Ad-hoc bench: fullpytorch predict() latency + peak VRAM at different
det_limit_side_len / det_limit_type values, on a single image.

Reads env:
  BENCH_IMAGE, BENCH_ITERS (default 10), BENCH_WARMUP (default 2).
Configs are hard-coded below — runs them all sequentially in-process,
tearing down each backend between configs so CUDA cache resets cleanly.
"""
from __future__ import annotations

import gc
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402

# Mirror run_server_fullpytorch.sh env so the compile / fp16 / compile-rec
# paths match production.
os.environ.setdefault("OCR_BACKEND", "fullpytorch")
os.environ.setdefault("OCR_AUTOKERNEL_ENABLED", "1")
os.environ.setdefault("AUTOKERNEL_ROOT", str(PROJECT_ROOT / "autokernel"))
os.environ.setdefault("AUTOKERNEL_PPOCR_ROOT", str(PROJECT_ROOT / "PaddleOCR2Pytorch"))
os.environ.setdefault("AUTOKERNEL_PPOCRV5_SERVER_DET_PTH", str(PROJECT_ROOT / "autokernel/workspace/ppocrv5/server_det.pth"))
os.environ.setdefault("AUTOKERNEL_PPOCRV5_SERVER_REC_PTH", str(PROJECT_ROOT / "autokernel/workspace/ppocrv5/server_rec.pth"))
os.environ.setdefault("OCR_AUTOKERNEL_AUTO_CONVERT_WEIGHTS", "0")
os.environ.setdefault("OCR_AUTOKERNEL_DTYPE", "float16")
os.environ.setdefault("OCR_AUTOKERNEL_DET_DTYPE", "float16")
os.environ.setdefault("OCR_AUTOKERNEL_TORCH_COMPILE_REC", "1")
os.environ.setdefault("OCR_AUTOKERNEL_TORCH_COMPILE_MODE", "default")
os.environ.setdefault("OCR_AUTOKERNEL_REC_BATCH_SIZE", "8")
os.environ.setdefault("OCR_AUTOKERNEL_WARMUP_IMAGE_PATH", str(PROJECT_ROOT / "tests/data/good_data_3.png"))

from src.services.ocr_backends import create_ocr_backend  # noqa: E402
from types import SimpleNamespace  # noqa: E402


def _make_settings(limit_side_len: int, limit_type: str) -> SimpleNamespace:
    return SimpleNamespace(
        ocr_backend="fullpytorch",
        ocr_autokernel_enabled=True,
        ocr_server_config_path="configs/PaddleOCR_server_nohpi.yaml",
        ocr_mobile_config_path="configs/PaddleOCR_mobile.yaml",
        ocr_autokernel_root=os.environ["AUTOKERNEL_ROOT"],
        ocr_autokernel_ppocr_root=os.environ["AUTOKERNEL_PPOCR_ROOT"],
        ocr_autokernel_workspace_path=str(PROJECT_ROOT / "autokernel/workspace/graph_capture_eval"),
        ocr_autokernel_det_weights_path=os.environ["AUTOKERNEL_PPOCRV5_SERVER_DET_PTH"],
        ocr_autokernel_rec_weights_path=os.environ["AUTOKERNEL_PPOCRV5_SERVER_REC_PTH"],
        ocr_autokernel_det_source_path="",
        ocr_autokernel_rec_source_path="",
        ocr_autokernel_auto_convert_weights=False,
        ocr_autokernel_optimize_recognizer=True,
        ocr_autokernel_optimize_detector=False,
        ocr_autokernel_rec_batch_size=8,
        ocr_autokernel_rec_image_shape="3,48,320",
        ocr_autokernel_rec_bucket_max_width_ratio=1.30,
        ocr_autokernel_det_limit_side_len=limit_side_len,
        ocr_autokernel_det_limit_type=limit_type,
        ocr_autokernel_torch_compile=False,
        ocr_autokernel_torch_compile_det=False,
        ocr_autokernel_torch_compile_rec=True,
        ocr_autokernel_torch_compile_mode="default",
        ocr_autokernel_torch_compile_dynamic=True,
        ocr_autokernel_warmup_image_path=os.environ["OCR_AUTOKERNEL_WARMUP_IMAGE_PATH"],
        ocr_autokernel_dtype="float16",
        ocr_autokernel_det_dtype="float16",
        ocr_autokernel_exclude_kernel_types=None,
    )


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _pct(xs, p):
    xs = sorted(xs)
    return xs[max(0, int(round(p * (len(xs) - 1))))]


def bench(limit_side_len: int, limit_type: str, image_rgb: np.ndarray, iters: int, warmup: int) -> dict:
    label = f"{limit_type}/{limit_side_len}"
    print(f"\n=== {label} ===", flush=True)

    settings = _make_settings(limit_side_len, limit_type)
    t0 = time.perf_counter()
    backend = create_ocr_backend(settings, use_gpu=True)
    print(f"  build: {time.perf_counter()-t0:.1f}s", flush=True)

    # Skip stock shape-priming warmup — it's ~5-8 min (compile-rec across
    # a grid of widths) and doesn't change steady-state numbers on a fixed
    # image. Use extra per-image warmup iters instead.
    t0 = time.perf_counter()
    for _ in range(warmup):
        _sync()
        backend.predict(image_rgb)
    _sync()
    print(f"  per-image warmup ({warmup} iters): {time.perf_counter()-t0:.1f}s", flush=True)

    torch.cuda.reset_peak_memory_stats()

    timings = []
    for i in range(iters):
        _sync()
        t0 = time.perf_counter()
        backend.predict(image_rgb)
        _sync()
        ms = (time.perf_counter() - t0) * 1000
        timings.append(ms)
        print(f"    iter {i+1:2d}  {ms:8.1f} ms", flush=True)

    peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    reserved_mb = torch.cuda.max_memory_reserved() / (1024 * 1024)

    closer = getattr(backend, "close", None)
    if callable(closer):
        try: closer()
        except Exception: pass
    del backend
    gc.collect(); gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    try: torch._dynamo.reset()
    except Exception: pass

    return {
        "label": label,
        "min": min(timings),
        "median": statistics.median(timings),
        "mean": statistics.fmean(timings),
        "p95": _pct(timings, 0.95),
        "max": max(timings),
        "peak_mb": peak_mb,
        "reserved_mb": reserved_mb,
        "n": len(timings),
    }


def main() -> None:
    image_path = Path(os.environ.get("BENCH_IMAGE", "/home/jupyter/james_playground/dokumentasi_kredit_perijinan_siup_2102240969591_page_1.png"))
    iters = int(os.environ.get("BENCH_ITERS", "10"))
    warmup = int(os.environ.get("BENCH_WARMUP", "2"))

    if not image_path.exists():
        sys.exit(f"image not found: {image_path}")

    img = np.asarray(Image.open(image_path).convert("RGB"))
    print(f"image: {image_path.name}  shape={img.shape}  iters={iters}  warmup={warmup}")
    print(f"device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")

    configs = [
        (64, "min"),       # current prod
        (1280, "max"),
        (960, "max"),
    ]

    results = []
    for side_len, limit_type in configs:
        results.append(bench(side_len, limit_type, img, iters, warmup))

    print("\n" + "=" * 86)
    print(f"{'config':<14} {'min':>8} {'med':>8} {'mean':>8} {'p95':>8} {'max':>8} {'peakMB':>9} {'resvMB':>9}")
    print("-" * 86)
    for r in results:
        print(f"{r['label']:<14} {r['min']:8.1f} {r['median']:8.1f} {r['mean']:8.1f} {r['p95']:8.1f} {r['max']:8.1f} {r['peak_mb']:9.0f} {r['reserved_mb']:9.0f}")
    print()


if __name__ == "__main__":
    main()
