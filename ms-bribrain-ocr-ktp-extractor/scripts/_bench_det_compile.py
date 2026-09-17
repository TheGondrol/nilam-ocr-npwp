#!/usr/bin/env python3
"""Ad-hoc bench: fullpytorch predict() latency + peak VRAM, det-compile OFF
vs ON (at the new max/1280 default). Also diffs rec_texts between configs so
we catch any silent accuracy regression from the stem rewrite.

Reads env:
  BENCH_ITERS (default 10), BENCH_WARMUP (default 3).
Uses tests/data/good_data*.png.
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

# Mirror run_server_fullpytorch.sh env so compile / fp16 / compile-rec paths
# match production.
os.environ.setdefault("OCR_BACKEND", "fullpytorch")
os.environ.setdefault("OCR_AUTOKERNEL_ENABLED", "1")
os.environ.setdefault("AUTOKERNEL_ROOT", str(PROJECT_ROOT / "autokernel"))
os.environ.setdefault("AUTOKERNEL_PPOCR_ROOT", str(PROJECT_ROOT / "PaddleOCR2Pytorch"))
os.environ.setdefault(
    "AUTOKERNEL_PPOCRV5_SERVER_DET_PTH",
    str(PROJECT_ROOT / "autokernel/workspace/ppocrv5/server_det.pth"),
)
os.environ.setdefault(
    "AUTOKERNEL_PPOCRV5_SERVER_REC_PTH",
    str(PROJECT_ROOT / "autokernel/workspace/ppocrv5/server_rec.pth"),
)
os.environ.setdefault("OCR_AUTOKERNEL_AUTO_CONVERT_WEIGHTS", "0")
os.environ.setdefault("OCR_AUTOKERNEL_DTYPE", "float16")
os.environ.setdefault("OCR_AUTOKERNEL_DET_DTYPE", "float16")
os.environ.setdefault("OCR_AUTOKERNEL_TORCH_COMPILE_REC", "1")
os.environ.setdefault("OCR_AUTOKERNEL_TORCH_COMPILE_MODE", "default")
os.environ.setdefault("OCR_AUTOKERNEL_REC_BATCH_SIZE", "8")
os.environ.setdefault(
    "OCR_AUTOKERNEL_WARMUP_IMAGE_PATH",
    str(PROJECT_ROOT / "tests/data/good_data_3.png"),
)

from src.services.ocr_backends import create_ocr_backend  # noqa: E402
from types import SimpleNamespace  # noqa: E402


IMAGES = sorted((PROJECT_ROOT / "tests/data").glob("good_data*.png"))


def _make_settings(compile_det: bool) -> SimpleNamespace:
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
        ocr_autokernel_det_limit_side_len=1280,
        ocr_autokernel_det_limit_type="max",
        ocr_autokernel_torch_compile=False,
        ocr_autokernel_torch_compile_det=compile_det,
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


def _extract_texts(result):
    if not result or result[0] is None:
        return []
    first = result[0]
    if hasattr(first, "get"):
        return [str(t) for t in (first.get("rec_texts") or [])]
    try:
        return [str(t) for t in (first["rec_texts"] or [])]
    except Exception:
        return []


def bench(compile_det: bool, iters: int, warmup: int) -> dict:
    label = "det_compile=ON" if compile_det else "det_compile=OFF"
    print(f"\n=== {label} ===", flush=True)

    settings = _make_settings(compile_det)
    t0 = time.perf_counter()
    backend = create_ocr_backend(settings, use_gpu=True)
    print(f"  build: {time.perf_counter()-t0:.1f}s", flush=True)

    per_image = {}
    texts = {}
    peak_mb_all = 0.0
    reserved_mb_all = 0.0

    for img_path in IMAGES:
        img = np.asarray(Image.open(img_path).convert("RGB"))
        print(f"  -- {img_path.name} ({img.shape[1]}x{img.shape[0]}) --", flush=True)

        # per-image warmup to absorb first-shape compile and JIT guards
        t0 = time.perf_counter()
        for _ in range(warmup):
            _sync()
            backend.predict(img)
        _sync()
        print(f"    warmup ({warmup} iters): {time.perf_counter()-t0:.1f}s", flush=True)

        torch.cuda.reset_peak_memory_stats()
        timings = []
        for i in range(iters):
            _sync()
            t0 = time.perf_counter()
            result = backend.predict(img)
            _sync()
            ms = (time.perf_counter() - t0) * 1000
            timings.append(ms)
            if i == iters - 1:
                texts[img_path] = _extract_texts(result)
            print(f"    iter {i+1:2d}  {ms:8.1f} ms", flush=True)

        peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        reserved_mb = torch.cuda.max_memory_reserved() / (1024 * 1024)
        peak_mb_all = max(peak_mb_all, peak_mb)
        reserved_mb_all = max(reserved_mb_all, reserved_mb)

        per_image[img_path] = {
            "min": min(timings),
            "median": statistics.median(timings),
            "mean": statistics.fmean(timings),
            "p95": _pct(timings, 0.95),
            "max": max(timings),
            "n": len(timings),
            "peak_mb": peak_mb,
        }

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
        "per_image": per_image,
        "texts": texts,
        "peak_mb_all": peak_mb_all,
        "reserved_mb_all": reserved_mb_all,
    }


def diff_lines(a, b):
    sa, sb = set(a), set(b)
    common = sa & sb
    only_a = sorted(sa - sb)
    only_b = sorted(sb - sa)
    return len(common), only_a, only_b


def main() -> None:
    iters = int(os.environ.get("BENCH_ITERS", "10"))
    warmup = int(os.environ.get("BENCH_WARMUP", "3"))
    only = os.environ.get("BENCH_ONLY", "both").lower()  # off|on|both
    if not IMAGES:
        sys.exit("No good_data images found")
    print(f"Images: {[p.name for p in IMAGES]}  iters={iters}  warmup={warmup}  only={only}")
    print(f"device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")

    baseline = bench(False, iters, warmup) if only in ("off", "both") else None
    candidate = bench(True, iters, warmup) if only in ("on", "both") else None

    if baseline is None or candidate is None:
        print("\n(single-config run — skipping comparison tables)")
        return

    print("\n" + "=" * 94)
    print(f"  Latency (ms) — det-compile OFF vs ON  [n={iters}/image, warmup={warmup}]")
    print("=" * 94)
    hdr = f"{'image':<22} {'cfg':<5} {'min':>7} {'med':>7} {'mean':>7} {'p95':>7} {'max':>7}"
    print(hdr); print("-" * len(hdr))
    for img in IMAGES:
        for cfg, r in (("OFF", baseline), ("ON", candidate)):
            s = r["per_image"][img]
            print(f"{img.name:<22} {cfg:<5} {s['min']:7.1f} {s['median']:7.1f} "
                  f"{s['mean']:7.1f} {s['p95']:7.1f} {s['max']:7.1f}")
        off = baseline["per_image"][img]["median"]
        on = candidate["per_image"][img]["median"]
        speedup = off / on if on else float("inf")
        delta = off - on
        verb = "faster" if speedup > 1 else "slower"
        print(f"  → ON vs OFF (median): {speedup:.2f}× {verb} ({delta:+.1f} ms)")
        print()

    print(f"peak VRAM across all images:  OFF={baseline['peak_mb_all']:.0f} MB   "
          f"ON={candidate['peak_mb_all']:.0f} MB   "
          f"(reserved OFF={baseline['reserved_mb_all']:.0f} MB, "
          f"ON={candidate['reserved_mb_all']:.0f} MB)")

    # Parity check
    print("\n" + "=" * 94)
    print("  Text parity — rec_texts OFF (baseline) vs ON (candidate), case+whitespace sensitive")
    print("=" * 94)
    print(f"{'image':<22} {'base':>5} {'cand':>5} {'∩':>5} {'base-only':>9} {'cand-only':>9}")
    print("-" * 70)
    totals = [0, 0, 0, 0, 0]
    for img in IMAGES:
        a = baseline["texts"].get(img, [])
        b = candidate["texts"].get(img, [])
        common, only_a, only_b = diff_lines(a, b)
        print(f"{img.name:<22} {len(a):5d} {len(b):5d} {common:5d} {len(only_a):9d} {len(only_b):9d}")
        totals[0] += len(a); totals[1] += len(b); totals[2] += common
        totals[3] += len(only_a); totals[4] += len(only_b)
    print("-" * 70)
    print(f"{'TOTAL':<22} {totals[0]:5d} {totals[1]:5d} {totals[2]:5d} {totals[3]:9d} {totals[4]:9d}")

    any_diff = False
    for img in IMAGES:
        a = baseline["texts"].get(img, [])
        b = candidate["texts"].get(img, [])
        _, only_a, only_b = diff_lines(a, b)
        if not only_a and not only_b:
            continue
        any_diff = True
        print(f"\n-- {img.name} --")
        if only_a:
            print(f"  only in OFF: {len(only_a)}")
            for t in only_a:
                print(f"    - {t!r}")
        if only_b:
            print(f"  only in ON:  {len(only_b)}")
            for t in only_b:
                print(f"    + {t!r}")
    if not any_diff:
        print("\n(no text-level differences across any image)")

    recall = totals[2] / totals[0] if totals[0] else 0.0
    print(f"\nLine recall (ON preserves OFF): {totals[2]}/{totals[0]} = {recall*100:.2f}%")


if __name__ == "__main__":
    main()
