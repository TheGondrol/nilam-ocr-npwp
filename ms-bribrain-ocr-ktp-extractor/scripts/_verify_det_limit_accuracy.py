#!/usr/bin/env python3
"""Check that switching det to max/1280 preserves recognized text on the
good_data test images. Runs fullpytorch backend twice — baseline (min/64)
vs candidate (max/1280) — on each image and diffs the rec_texts.
"""
from __future__ import annotations

import gc
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402

os.environ.setdefault("AUTOKERNEL_ROOT", str(PROJECT_ROOT / "autokernel"))
os.environ.setdefault("AUTOKERNEL_PPOCR_ROOT", str(PROJECT_ROOT / "PaddleOCR2Pytorch"))
os.environ.setdefault("AUTOKERNEL_PPOCRV5_SERVER_DET_PTH", str(PROJECT_ROOT / "autokernel/workspace/ppocrv5/server_det.pth"))
os.environ.setdefault("AUTOKERNEL_PPOCRV5_SERVER_REC_PTH", str(PROJECT_ROOT / "autokernel/workspace/ppocrv5/server_rec.pth"))

from scripts._bench_det_limit import _make_settings  # noqa: E402
from src.services.ocr_backends import create_ocr_backend  # noqa: E402


IMAGES = sorted((PROJECT_ROOT / "tests/data").glob("good_data*.png"))


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


def run_config(limit_side_len: int, limit_type: str, images: list[Path]) -> dict[Path, list[str]]:
    label = f"{limit_type}/{limit_side_len}"
    print(f"\n=== Building backend for {label} ===", flush=True)
    backend = create_ocr_backend(_make_settings(limit_side_len, limit_type), use_gpu=True)
    out: dict[Path, list[str]] = {}
    for img_path in images:
        img = np.asarray(Image.open(img_path).convert("RGB"))
        # one warmup predict per image to absorb first-shape compile
        backend.predict(img)
        texts = _extract_texts(backend.predict(img))
        out[img_path] = texts
        print(f"  {img_path.name}: {len(texts)} lines", flush=True)

    closer = getattr(backend, "close", None)
    if callable(closer):
        try: closer()
        except Exception: pass
    del backend
    gc.collect(); gc.collect()
    torch.cuda.empty_cache(); torch.cuda.synchronize()
    try: torch._dynamo.reset()
    except Exception: pass
    return out


def diff_lines(a: list[str], b: list[str]) -> tuple[int, list[str], list[str]]:
    """Exact-match line count + each side's set diff (case + whitespace sensitive)."""
    sa, sb = set(a), set(b)
    common = sa & sb
    only_a = sorted(sa - sb)
    only_b = sorted(sb - sa)
    return len(common), only_a, only_b


def main() -> None:
    if not IMAGES:
        sys.exit("No good_data images found")
    print(f"Images: {[p.name for p in IMAGES]}")

    baseline = run_config(64, "min", IMAGES)
    candidate = run_config(1280, "max", IMAGES)

    print("\n" + "=" * 86)
    print(f"{'image':<22} {'base':>5} {'cand':>5} {'∩':>5} {'base-only':>9} {'cand-only':>9}")
    print("-" * 86)
    totals = [0, 0, 0, 0, 0]
    for img in IMAGES:
        a = baseline[img]
        b = candidate[img]
        common, only_a, only_b = diff_lines(a, b)
        print(f"{img.name:<22} {len(a):5d} {len(b):5d} {common:5d} {len(only_a):9d} {len(only_b):9d}")
        totals[0] += len(a); totals[1] += len(b); totals[2] += common
        totals[3] += len(only_a); totals[4] += len(only_b)
    print("-" * 86)
    print(f"{'TOTAL':<22} {totals[0]:5d} {totals[1]:5d} {totals[2]:5d} {totals[3]:9d} {totals[4]:9d}")

    # Detailed diffs
    print("\n=== Per-image diffs (exact string match) ===")
    any_diff = False
    for img in IMAGES:
        a = baseline[img]
        b = candidate[img]
        _, only_a, only_b = diff_lines(a, b)
        if not only_a and not only_b:
            continue
        any_diff = True
        print(f"\n-- {img.name} --")
        if only_a:
            print(f"  only in base (min/64): {len(only_a)}")
            for t in only_a:
                print(f"    - {t!r}")
        if only_b:
            print(f"  only in cand (max/1280): {len(only_b)}")
            for t in only_b:
                print(f"    + {t!r}")
    if not any_diff:
        print("  (no text-level differences across any image)")

    recall = totals[2] / totals[0] if totals[0] else 0.0
    print(f"\nLine recall (cand preserves base): {totals[2]}/{totals[0]} = {recall*100:.2f}%")


if __name__ == "__main__":
    main()
