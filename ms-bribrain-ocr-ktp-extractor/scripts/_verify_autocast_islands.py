#!/usr/bin/env python3
"""Compare det boxes and rec_texts between prod (_MixedPrecisionDetNet) and
autocast-fp16-with-fp32-islands across all 4 test images.

Builds the backend twice (fresh state each time) to avoid cross-contamination
and reports exact diffs.
"""
from __future__ import annotations

import gc
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402

from scripts.compare_ocr_lines import _build_fullpytorch_backend  # noqa: E402
from scripts._profile_all_images import _apply_det_autocast_islands  # noqa: E402


IMAGES = sorted((PROJECT_ROOT / "tests/data").glob("good_data*.png"))


def _extract(result):
    if not result or result[0] is None:
        return {"texts": [], "boxes": []}
    first = result[0]
    if hasattr(first, "get"):
        texts = [str(t) for t in (first.get("rec_texts") or [])]
        boxes = first.get("dt_polys") or first.get("rec_polys") or []
    else:
        texts = [str(t) for t in (first["rec_texts"] or [])]
        boxes = first.get("dt_polys") or first.get("rec_polys") or []
    boxes_np = [np.asarray(b, dtype=np.float32) for b in boxes]
    return {"texts": texts, "boxes": boxes_np}


def run(autocast: bool, images):
    print(f"\n=== Building backend (autocast_islands={autocast}) ===", flush=True)
    backend = _build_fullpytorch_backend()
    if autocast:
        _apply_det_autocast_islands(backend)
    results = {}
    for img_path in images:
        img = np.asarray(Image.open(img_path).convert("RGB"))
        backend.predict(img)  # warmup
        out = backend.predict(img)
        results[img_path.name] = _extract(out)
        r = results[img_path.name]
        print(f"  {img_path.name}: {len(r['texts'])} lines, {len(r['boxes'])} boxes")
    closer = getattr(backend, "close", None)
    if callable(closer):
        try: closer()
        except Exception: pass
    del backend
    gc.collect(); gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache(); torch.cuda.synchronize()
    try: torch._dynamo.reset()
    except Exception: pass
    return results


def main() -> int:
    if not IMAGES:
        sys.exit("No good_data images found")

    base = run(autocast=False, images=IMAGES)
    cand = run(autocast=True, images=IMAGES)

    print("\n" + "=" * 88)
    print(f"{'image':<22} {'base lines':>10} {'cand lines':>10} {'∩':>5} {'base-only':>10} {'cand-only':>10} {'box Δmax':>9}")
    print("-" * 88)
    all_match = True
    for name in base:
        a = base[name]["texts"]; b = cand[name]["texts"]
        sa, sb = set(a), set(b)
        common = len(sa & sb)
        only_a = sorted(sa - sb); only_b = sorted(sb - sa)

        # box geometry — compare up to the min count, pairwise L-inf
        boxes_a, boxes_b = base[name]["boxes"], cand[name]["boxes"]
        box_dmax = 0.0
        n = min(len(boxes_a), len(boxes_b))
        for i in range(n):
            if boxes_a[i].shape == boxes_b[i].shape:
                box_dmax = max(box_dmax, float(np.max(np.abs(boxes_a[i] - boxes_b[i]))))

        print(f"{name:<22} {len(a):>10} {len(b):>10} {common:>5} {len(only_a):>10} {len(only_b):>10} {box_dmax:>9.2f}")
        if only_a or only_b or box_dmax > 2.0:
            all_match = False

    print("\nDetailed text diffs:")
    any_diff = False
    for name in base:
        a, b = base[name]["texts"], cand[name]["texts"]
        sa, sb = set(a), set(b)
        only_a = sorted(sa - sb); only_b = sorted(sb - sa)
        if not only_a and not only_b:
            continue
        any_diff = True
        print(f"\n-- {name} --")
        if only_a:
            print(f"  only in base ({len(only_a)}):")
            for t in only_a: print(f"    - {t!r}")
        if only_b:
            print(f"  only in cand ({len(only_b)}):")
            for t in only_b: print(f"    + {t!r}")
    if not any_diff:
        print("  (no text-level differences on any image)")

    print(f"\nOverall match: {'PASS' if all_match else 'DIFF'}")
    return 0 if all_match else 1


if __name__ == "__main__":
    raise SystemExit(main())
