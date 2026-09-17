#!/usr/bin/env python3
"""Line-by-line comparison of fp32 det vs autocast+islands det.

Uses the real PaddleOCR TextDetector pipeline (DB post-process + polygons),
not a simplified connected-components extractor, so the comparison matches
what the production OCR backend actually emits.
"""

from __future__ import annotations

import copy
import os
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT
SRC_ROOT = PROJECT_ROOT / "src"
os.chdir(PROJECT_ROOT)
for _p in (PROJECT_ROOT, SRC_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from PIL import Image  # noqa: E402

from scripts.compare_ocr_lines import _build_fullpytorch_backend  # noqa: E402

IMAGES = [
    PROJECT_ROOT / "tests/data/good_data.png",
    PROJECT_ROOT / "tests/data/good_data_2.png",
    PROJECT_ROOT / "tests/data/good_data_3.png",
]

HOT_OPS = ("neck", "head")


def _install_islands(det_net, names):
    import torch
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

    for qname in names:
        parts = qname.split(".")
        parent = det_net
        for p in parts[:-1]:
            parent = getattr(parent, p) if not p.isdigit() else parent[int(p)]
        leaf = parts[-1]
        target = getattr(parent, leaf) if not leaf.isdigit() else parent[int(leaf)]
        island = Fp32Island(target)
        if leaf.isdigit():
            parent[int(leaf)] = island
        else:
            setattr(parent, leaf, island)


def _wrap_autocast(det_net):
    import torch

    original_forward = det_net.forward

    def forward(x):
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            return original_forward(x)

    det_net.forward = forward  # type: ignore[method-assign]


def _poly_to_box(poly: np.ndarray) -> tuple[int, int, int, int]:
    xs = poly[:, 0]
    ys = poly[:, 1]
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _load_rgb(p: Path) -> np.ndarray:
    return np.asarray(Image.open(p).convert("RGB"))


def main() -> int:
    import torch

    # fp32 reference backend
    ref_backend = _build_fullpytorch_backend()
    ref_backend.system.text_detector.net = ref_backend.system.text_detector.net.float().eval()

    # autocast+islands backend (separate build so weights don't alias)
    ac_backend = _build_fullpytorch_backend()
    ac_net = ac_backend.system.text_detector.net.float().eval()
    _install_islands(ac_net, HOT_OPS)
    _wrap_autocast(ac_net)

    grand = {"total_ref": 0, "total_ac": 0, "missing": 0, "extra": 0, "matched": 0, "shifted": 0}

    for img_path in IMAGES:
        img = _load_rgb(img_path)

        # warmup
        for _ in range(2):
            ref_backend._detect_boxes(img)
            ac_backend._detect_boxes(img)
            if torch.cuda.is_available():
                torch.cuda.synchronize()

        ref_boxes = ref_backend._detect_boxes(img)
        ac_boxes = ac_backend._detect_boxes(img)

        # Convert polygons to xyxy
        ref_xyxy = [_poly_to_box(np.asarray(b)) for b in ref_boxes]
        ac_xyxy = [_poly_to_box(np.asarray(b)) for b in ac_boxes]

        # Sort reading order (y, x)
        ref_sorted = sorted(ref_xyxy, key=lambda b: (b[1], b[0]))
        ac_sorted = sorted(ac_xyxy, key=lambda b: (b[1], b[0]))

        # Greedy matching: for each ref box, find closest ac box by center distance
        ref_centers = [((x1 + x2) / 2, (y1 + y2) / 2) for (x1, y1, x2, y2) in ref_sorted]
        ac_centers = [((x1 + x2) / 2, (y1 + y2) / 2) for (x1, y1, x2, y2) in ac_sorted]

        matched_ac = set()
        rows = []
        for ri, (rc, rb) in enumerate(zip(ref_centers, ref_sorted)):
            best_ai, best_d = -1, float("inf")
            for ai, ac_c in enumerate(ac_centers):
                if ai in matched_ac:
                    continue
                d = ((rc[0] - ac_c[0]) ** 2 + (rc[1] - ac_c[1]) ** 2) ** 0.5
                if d < best_d:
                    best_d, best_ai = d, ai
            if best_ai >= 0 and best_d < 20:  # within 20 px → same line
                matched_ac.add(best_ai)
                ab = ac_sorted[best_ai]
                rows.append(("MATCH", ri, rb, best_ai, ab, best_d))
            else:
                rows.append(("MISSING", ri, rb, -1, None, best_d))

        extras = [(ai, ac_sorted[ai]) for ai in range(len(ac_sorted)) if ai not in matched_ac]

        print(f"\n{'=' * 80}")
        print(f"{img_path.name}  ref={len(ref_sorted)}  ac={len(ac_sorted)}")
        print(f"{'=' * 80}")
        header = f"{'#':>3}  {'status':8}  {'ref xyxy':>24}  {'ac xyxy':>24}  {'Δxyxy':>16}  {'dist':>5}"
        print(header)
        print("-" * len(header))
        for tag, ri, rb, ai, ab, d in rows:
            rs = f"{rb[0]:4d},{rb[1]:4d},{rb[2]:4d},{rb[3]:4d}"
            if ab is None:
                print(f"{ri:3d}  {tag:8s}  {rs:>24s}  {'<dropped>':>24s}  {'—':>16s}  {d:5.1f}")
                grand["missing"] += 1
            else:
                as_ = f"{ab[0]:4d},{ab[1]:4d},{ab[2]:4d},{ab[3]:4d}"
                dx = [ab[i] - rb[i] for i in range(4)]
                delt = f"{dx[0]:+d},{dx[1]:+d},{dx[2]:+d},{dx[3]:+d}"
                print(f"{ri:3d}  {tag:8s}  {rs:>24s}  {as_:>24s}  {delt:>16s}  {d:5.1f}")
                if any(dx):
                    grand["shifted"] += 1
                else:
                    grand["matched"] += 1
        for ai, ab in extras:
            as_ = f"{ab[0]:4d},{ab[1]:4d},{ab[2]:4d},{ab[3]:4d}"
            print(f"{ai:3d}  {'EXTRA':8s}  {'<no ref>':>24s}  {as_:>24s}")
            grand["extra"] += 1

        grand["total_ref"] += len(ref_sorted)
        grand["total_ac"] += len(ac_sorted)

    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")
    print(f"total ref boxes: {grand['total_ref']}")
    print(f"total ac  boxes: {grand['total_ac']}")
    print(f"  exact-match (Δ=0)        : {grand['matched']}")
    print(f"  shifted (matched, Δ≠0)   : {grand['shifted']}")
    print(f"  missing in ac            : {grand['missing']}")
    print(f"  extra in ac (not in ref) : {grand['extra']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
