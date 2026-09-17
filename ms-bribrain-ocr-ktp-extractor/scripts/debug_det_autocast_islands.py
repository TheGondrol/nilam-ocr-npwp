#!/usr/bin/env python3
"""fp16 detector via autocast + fp32 islands on the 5 hot ops.

Hot ops identified in the BN-fold experiment (all overflow fp16):
  net.neck.pan_lat_conv.0  (92,625)
  net.neck.pan_lat_conv.1  (~51k)
  net.neck.pan_lat_conv.2  (~35k)
  net.neck.pan_lat_conv.3  (~25k)
  net.head.binarize.conv1  (pre-BN 1.2M)

Strategy: keep model fp32, wrap the 5 hot modules so they force fp32
internally (and cast back to the surrounding autocast dtype), and run the
whole forward under torch.autocast(fp16). Autocast picks fp16 for every
other conv/linear while the islands stay safe.

Compares fp32 reference vs autocast+islands: prob-map diff + line extraction
on the 3 good images, and times the forward pass.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT
SRC_ROOT = PROJECT_ROOT / "src"
os.chdir(PROJECT_ROOT)
for _p in (PROJECT_ROOT, SRC_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

DEFAULT_PPOCR_ROOT = PROJECT_ROOT / "PaddleOCR2Pytorch"
DEFAULT_AUTOKERNEL_ROOT = REPO_ROOT / "autokernel"
DEFAULT_DET_PTH = DEFAULT_AUTOKERNEL_ROOT / "workspace/ppocrv5/server_det.pth"
IMAGES = [
    PROJECT_ROOT / "tests/data/good_data.png",
    PROJECT_ROOT / "tests/data/good_data_2.png",
    PROJECT_ROOT / "tests/data/good_data_3.png",
]

HOT_OPS = ("net.neck", "net.head")


def _build():
    for p in (DEFAULT_AUTOKERNEL_ROOT, DEFAULT_PPOCR_ROOT):
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)
    os.environ["AUTOKERNEL_PPOCR_ROOT"] = str(DEFAULT_PPOCR_ROOT)
    os.environ["AUTOKERNEL_PPOCRV5_SERVER_DET_PTH"] = str(DEFAULT_DET_PTH)
    import importlib
    import models.ppocrv5_server as ms
    importlib.reload(ms)
    return ms.PPOCRv5ServerDetModel()


def _preprocess(image_path: Path, limit_side_len: int = 1072) -> np.ndarray:
    import cv2
    img = cv2.imread(str(image_path))
    h, w = img.shape[:2]
    ratio = float(limit_side_len) / max(h, w) if max(h, w) > limit_side_len else 1.0
    rh = max(int(round(h * ratio / 32)) * 32, 32)
    rw = max(int(round(w * ratio / 32)) * 32, 32)
    r = cv2.resize(img, (rw, rh))
    r = cv2.cvtColor(r, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    r = (r - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
    return r.transpose(2, 0, 1)[None, ...]


def _install_islands(model, names):
    import torch
    import torch.nn as nn

    class Fp32Island(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, x, *args, **kwargs):
            import torch
            def first_tensor(v):
                if isinstance(v, torch.Tensor):
                    return v
                if isinstance(v, (list, tuple)):
                    for t in v:
                        r = first_tensor(t)
                        if r is not None:
                            return r
                return None
            out_dtype = torch.float32  # island always emits fp32
            def to_fp32(v):
                if isinstance(v, torch.Tensor):
                    return v.float()
                if isinstance(v, list):
                    return [to_fp32(t) for t in v]
                if isinstance(v, tuple):
                    return tuple(to_fp32(t) for t in v)
                return v
            def to_out(v):
                if isinstance(v, torch.Tensor):
                    return v.to(out_dtype)
                if isinstance(v, list):
                    return [to_out(t) for t in v]
                if isinstance(v, tuple):
                    return tuple(to_out(t) for t in v)
                return v
            with torch.autocast(device_type="cuda", enabled=False):
                y = self.inner(to_fp32(x), *to_fp32(args), **kwargs)
            return to_out(y)

    # Locate each named module and replace in its parent
    wrapped = []
    for qname in names:
        parts = qname.split(".")
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p) if not p.isdigit() else parent[int(p)]
        leaf = parts[-1]
        target = getattr(parent, leaf) if not leaf.isdigit() else parent[int(leaf)]
        island = Fp32Island(target)
        if leaf.isdigit():
            parent[int(leaf)] = island
        else:
            setattr(parent, leaf, island)
        wrapped.append(qname)
    return wrapped


def _extract_boxes(prob_map: np.ndarray, thresh=0.3, box_thresh=0.5):
    import cv2
    mask = (prob_map > thresh).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=4)
    boxes = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < 4:
            continue
        region = prob_map[y:y+h, x:x+w]
        score = float(region[labels[y:y+h, x:x+w] == i].mean())
        if score < box_thresh:
            continue
        boxes.append((x, y, w, h, score))
    boxes.sort(key=lambda b: (b[1], b[0]))
    return boxes


def _time(fn, n=6, warmup=2):
    import torch
    ts = []
    for i in range(n):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1000
        if i >= warmup:
            ts.append(dt)
    return out, ts


def _trace_first_nan(model, x):
    """Run model under autocast with hooks, return first (name, max_abs, nan, inf)."""
    import torch
    events = []

    def mk(name):
        def hook(mod, inp, out):
            if isinstance(out, torch.Tensor):
                f = out.detach()
                nan = torch.isnan(f).any().item()
                inf = torch.isinf(f).any().item()
                finite = f[torch.isfinite(f)]
                mx = float(finite.abs().max().item()) if finite.numel() else float("inf")
                events.append((name, mx, bool(nan), bool(inf), str(out.dtype)))
        return hook

    handles = []
    for name, m in model.named_modules():
        if name == "":
            continue
        handles.append(m.register_forward_hook(mk(name)))
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
        model(x)
    for h in handles:
        h.remove()
    first = None
    for e in events:
        if e[2] or e[3]:
            first = e
            break
    top = sorted([e for e in events if not e[2] and not e[3]], key=lambda e: -e[1])[:10]
    return first, top


def main() -> int:
    import torch
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", action="store_true")
    args = ap.parse_args()

    ref = _build().cuda().float().eval()
    mdl = _build().cuda().float().eval()
    wrapped = _install_islands(mdl, HOT_OPS)
    print(f"Wrapped {len(wrapped)} fp32 islands: {wrapped}")

    if args.trace:
        x = torch.from_numpy(_preprocess(IMAGES[0])).cuda().float()
        first, top = _trace_first_nan(mdl, x)
        print(f"\nfirst NaN/Inf: {first}")
        print("top-10 |max| finite (non-NaN) under autocast:")
        for e in top:
            print(f"  {e[0]:55s}  max={e[1]:12.1f}  dtype={e[4]}")
        return 0

    report_lines = []
    for img in IMAGES:
        np_in = _preprocess(img)
        x = torch.from_numpy(np_in).cuda().float()

        with torch.inference_mode():
            ref_out, ref_ts = _time(lambda: ref(x), n=8, warmup=3)

            def run_ac():
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    return mdl(x)

            ac_out, ac_ts = _time(run_ac, n=8, warmup=3)

        ref_maps = ref_out["maps"] if isinstance(ref_out, dict) else ref_out
        ac_maps = ac_out["maps"] if isinstance(ac_out, dict) else ac_out
        ref_maps = ref_maps.float().cpu()
        ac_maps = ac_maps.float().cpu()

        diff = (ref_maps - ac_maps).abs()
        p_ref = ref_maps[0, 0].numpy()
        p_ac = ac_maps[0, 0].numpy()
        b_ref = _extract_boxes(p_ref)
        b_ac = _extract_boxes(p_ac)

        ref_med = sorted(ref_ts)[len(ref_ts)//2] if ref_ts else 0.0
        ac_med = sorted(ac_ts)[len(ac_ts)//2] if ac_ts else 0.0
        line = (
            f"\n{img.name}  shape={list(x.shape)}\n"
            f"  fp32 ref  median={ref_med:.1f} ms  timings={[f'{t:.1f}' for t in ref_ts]}\n"
            f"  autocast  median={ac_med:.1f} ms  timings={[f'{t:.1f}' for t in ac_ts]}\n"
            f"  prob-map diff: max={diff.max().item():.4e}  mean={diff.mean().item():.4e}\n"
            f"  boxes: ref={len(b_ref)} ac={len(b_ac)}"
        )
        print(line)
        report_lines.append(line)

        n_common = min(len(b_ref), len(b_ac))
        diffs = []
        for i in range(n_common):
            r = b_ref[i]; a = b_ac[i]
            diffs.append(abs(r[0]-a[0]) + abs(r[1]-a[1]) + abs(r[2]-a[2]) + abs(r[3]-a[3]))
        if diffs:
            print(f"  avg box-xywh L1 diff (first {n_common}): {sum(diffs)/len(diffs):.2f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
