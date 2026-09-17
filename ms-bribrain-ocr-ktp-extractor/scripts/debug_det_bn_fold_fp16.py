#!/usr/bin/env python3
"""BN-folding weight rescaling to unlock full-fp16 detector.

Idea: for each Conv→BN pair, fold a scale factor alpha into the conv
(divide conv.weight and conv.bias by alpha; divide bn.running_mean by
alpha and bn.running_var by alpha**2). Post-BN output is mathematically
identical (up to BN's epsilon, which is negligible when var >> eps/alpha**2),
but the pre-BN activations shrink into fp16's safe range.

Pipeline:
  1) Build fp32 reference detector. Calibrate pre-BN activation max on the
     3 good images.
  2) Build naive fp16 detector (baseline — known-broken).
  3) Build another detector, apply BN-fold rescale using fp32 stats, then
     convert to fp16.
  4) Run all three on each image. Report prob-map drift vs fp32, and
     extract DB polygons (text-line boxes) to compare line-by-line counts.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Tuple

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

# Per-channel pre-BN activation target. Fp16 max is 65504; keep 30x headroom.
TARGET_PRE_BN_MAX = 2000.0
# Only rescale when the fp32 measured max is above this (avoids tiny-activation
# denominators and pointless rescales).
MIN_RESCALE_ACTIVATION = 100.0


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _build(ak_root: Path, ppocr_root: Path, det_pth: Path):
    for p in (ak_root, ppocr_root):
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)
    os.environ["AUTOKERNEL_PPOCR_ROOT"] = str(ppocr_root)
    os.environ["AUTOKERNEL_PPOCRV5_SERVER_DET_PTH"] = str(det_pth)
    import importlib
    import models.ppocrv5_server as m
    importlib.reload(m)
    return m.PPOCRv5ServerDetModel()


def _preprocess(image_path: Path, limit_side_len: int = 1072) -> np.ndarray:
    import cv2
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(image_path)
    h, w = img.shape[:2]
    ratio = float(limit_side_len) / max(h, w) if max(h, w) > limit_side_len else 1.0
    rh = max(int(round(h * ratio / 32)) * 32, 32)
    rw = max(int(round(w * ratio / 32)) * 32, 32)
    r = cv2.resize(img, (rw, rh))
    r = cv2.cvtColor(r, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    r = (r - np.array([0.485, 0.456, 0.406], np.float32)) / np.array(
        [0.229, 0.224, 0.225], np.float32
    )
    return r.transpose(2, 0, 1)[None, ...]


# ---------------------------------------------------------------------------
# BN folding: discover Conv→BN pairs and rescale in place
# ---------------------------------------------------------------------------

def _find_conv_bn_pairs(model) -> List[Tuple[str, "nn.Conv2d", "nn.BatchNorm2d"]]:
    """Return list of (qualified_name, conv, bn) where bn is applied directly
    to conv's output.

    Strategy: for every nn.BatchNorm2d, walk parent modules to find a sibling
    nn.Conv2d that the BN's parent module consumes as input to the BN. We
    handle two common cases:

      (a) parent has attributes ``.conv`` (Conv2d) and ``.bn`` (BatchNorm2d).
          This matches ConvBNAct and many custom blocks.
      (b) parent has both ``.conv1`` + ``.conv_bn1`` style (PFHeadLocal head).
    """
    import torch.nn as nn

    pairs: list[tuple[str, nn.Conv2d, nn.BatchNorm2d]] = []
    seen_bns: set[int] = set()
    for name, parent in model.named_modules():
        # (a) .conv + .bn
        conv = getattr(parent, "conv", None)
        bn = getattr(parent, "bn", None)
        if isinstance(conv, nn.Conv2d) and isinstance(bn, nn.BatchNorm2d):
            if id(bn) not in seen_bns:
                pairs.append((f"{name}.conv→bn", conv, bn))
                seen_bns.add(id(bn))
            continue

        # (b) .conv1/.conv_bn1 and .conv2/.conv_bn2 (PFHeadLocal DBHead)
        for ci, bi in (("conv1", "conv_bn1"), ("conv2", "conv_bn2")):
            conv = getattr(parent, ci, None)
            bn = getattr(parent, bi, None)
            if isinstance(conv, nn.Conv2d) and isinstance(bn, nn.BatchNorm2d):
                if id(bn) not in seen_bns:
                    pairs.append((f"{name}.{ci}→{bi}", conv, bn))
                    seen_bns.add(id(bn))
    return pairs


def calibrate_pre_bn_max(model, inputs, pairs) -> dict:
    """Run fp32 forwards, record per-pair max|conv_output| across all inputs
    and spatial locations. One global scalar per pair (not per-channel) — the
    BN then handles per-channel normalization.
    """
    import torch

    # Hook conv outputs (== pre-BN activation)
    stats: dict[int, float] = {id(c): 0.0 for _, c, _ in pairs}
    handles = []

    def mk_hook(cid):
        def hook(_m, _i, out):
            m = float(out.detach().abs().max().item())
            if m > stats[cid]:
                stats[cid] = m
        return hook

    for _, conv, _ in pairs:
        handles.append(conv.register_forward_hook(mk_hook(id(conv))))

    with torch.inference_mode():
        for x in inputs:
            model(x)

    for h in handles:
        h.remove()

    return {id(c): stats[id(c)] for _, c, _ in pairs}


def apply_bn_fold_rescale(pairs, cal_stats: dict, target: float, min_act: float) -> list:
    """For each (conv, bn) with measured pre-BN max > min_act, divide conv
    weights (and bias if present) by alpha = measured_max / target, and divide
    bn.running_mean by alpha and bn.running_var by alpha**2. This keeps
    post-BN output algebraically identical (up to eps), but shrinks pre-BN
    values by factor alpha.
    """
    import torch

    applied = []
    for name, conv, bn in pairs:
        m = cal_stats[id(conv)]
        if m <= min_act:
            applied.append((name, m, 1.0, "skipped (below min)"))
            continue
        alpha = m / target  # > 1 when rescale needed
        if alpha <= 1.0:
            applied.append((name, m, 1.0, "skipped (already in range)"))
            continue
        with torch.no_grad():
            conv.weight.data.div_(alpha)
            if conv.bias is not None:
                conv.bias.data.div_(alpha)
            bn.running_mean.data.div_(alpha)
            bn.running_var.data.div_(alpha * alpha)
            # Scale eps to keep sqrt(var/α² + eps') == sqrt(var+eps)/α exactly.
            bn.eps = bn.eps / (alpha * alpha)
        applied.append((name, m, alpha, "applied"))
    return applied


# ---------------------------------------------------------------------------
# DB post-processing: sigmoid-threshold → boxes (count lines per image)
# ---------------------------------------------------------------------------

def _extract_line_boxes(prob_map: np.ndarray, thresh: float = 0.3,
                         box_thresh: float = 0.5, min_area: int = 10) -> list:
    """Turn a DB probability map into a list of bounding rects.
    One rect per connected component passing box_thresh on its region mean.
    Simple, independent of Paddle's postprocess (which we don't strictly need
    for a line-count comparison).
    """
    import cv2
    m = (prob_map >= thresh).astype(np.uint8) * 255
    n_comp, labels, stats_, centroids = cv2.connectedComponentsWithStats(m, 8)
    boxes = []
    for i in range(1, n_comp):
        x, y, w, h, area = stats_[i]
        if area < min_area:
            continue
        region = prob_map[y:y + h, x:x + w]
        mask = labels[y:y + h, x:x + w] == i
        if mask.sum() == 0:
            continue
        mean_score = float(region[mask].mean())
        if mean_score < box_thresh:
            continue
        boxes.append((int(x), int(y), int(w), int(h), mean_score))
    # sort top-to-bottom, left-to-right for stable line ordering
    boxes.sort(key=lambda b: (b[1], b[0]))
    return boxes


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> int:
    import torch

    assert torch.cuda.is_available(), "CUDA required"
    ak = DEFAULT_AUTOKERNEL_ROOT.resolve()
    pp = DEFAULT_PPOCR_ROOT.resolve()
    pth = DEFAULT_DET_PTH.resolve()

    # Pre-process each image; we run them independently (variable shapes).
    preproc = []
    for img in IMAGES:
        x = torch.from_numpy(_preprocess(img)).cuda().float()
        preproc.append((img.name, x))
        print(f"  {img.name}: {list(x.shape)}")

    # ---- (a) FP32 reference ------------------------------------------------
    print("\n[1/3] FP32 reference...")
    m32 = _build(ak, pp, pth).cuda().float().eval()
    ref_prob_maps = {}
    with torch.inference_mode():
        for name, x in preproc:
            out = m32(x)
            ref_prob_maps[name] = out[:, 0].float().cpu().numpy()[0]
            print(f"  {name}: prob max={ref_prob_maps[name].max():.3f} "
                  f"mean={ref_prob_maps[name].mean():.4f} "
                  f"frac>0.3={(ref_prob_maps[name] > 0.3).mean():.4f}")

    # Calibrate pre-BN stats on fp32 model (shared across all 3 images)
    pairs = _find_conv_bn_pairs(m32)
    print(f"\n  Found {len(pairs)} Conv→BN pairs")
    cal = calibrate_pre_bn_max(m32, [x for _, x in preproc], pairs)

    pre_bn_maxes = sorted(cal.values(), reverse=True)
    print(f"  Pre-BN max stats over all pairs:")
    print(f"    top-5: {[f'{v:.1f}' for v in pre_bn_maxes[:5]]}")
    print(f"    p95={np.percentile(pre_bn_maxes, 95):.1f}  "
          f"p50={np.percentile(pre_bn_maxes, 50):.1f}")
    over_fp16 = sum(1 for v in pre_bn_maxes if v > 65504.0)
    over_10k = sum(1 for v in pre_bn_maxes if v > 10000.0)
    print(f"    pairs > fp16 max (65504): {over_fp16}")
    print(f"    pairs > 10000          : {over_10k}")

    # Also measure ALL convs (including bare ones in neck) in fp32 to find
    # additional overflow candidates beyond Conv→BN pairs.
    print("\n  Scanning ALL Conv2d / ConvTranspose2d outputs in fp32...")
    import torch.nn as nn
    all_conv_stats: dict[str, float] = {}
    handles = []
    def _mk(nm):
        def h(_m, _i, out):
            v = float(out.detach().abs().max().item())
            all_conv_stats[nm] = max(all_conv_stats.get(nm, 0.0), v)
        return h
    for nm, mod in m32.named_modules():
        if isinstance(mod, (nn.Conv2d, nn.ConvTranspose2d)):
            handles.append(mod.register_forward_hook(_mk(nm)))
    with torch.inference_mode():
        for _, x in preproc:
            m32(x)
    for h in handles: h.remove()
    top_convs = sorted(all_conv_stats.items(), key=lambda kv: kv[1], reverse=True)[:15]
    print(f"  Top-15 abs-max conv outputs (any kind):")
    for nm, v in top_convs:
        flag = "OVERFLOW" if v > 65504 else ("HOT" if v > 1000 else "")
        print(f"    {nm[:68]:68s} {v:12.1f}  {flag}")

    # ---- (b) Naive FP16 (baseline) — trace first NaN/Inf -------------------
    print("\n[2/3] Naive FP16 baseline (no BN fold)...")
    m16 = _build(ak, pp, pth).cuda().half().eval()
    # Hook every module to find first NaN/Inf emergence.
    first_bad = {}
    handles = []
    def _mk_bad(nm):
        def h(_m, _i, out):
            if isinstance(out, torch.Tensor):
                nan = bool(torch.isnan(out).any().item())
                inf = bool(torch.isinf(out).any().item())
                if (nan or inf) and nm not in first_bad:
                    first_bad[nm] = (nan, inf,
                                      float(out.detach().abs().float().max().item()))
        return h
    for nm, mod in m16.named_modules():
        if nm == "":
            continue
        handles.append(mod.register_forward_hook(_mk_bad(nm)))
    naive_prob_maps = {}
    with torch.inference_mode():
        for name, x in preproc:
            first_bad.clear()
            out = m16(x.half())
            naive_prob_maps[name] = out[:, 0].float().cpu().numpy()[0]
            # Report first 5 bad layers per image
            bad_items = list(first_bad.items())[:8]
            print(f"  {name}: first bad layers in fp16:")
            for nm, (nan, inf, m) in bad_items:
                tag = "NaN" if nan else "Inf"
                print(f"    {nm[:68]:68s} {tag} (abs_max={m:.3e})")
    for h in handles: h.remove()

    # ---- (c) FP16 + BN fold rescale (head only) + selective fp32 shield ---
    # We tried an exact cascaded rescale through the LKPAN→incl residual; the
    # math is algebraically correct (verified in fp32 to 4e-5), but in fp16
    # the cascade produces ~20% relative error because the tiny α-scaled
    # rescaled weights combined with the α-amplified compensation at the head
    # eat too much fp16 precision. So we keep the simple BN fold for the
    # 1.2M overflow at head.binarize.conv1, and keep the 4 bare `pan_lat_conv`
    # layers in fp32 (their outputs are what actually overflow fp16; they run
    # in fp32, then we cast back to fp16 for the downstream pipeline).
    print("\n[3/3] FP16 + BN-fold (head) + fp32-island (pan_lat_conv)...")
    m_fold = _build(ak, pp, pth).cuda().float().eval()
    fold_pairs = _find_conv_bn_pairs(m_fold)
    # Use the cal we already have — but it's keyed by id of m32's convs.
    # Recalibrate on m_fold's own convs for correctness.
    fold_cal = calibrate_pre_bn_max(m_fold, [x for _, x in preproc], fold_pairs)
    applied = apply_bn_fold_rescale(
        fold_pairs, fold_cal, target=TARGET_PRE_BN_MAX, min_act=MIN_RESCALE_ACTIVATION
    )

    # Replace each pan_lat_conv with an fp32-island wrapper:
    # the conv runs its weights in fp32 while the rest of the neck stays fp16.
    class Fp32Island(nn.Module):
        def __init__(self, inner: nn.Module):
            super().__init__()
            self.inner = inner
        def forward(self, x):
            return self.inner(x.float()).to(x.dtype)
    for idx in range(4):
        orig = m_fold.net.neck.pan_lat_conv[idx]
        m_fold.net.neck.pan_lat_conv[idx] = Fp32Island(orig)
    # pan_lat_conv outputs exceed fp16 max (92k > 65k), so casting back to
    # fp16 at the island boundary would still overflow. Keep the downstream
    # consumer (the IntraCLBlock residual) in fp32 too.
    m_fold.net.neck.incl1 = Fp32Island(m_fold.net.neck.incl1)
    m_fold.net.neck.incl2 = Fp32Island(m_fold.net.neck.incl2)
    m_fold.net.neck.incl3 = Fp32Island(m_fold.net.neck.incl3)
    m_fold.net.neck.incl4 = Fp32Island(m_fold.net.neck.incl4)

    cascade_applied = [("pan_lat_conv[0..3] kept in fp32", 0.0, 1.0, "fp32-island")]

    # Keep the code below as dead branch (disabled) — it's the exact-cascade
    # experiment and is documented in the script header. Re-enable by flipping
    # USE_EXACT_CASCADE to True; it preserves fp32 correctness but still
    # suffers ~20% fp16 error.
    USE_EXACT_CASCADE = False
    if not USE_EXACT_CASCADE:
        pass
    # Cascade rescale: pan_lat_conv[i] → incl_{i+1} residual → concat →
    # head.binarize.conv1. The only branchy op in between is the IntraCLBlock
    # residual (x + x_relation). We preserve mathematical equivalence by:
    #   (1) pan_lat_conv[i].weight /= α_i   (bias=False, nothing else to scale)
    #   (2) inside incl_{i+1}: divide every conv bias by α_i (weights are linear
    #       in x so they don't need scaling); divide bn.running_mean by α_i
    #       and bn.running_var by α_i²; divide bn.weight (γ) and bn.bias (β)
    #       by α_i. This makes x_relation = orig_x_relation / α_i so that the
    #       residual sum = (x + x_relation)/α_i is uniformly scaled.
    #   (3) head.binarize.conv1.weight has its input-channel slice for p_i
    #       multiplied by α_i to exactly cancel the scaling. (The head conv
    #       already has its BN-fold factor applied; the two scale ops commute
    #       since they act on disjoint dimensions of the weight.)
    #
    # p5 = pan_lat_conv[3] → incl4,  p4 = pan_lat_conv[2] → incl3,
    # p3 = pan_lat_conv[1] → incl2,  p2 = pan_lat_conv[0] → incl1.
    # fuse = cat([p5, p4, p3, p2], dim=1), so head.binarize.conv1 in-channels
    # 0:C = p5, C:2C = p4, 2C:3C = p3, 3C:4C = p2, where C = out_channels//4.
    LKPAN_TARGET = 2000.0
    if False:  # disabled exact-cascade path
        cascade_applied: list = []
    neck = m_fold.net.neck
    head_conv1 = m_fold.net.head.binarize.conv1
    bn_head = m_fold.net.head.binarize.conv_bn1

    out_c = head_conv1.in_channels
    C = out_c // 4
    pan_to_incl = {
        3: (neck.incl4, slice(0, C)),       # p5
        2: (neck.incl3, slice(C, 2 * C)),   # p4
        1: (neck.incl2, slice(2 * C, 3 * C)),  # p3
        0: (neck.incl1, slice(3 * C, 4 * C)),  # p2
    }

    # Measure pan_lat_conv outputs in fp32 for calibration
    pan_lat_max: dict[int, float] = {}

    for idx, (incl, ch_slice) in pan_to_incl.items():
        if True:  # cascade disabled
            continue
        m = pan_lat_max.get(idx, 0.0)
        if m <= LKPAN_TARGET:
            cascade_applied.append((f"pan_lat_conv[{idx}]", m, 1.0, "below target"))
            continue
        alpha = m / LKPAN_TARGET
        with torch.no_grad():
            plc = neck.pan_lat_conv[idx]
            plc.weight.data.div_(alpha)
            # plc is bias=False by construction, but guard anyway
            if plc.bias is not None:
                plc.bias.data.div_(alpha)
            # Inside incl: scale all conv biases by 1/α
            for sub in incl.modules():
                if isinstance(sub, nn.Conv2d) and sub.bias is not None:
                    sub.bias.data.div_(alpha)
            # Scale incl.bn so its output is 1/α of original (exact when eps
            # is scaled so sqrt(var/α² + eps') == sqrt(var+eps)/α).
            incl.bn.running_mean.data.div_(alpha)
            incl.bn.running_var.data.div_(alpha * alpha)
            incl.bn.eps = incl.bn.eps / (alpha * alpha)
            incl.bn.weight.data.div_(alpha)
            incl.bn.bias.data.div_(alpha)
            # Compensate at head.binarize.conv1: multiply input-channel slice
            head_conv1.weight.data[:, ch_slice, :, :].mul_(alpha)
        cascade_applied.append(
            (f"pan_lat_conv[{idx}] → incl/{['incl1','incl2','incl3','incl4'][idx]} → head ch[{ch_slice.start}:{ch_slice.stop}]",
             m, alpha, "cascaded")
        )

    print(f"\n  LKPAN cascade rescales:")
    print(f"  {'chain':70s} {'pre max':>10s} {'alpha':>8s}")
    for name, m, alpha, status in cascade_applied:
        print(f"  {name[:70]:70s} {m:10.1f} {alpha:8.2f}  {status}")
    n_applied = sum(1 for *_, s in applied if s == "applied")
    print(f"  Rescaled {n_applied}/{len(applied)} pairs (target pre-BN max={TARGET_PRE_BN_MAX})")

    # Show top-10 rescales by alpha
    applied_sorted = sorted(applied, key=lambda r: r[2], reverse=True)
    print(f"\n  Top-10 rescales (largest alpha):")
    print(f"  {'pair':60s} {'pre-BN max':>12s} {'alpha':>8s}")
    for row in applied_sorted[:10]:
        name, m, alpha, status = row
        print(f"  {name[:60]:60s} {m:12.1f} {alpha:8.2f}  {status}")

    # Sanity re-check: compute new pre-BN max on fp32 rescaled model
    recal = calibrate_pre_bn_max(m_fold, [x for _, x in preproc], fold_pairs)
    new_top = sorted(recal.values(), reverse=True)
    print(f"\n  Post-rescale pre-BN max top-5: {[f'{v:.1f}' for v in new_top[:5]]}")
    print(f"    (all should be <= ~{TARGET_PRE_BN_MAX} for rescaled pairs; "
          f"un-rescaled pairs keep their original value)")

    # FIRST: verify math is correct in fp32 (should be ~1e-5 diff vs reference)
    print("\n  Sanity check: rescaled model in FP32 should match ref exactly...")
    with torch.inference_mode():
        for name, x in preproc:
            out = m_fold(x)
            fold32 = out[:, 0].float().cpu().numpy()[0]
            diff = np.abs(fold32 - ref_prob_maps[name])
            print(f"    {name}: fp32-rescaled vs fp32-ref  max={diff.max():.3e} "
                  f"mean={diff.mean():.3e}")

    # Convert to fp16 and run, but KEEP fp32-island inner convs in fp32
    m_fold.half().eval()
    for idx in range(4):
        sub = m_fold.net.neck.pan_lat_conv[idx]
        if isinstance(sub, Fp32Island):
            sub.inner.float()
    for mod in [m_fold.net.neck.incl1, m_fold.net.neck.incl2,
                m_fold.net.neck.incl3, m_fold.net.neck.incl4]:
        if isinstance(mod, Fp32Island):
            mod.inner.float()
    # First trace NaN/Inf to find new trouble layers in the rescaled fp16 model
    print("\n  Tracing NaN/Inf in fp16+fold model...")
    fold_bad = {}
    h2 = []
    def _mkf(nm):
        def h(_m, _i, out):
            if isinstance(out, torch.Tensor):
                if (torch.isnan(out).any() or torch.isinf(out).any()) and nm not in fold_bad:
                    fold_bad[nm] = (bool(torch.isnan(out).any()),
                                     bool(torch.isinf(out).any()),
                                     float(out.detach().abs().float().max().item()))
        return h
    for nm, mod in m_fold.named_modules():
        if nm == "":
            continue
        h2.append(mod.register_forward_hook(_mkf(nm)))
    # Also scan BN running_var for fp16 underflow
    print("  BN running_var ranges after rescale (in fp16 model):")
    for nm, mod in m_fold.named_modules():
        if isinstance(mod, nn.BatchNorm2d):
            rv = mod.running_var.float().cpu()
            mn = float(rv.min().item())
            mx = float(rv.max().item())
            if mn < 1e-4 or mx < 1e-4:
                print(f"    {nm[:62]:62s}  var range=[{mn:.2e},{mx:.2e}] eps={mod.eps:.2e}")
    fold_prob_maps = {}
    with torch.inference_mode():
        for name, x in preproc:
            fold_bad.clear()
            out = m_fold(x.half())
            fold_prob_maps[name] = out[:, 0].float().cpu().numpy()[0]
            bad_items = list(fold_bad.items())[:8]
            if bad_items:
                print(f"  {name}: first NaN/Inf layers in fp16+fold:")
                for nm, (nan, inf, m) in bad_items:
                    tag = "NaN" if nan else "Inf"
                    print(f"    {nm[:68]:68s} {tag} (abs_max={m:.3e})")
            else:
                print(f"  {name}: no NaN/Inf — pure precision drift")
    for h in h2: h.remove()

    # Per-layer drift diagnosis: compare fold fp32 vs fold fp16 layer-by-layer
    # Rebuild fold fp32 clone with same transformations
    print("\n  Per-layer drift: fold fp32 vs fold fp16 (first bad image only)")
    store32, store16 = {}, {}
    _, first_x = preproc[0]
    # fold fp32 reference (rebuild, reapply same transforms)
    m_fold32 = _build(ak, pp, pth).cuda().float().eval()
    fp2 = _find_conv_bn_pairs(m_fold32)
    cal2 = calibrate_pre_bn_max(m_fold32, [first_x], fp2)
    apply_bn_fold_rescale(fp2, cal2, TARGET_PRE_BN_MAX, MIN_RESCALE_ACTIVATION)
    # Reapply cascade identically
    neck2 = m_fold32.net.neck
    head2 = m_fold32.net.head.binarize.conv1
    plc_max2 = {}
    _h2 = []
    for idx in range(4):
        def mk(idx):
            def _h(_m, _i, out):
                v = float(out.detach().abs().max().item())
                if v > plc_max2.get(idx, 0.0): plc_max2[idx] = v
            return _h
        _h2.append(neck2.pan_lat_conv[idx].register_forward_hook(mk(idx)))
    with torch.inference_mode(): m_fold32(first_x)
    for h in _h2: h.remove()
    pan_to_incl2 = {3: (neck2.incl4, slice(0, C)), 2: (neck2.incl3, slice(C, 2*C)),
                     1: (neck2.incl2, slice(2*C, 3*C)), 0: (neck2.incl1, slice(3*C, 4*C))}
    for idx, (incl, chs) in pan_to_incl2.items():
        m = plc_max2.get(idx, 0.0)
        if m <= LKPAN_TARGET: continue
        a = m / LKPAN_TARGET
        with torch.no_grad():
            neck2.pan_lat_conv[idx].weight.data.div_(a)
            for sub in incl.modules():
                if isinstance(sub, nn.Conv2d) and sub.bias is not None:
                    sub.bias.data.div_(a)
            incl.bn.running_mean.data.div_(a)
            incl.bn.running_var.data.div_(a * a)
            incl.bn.eps = incl.bn.eps / (a * a)
            incl.bn.weight.data.div_(a)
            incl.bn.bias.data.div_(a)
            head2.weight.data[:, chs, :, :].mul_(a)
    handles32, handles16 = [], []
    def _mk32(nm):
        def h(_m, _i, out):
            if isinstance(out, torch.Tensor):
                store32[nm] = out.detach().float().cpu()
        return h
    def _mk16(nm):
        def h(_m, _i, out):
            if isinstance(out, torch.Tensor):
                store16[nm] = out.detach().float().cpu()
        return h
    for nm, mod in m_fold32.named_modules():
        if nm: handles32.append(mod.register_forward_hook(_mk32(nm)))
    for nm, mod in m_fold.named_modules():
        if nm: handles16.append(mod.register_forward_hook(_mk16(nm)))
    with torch.inference_mode():
        m_fold32(first_x)
        m_fold(first_x.half())
    for h in handles32 + handles16: h.remove()
    # Compare & rank
    rows = []
    for nm in sorted(set(store32) & set(store16)):
        a, b = store32[nm], store16[nm]
        if a.shape != b.shape: continue
        d = (a - b).abs()
        dmax = float(d.max().item()) if d.numel() else 0.0
        ref_max = float(a.abs().max().item()) if a.numel() else 0.0
        rel = dmax / ref_max if ref_max > 0 else 0.0
        rows.append((nm, dmax, rel, ref_max))
    # Find first layer where rel > 50% — where fp16 diverges catastrophically
    rows_in_order = rows  # already sorted by name, but we want by forward order
    # Use module order
    order = {nm: i for i, (nm, _) in enumerate(m_fold.named_modules()) if nm}
    rows_in_order = sorted(rows, key=lambda r: order.get(r[0], 10**9))
    catastrophic = [r for r in rows_in_order if r[2] > 0.5]
    print(f"  First 8 layers where fp16 rel-diff > 50%:")
    for nm, dmax, rel, refm in catastrophic[:8]:
        print(f"    {nm[:66]:66s} abs={dmax:.2e} rel={rel:5.1%} ref_max={refm:.2e}")
    rows.sort(key=lambda r: r[1], reverse=True)
    print(f"\n  Top-10 layers by absolute fp16↔fp32 diff:")
    for nm, dmax, rel, refm in rows[:10]:
        print(f"    {nm[:66]:66s} abs={dmax:.2e} rel={rel:5.1%} ref_max={refm:.2e}")

    # ---- Compare -----------------------------------------------------------
    print("\n" + "=" * 78)
    print("  Prob-map drift vs FP32 reference")
    print("=" * 78)
    print(f"  {'image':20s} {'baseline fp16 diff':>28s} {'fp16+BN-fold diff':>28s}")
    print(f"  {'':20s} {'max / mean':>28s} {'max / mean':>28s}")
    for name, _ in preproc:
        a = ref_prob_maps[name]
        b = naive_prob_maps[name]
        c = fold_prob_maps[name]
        d_naive = np.abs(a - b)
        d_fold = np.abs(a - c)
        print(f"  {name:20s} "
              f"{d_naive.max():.3e} / {d_naive.mean():.3e}    "
              f"{d_fold.max():.3e} / {d_fold.mean():.3e}")

    # ---- Line-count comparison from prob maps -----------------------------
    print("\n" + "=" * 78)
    print("  Detector line-box count per image (prob>=0.3 connected components)")
    print("=" * 78)
    print(f"  {'image':20s} {'fp32':>6s} {'naive fp16':>11s} {'fp16+fold':>10s}")
    line_reports = {}
    for name, _ in preproc:
        b32 = _extract_line_boxes(ref_prob_maps[name])
        b16 = _extract_line_boxes(naive_prob_maps[name])
        bfo = _extract_line_boxes(fold_prob_maps[name])
        line_reports[name] = (b32, b16, bfo)
        print(f"  {name:20s} {len(b32):6d} {len(b16):11d} {len(bfo):10d}")

    # ---- Line-by-line alignment (fp32 as ref) -----------------------------
    print("\n" + "=" * 78)
    print("  Line-by-line alignment: FP32 ref vs FP16+BN-fold")
    print("=" * 78)

    def _center(b):
        return (b[0] + b[2] / 2.0, b[1] + b[3] / 2.0)

    for name, _ in preproc:
        b32, _, bfo = line_reports[name]
        print(f"\n  Image: {name}")
        print(f"  {'#':>3}  {'fp32 box (x,y,w,h,score)':<38s}  "
              f"{'fp16+fold box':<38s}  match")
        used = set()
        exact = 0
        for i, rb in enumerate(b32):
            rc = _center(rb)
            best_j, best_d = None, 1e18
            for j, ob in enumerate(bfo):
                if j in used:
                    continue
                oc = _center(ob)
                d = (rc[0] - oc[0]) ** 2 + (rc[1] - oc[1]) ** 2
                if d < best_d:
                    best_d, best_j = d, j
            tol_px = 20.0
            if best_j is not None and best_d <= tol_px * tol_px:
                used.add(best_j)
                ob = bfo[best_j]
                ok = (abs(rb[0] - ob[0]) <= 2 and abs(rb[1] - ob[1]) <= 2
                      and abs(rb[2] - ob[2]) <= 4 and abs(rb[3] - ob[3]) <= 4)
                mark = "✓" if ok else "~"
                if ok:
                    exact += 1
                print(f"  {i:3d}  "
                      f"{str(rb):<38s}  {str(ob):<38s}  {mark}")
            else:
                print(f"  {i:3d}  {str(rb):<38s}  {'<no-match>':<38s}  ✗")
        orphans = [j for j in range(len(bfo)) if j not in used]
        print(f"    exact matches: {exact}/{len(b32)}  "
              f"orphans in fold: {len(orphans)}")

    # ---- Final verdict -----------------------------------------------------
    naive_max_err = max(
        np.abs(ref_prob_maps[n] - naive_prob_maps[n]).max() for n, _ in preproc
    )
    fold_max_err = max(
        np.abs(ref_prob_maps[n] - fold_prob_maps[n]).max() for n, _ in preproc
    )
    print("\n" + "=" * 78)
    print(f"  Verdict: naive fp16 max err = {naive_max_err:.3e}, "
          f"fp16+BN-fold max err = {fold_max_err:.3e}")
    print("=" * 78)
    return 0 if fold_max_err < 0.05 else 1


if __name__ == "__main__":
    raise SystemExit(main())
