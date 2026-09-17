#!/usr/bin/env python3
"""Investigate *why* net.head.binarize.conv1 produces 1.2M activations."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT
os.chdir(PROJECT_ROOT)
for _p in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

DEFAULT_PPOCR_ROOT = PROJECT_ROOT / "PaddleOCR2Pytorch"
DEFAULT_AUTOKERNEL_ROOT = REPO_ROOT / "autokernel"
DEFAULT_DET_PTH = DEFAULT_AUTOKERNEL_ROOT / "workspace/ppocrv5/server_det.pth"
DEFAULT_IMAGE = PROJECT_ROOT / "tests/data/good_data.png"


def main() -> int:
    for p in (DEFAULT_AUTOKERNEL_ROOT, DEFAULT_PPOCR_ROOT):
        sys.path.insert(0, str(p))
    os.environ["AUTOKERNEL_PPOCR_ROOT"] = str(DEFAULT_PPOCR_ROOT)
    os.environ["AUTOKERNEL_PPOCRV5_SERVER_DET_PTH"] = str(DEFAULT_DET_PTH)

    import cv2
    import importlib
    import torch

    ppocr_model = importlib.import_module("models.ppocrv5_server")
    m = ppocr_model.PPOCRv5ServerDetModel().cuda().float().eval()

    # Preprocess image
    img = cv2.imread(str(DEFAULT_IMAGE))
    h, w = img.shape[:2]
    r = 1072.0 / max(h, w) if max(h, w) > 1072 else 1.0
    rh = int(round(h * r / 32)) * 32
    rw = int(round(w * r / 32)) * 32
    img = cv2.resize(img, (rw, rh))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = (img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    x = torch.from_numpy(img.transpose(2, 0, 1)[None, ...]).cuda().float()

    # Weight stats
    w_conv1 = m.net.head.binarize.conv1.weight
    bn = m.net.head.binarize.conv_bn1
    print(f"head.binarize.conv1 weight: shape={list(w_conv1.shape)} "
          f"abs_max={w_conv1.abs().max().item():.4e} mean_abs={w_conv1.abs().mean().item():.4e}")
    print(f"head.binarize.conv_bn1 weight (gamma): abs_max={bn.weight.abs().max().item():.4e} "
          f"mean={bn.weight.mean().item():.4e}")
    print(f"head.binarize.conv_bn1 bias (beta): abs_max={bn.bias.abs().max().item():.4e}")
    print(f"head.binarize.conv_bn1 running_mean: abs_max={bn.running_mean.abs().max().item():.4e}")
    print(f"head.binarize.conv_bn1 running_var: min={bn.running_var.min().item():.4e} "
          f"max={bn.running_var.max().item():.4e}")
    print(f"head.binarize.conv_bn1 running_std: max={bn.running_var.sqrt().max().item():.4e}")

    # Hook neck output (input to head)
    neck_out = {}
    def hook(_m, _i, o):
        if isinstance(o, (list, tuple)):
            for i, t in enumerate(o):
                neck_out[f"neck_out[{i}]"] = t.detach()
        else:
            neck_out["neck_out"] = o.detach()
    h = m.net.neck.register_forward_hook(hook)

    # Hook conv1 output
    conv1_out = {}
    def chook(_m, _i, o):
        conv1_out["out"] = o.detach()
    m.net.head.binarize.conv1.register_forward_hook(chook)
    m.net.head.binarize.conv_bn1.register_forward_hook(
        lambda _m, _i, o: conv1_out.__setitem__("bn_out", o.detach())
    )

    with torch.inference_mode():
        _ = m(x)

    for k, v in neck_out.items():
        print(f"{k}: shape={list(v.shape)} abs_max={v.abs().max().item():.4e} "
              f"mean_abs={v.abs().mean().item():.4e}")
    for k, v in conv1_out.items():
        print(f"conv1.{k}: shape={list(v.shape)} abs_max={v.abs().max().item():.4e} "
              f"mean_abs={v.abs().mean().item():.4e}")

    # Does the BN "absorb" the scale? BN gamma/(sqrt(var)) × conv_out effectively
    gamma = bn.weight
    std = bn.running_var.sqrt()
    effective_scale = (gamma / std).abs()
    print(f"\nBN effective per-channel scale |gamma/std|: "
          f"min={effective_scale.min().item():.4e} max={effective_scale.max().item():.4e} "
          f"mean={effective_scale.mean().item():.4e}")
    print(f"running_std max={std.max().item():.4e}  (this means conv output was trained "
          f"to have std of this scale)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
