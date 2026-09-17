#!/usr/bin/env python3
"""Dump Paddle and PyTorch detector output statistics on a deterministic input.

Used to triage large logit discrepancies before running the full verifier.
"""

from __future__ import annotations

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

AUTOKERNEL_ROOT = REPO_ROOT / "autokernel"
PPOCR_ROOT = PROJECT_ROOT / "PaddleOCR2Pytorch"
DET_SOURCE = PROJECT_ROOT / "src/models/server_models/ppocrv5_server_det_source/inference.pdiparams"
DET_PTH = AUTOKERNEL_ROOT / "workspace/ppocrv5/server_det.pth"


def summarise(name: str, arr: np.ndarray) -> None:
    print(
        f"{name:>20s}: shape={arr.shape} dtype={arr.dtype} "
        f"min={arr.min():+.6f} max={arr.max():+.6f} "
        f"mean={arr.mean():+.6f} std={arr.std():.6f} "
        f"|sum|={np.abs(arr).sum():.4f}"
    )


def main() -> int:
    rng = np.random.default_rng(20260417)
    h, w = 640, 640
    image = np.zeros((h, w, 3), dtype=np.float32)
    yy, xx = np.mgrid[0:h, 0:w]
    image[..., 0] = (xx * 3 + yy * 7) % 256
    image[..., 1] = (yy * 5 + 31) % 256
    image[..., 2] = ((xx // 3 + yy // 2) * 11) % 256
    for _ in range(24):
        x1 = int(rng.integers(0, w - 2))
        y1 = int(rng.integers(0, h - 2))
        x2 = int(rng.integers(x1 + 1, w + 1))
        y2 = int(rng.integers(y1 + 1, h + 1))
        color = rng.integers(0, 256, size=(3,), dtype=np.uint8).astype(np.float32)
        image[y1:y2, x1:x2, :] = color
    image += rng.normal(loc=0.0, scale=18.0, size=image.shape).astype(np.float32)
    image = np.clip(image, 0.0, 255.0)
    image = (image / 255.0 - 0.5) / 0.5
    tensor = image.transpose(2, 0, 1).copy()  # (3, 640, 640)
    summarise("input", tensor)

    # ------- Paddle -------
    import paddle

    layer = paddle.jit.load(str(DET_SOURCE.with_suffix("")))
    layer.eval()
    with paddle.no_grad():
        x_paddle = paddle.to_tensor(tensor[np.newaxis, ...])
        y_paddle = layer(x_paddle)

    print("\n[Paddle]")
    print(f"  output type: {type(y_paddle).__name__}")
    if isinstance(y_paddle, (list, tuple)):
        for i, item in enumerate(y_paddle):
            summarise(f"paddle[{i}]", item.numpy())
    elif isinstance(y_paddle, dict):
        for k, v in y_paddle.items():
            summarise(f"paddle[{k}]", v.numpy())
    else:
        summarise("paddle", y_paddle.numpy())

    # ------- PyTorch -------
    for p in (AUTOKERNEL_ROOT, PPOCR_ROOT):
        p_str = str(p)
        if p_str not in sys.path:
            sys.path.insert(0, p_str)
    os.environ["AUTOKERNEL_PPOCR_ROOT"] = str(PPOCR_ROOT)
    os.environ["AUTOKERNEL_PPOCRV5_SERVER_DET_PTH"] = str(DET_PTH)

    import importlib
    import torch

    ppocr_model = importlib.import_module("models.ppocrv5_server")
    model = ppocr_model.PPOCRv5ServerDetModel().eval()
    with torch.inference_mode():
        x_torch = torch.from_numpy(tensor).unsqueeze(0)
        # Call the underlying net to see what keys it returns
        raw = model.net(x_torch)
        print("\n[PyTorch]")
        print(f"  raw output type: {type(raw).__name__}")
        if isinstance(raw, dict):
            for k, v in raw.items():
                summarise(f"torch[{k}]", v.numpy())
        else:
            summarise("torch", raw.numpy())

        # And what `forward()` (the wrapper) actually returns:
        y_torch = model(x_torch)
        summarise("torch[wrapper]", y_torch.numpy())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
