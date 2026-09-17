#!/usr/bin/env python3
"""Verify converted PyTorch .pth produces ~identical outputs to the source PIR model.

Runs a fixed deterministic input through both the Paddle PIR inference model
and the converted PyTorch model, then reports the max absolute element-wise
difference between their outputs. A well-converted model should differ only
by floating-point noise (< 1e-4 max-abs).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.chdir(PROJECT_ROOT)
for _p in (PROJECT_ROOT, PROJECT_ROOT / "src", PROJECT_ROOT / "PaddleOCR2Pytorch"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np
import paddle
import torch

from src.services.ppocrv5_conversion import (
    server_det_architecture,
    server_rec_architecture,
)
from pytorchocr.base_ocr_v20 import BaseOCRV20


PPOCR_ROOT = PROJECT_ROOT / "PaddleOCR2Pytorch"

DET_SRC = PROJECT_ROOT / "src/models/server_models/ppocrv5_server_det_source/inference"
DET_PTH = PROJECT_ROOT / "autokernel/workspace/ppocrv5/server_det.pth"

REC_SRC = PROJECT_ROOT / "src/models/server_models/ppocrv5_server_rec_source/inference"
REC_PTH = PROJECT_ROOT / "autokernel/workspace/ppocrv5/server_rec.pth"


class _Model(BaseOCRV20):
    def __init__(self, config):
        super().__init__(config)


def _run_paddle(prefix: Path, x: np.ndarray) -> np.ndarray:
    paddle.disable_static()
    layer = paddle.jit.load(str(prefix))
    layer.eval()
    with paddle.no_grad():
        out = layer(paddle.to_tensor(x))
    if isinstance(out, (list, tuple)):
        out = out[0]
    return out.numpy()


def _run_torch(arch: dict, pth_path: Path, x: np.ndarray) -> np.ndarray:
    model = _Model(arch)
    sd = torch.load(str(pth_path), map_location="cpu")
    missing, unexpected = model.net.load_state_dict(sd, strict=False)
    print(f"    load_state_dict: missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print(f"    missing[:5]: {missing[:5]}")
    if unexpected:
        print(f"    unexpected[:5]: {unexpected[:5]}")
    model.net.eval()
    with torch.no_grad():
        out = model.net(torch.from_numpy(x))
    # det: dict {"maps": tensor}; rec: dict {"ctc": tensor, "nrtr": tensor}
    if isinstance(out, dict):
        # Prefer the primary inference output: "maps" for det, "ctc" for rec.
        for preferred in ("maps", "ctc"):
            if preferred in out:
                key = preferred
                break
        else:
            key = sorted(out.keys())[0]
        out = out[key]
    if isinstance(out, (list, tuple)):
        out = out[0]
    return out.detach().numpy()


def _compare(name: str, paddle_out: np.ndarray, torch_out: np.ndarray) -> None:
    print(f"  paddle.shape={paddle_out.shape}  torch.shape={torch_out.shape}")
    if paddle_out.shape != torch_out.shape:
        print(f"  !! SHAPE MISMATCH for {name}")
        return
    diff = np.abs(paddle_out - torch_out)
    print(f"  max-abs diff = {diff.max():.6g}")
    print(f"  mean-abs diff = {diff.mean():.6g}")
    print(f"  paddle range = [{paddle_out.min():.4f}, {paddle_out.max():.4f}]")
    print(f"  torch  range = [{torch_out.min():.4f}, {torch_out.max():.4f}]")


def verify_det() -> None:
    print("=== Detector ===")
    rng = np.random.default_rng(0)
    x = rng.standard_normal((1, 3, 320, 320)).astype("float32")
    paddle_out = _run_paddle(DET_SRC, x)
    torch_out = _run_torch(server_det_architecture(PPOCR_ROOT), DET_PTH, x)
    _compare("det", paddle_out, torch_out)


def verify_rec() -> None:
    print("=== Recognizer ===")
    rng = np.random.default_rng(1)
    x = rng.standard_normal((1, 3, 48, 320)).astype("float32")
    paddle_out = _run_paddle(REC_SRC, x)
    arch = server_rec_architecture(PPOCR_ROOT)
    torch_out = _run_torch(arch, REC_PTH, x)
    _compare("rec", paddle_out, torch_out)


def main() -> int:
    verify_det()
    print()
    verify_rec()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
