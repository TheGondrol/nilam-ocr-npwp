#!/usr/bin/env python3
"""Inspect the one mismatching polygon pair on good_data.png to see how far apart
the paddle and pytorch boundaries are."""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT
AUTOKERNEL_ROOT = REPO_ROOT / "autokernel"
PPOCR_ROOT = PROJECT_ROOT / "PaddleOCR2Pytorch"
DET_PTH = AUTOKERNEL_ROOT / "workspace/ppocrv5/server_det.pth"
IMAGE = PROJECT_ROOT / "tests/data/good_data.png"

os.chdir(PROJECT_ROOT)
for p in (PROJECT_ROOT, AUTOKERNEL_ROOT, PPOCR_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
os.environ["AUTOKERNEL_PPOCR_ROOT"] = str(PPOCR_ROOT)
os.environ["AUTOKERNEL_PPOCRV5_SERVER_DET_PTH"] = str(DET_PTH)


def main() -> int:
    import torch
    from paddleocr import TextDetection

    paddle_det = TextDetection(model_name="PP-OCRv5_server_det")

    ppocr_model = importlib.import_module("models.ppocrv5_server")
    pytorchocr_utility = importlib.import_module("tools.infer.pytorchocr_utility")
    predict_det = importlib.import_module("tools.infer.predict_det")
    parser = pytorchocr_utility.init_args()
    args = parser.parse_args([])
    args.det_algorithm = "DB"
    args.det_yaml_path = str(PPOCR_ROOT / "configs/det/PP-OCRv5/PP-OCRv5_server_det.yml")
    args.det_model_path = str(DET_PTH)
    args.use_gpu = True
    args.image_dir = ""
    args.det_limit_side_len = 960
    args.det_limit_type = "max"
    args.det_db_thresh = 0.3
    args.det_db_box_thresh = 0.6
    args.det_db_unclip_ratio = 1.5
    detector = predict_det.TextDetector(args)
    det_wrapper = ppocr_model.PPOCRv5ServerDetModel()
    detector.net = det_wrapper.net.to("cuda").float().eval()

    image = cv2.imread(str(IMAGE))
    paddle_out = paddle_det.predict(image.copy())
    torch_out = detector(image.copy())

    paddle_polys = []
    for r in paddle_out:
        for p in r.get("dt_polys", []):
            paddle_polys.append(np.asarray(p, dtype=np.float64).reshape(-1, 2))

    torch_boxes = torch_out[0] if isinstance(torch_out, tuple) else torch_out
    torch_polys = [np.asarray(b, dtype=np.float64).reshape(-1, 2) for b in torch_boxes]

    print(f"paddle polys: {len(paddle_polys)}  torch polys: {len(torch_polys)}")

    # Find the worst-IoU pair.
    h, w = image.shape[:2]
    used = set()
    worst_iou = 1.0
    worst_i = -1
    worst_j = -1
    for i, pp in enumerate(paddle_polys):
        pc = pp.mean(axis=0)
        best_j = -1
        best_dist = float("inf")
        for j, tp in enumerate(torch_polys):
            if j in used:
                continue
            tc = tp.mean(axis=0)
            dist = float(((pc - tc) ** 2).sum())
            if dist < best_dist:
                best_dist = dist
                best_j = j
        used.add(best_j)

        mask_a = np.zeros((h, w), dtype=np.uint8)
        mask_b = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask_a, [pp.astype(np.int32)], 1)
        cv2.fillPoly(mask_b, [torch_polys[best_j].astype(np.int32)], 1)
        iou = float(np.logical_and(mask_a, mask_b).sum()) / max(
            float(np.logical_or(mask_a, mask_b).sum()), 1.0
        )
        if iou < worst_iou:
            worst_iou = iou
            worst_i = i
            worst_j = best_j

    print(f"\nWorst pair: paddle[{worst_i}] ↔ torch[{worst_j}]  IoU={worst_iou:.4f}")
    print(f"  paddle verts: {paddle_polys[worst_i].tolist()}")
    print(f"  torch  verts: {torch_polys[worst_j].tolist()}")
    diff = paddle_polys[worst_i] - torch_polys[worst_j]
    print(f"  vertex-wise diff: {diff.tolist()}")
    print(f"  max abs vertex diff: {float(np.abs(diff).max()):.2f} px")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
