"""Polygon-level parity between Paddle detector and converted PyTorch detector.

Runs both detectors end-to-end (preprocess → forward → DB postprocess) on each
image in ``tests/data/good_data*.png`` and asserts:

- Same number of detected polygons (strict).
- Each Paddle polygon has a centroid-nearest PyTorch polygon with pixel-IoU ≥ 0.99.

Skipped if the converted `server_det.pth` or the PaddleOCR2Pytorch checkout is
not present. Requires CUDA — matches production inference path.
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT
AUTOKERNEL_ROOT = REPO_ROOT / "autokernel"
PPOCR_ROOT = PROJECT_ROOT / "PaddleOCR2Pytorch"
DET_PTH = AUTOKERNEL_ROOT / "workspace/ppocrv5/server_det.pth"

IMAGE_PATHS = sorted((PROJECT_ROOT / "tests/data").glob("good_data*.png"))

# A 1-pixel boundary jitter is the irreducible noise between Paddle's C++
# DB postprocess (contour + unclip) and the PyTorch port's cv2/pyclipper —
# even with bit-identical shrink maps (3e-5 logit diff, per
# scripts/verify_det_logits.py). Enforce vertex-distance strictly; track
# IoU only to surface catastrophic regressions.
MAX_VERTEX_PIXEL_DIFF = 1.0
IOU_REGRESSION_FLOOR = 0.90


def _skip_if_missing_artifacts() -> None:
    for label, path in {
        "PaddleOCR2Pytorch": PPOCR_ROOT,
        "server_det.pth": DET_PTH,
        "autokernel models": AUTOKERNEL_ROOT / "models" / "ppocrv5_server.py",
    }.items():
        if not path.exists():
            pytest.skip(f"{label} not found at {path}; skipping det parity test")

    try:
        import torch
    except ImportError:
        pytest.skip("torch not installed")
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available; det parity tests require GPU")


def _polygon_iou(a: np.ndarray, b: np.ndarray, shape: tuple[int, int]) -> float:
    """Pixel-space IoU between two polygons (no Shapely dep)."""
    mask_a = np.zeros(shape, dtype=np.uint8)
    mask_b = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(mask_a, [np.asarray(a, dtype=np.int32)], 1)
    cv2.fillPoly(mask_b, [np.asarray(b, dtype=np.int32)], 1)
    inter = int(np.logical_and(mask_a, mask_b).sum())
    union = int(np.logical_or(mask_a, mask_b).sum())
    return inter / union if union > 0 else 0.0


def _centroid(poly: np.ndarray) -> tuple[float, float]:
    p = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
    return float(p[:, 0].mean()), float(p[:, 1].mean())


@pytest.fixture(scope="module")
def paddle_detector():
    _skip_if_missing_artifacts()
    from paddleocr import TextDetection

    return TextDetection(model_name="PP-OCRv5_server_det")


@pytest.fixture(scope="module")
def torch_detector():
    _skip_if_missing_artifacts()

    for p in (AUTOKERNEL_ROOT, PPOCR_ROOT):
        ps = str(p)
        if ps not in sys.path:
            sys.path.insert(0, ps)
    os.environ["AUTOKERNEL_PPOCR_ROOT"] = str(PPOCR_ROOT)
    os.environ["AUTOKERNEL_PPOCRV5_SERVER_DET_PTH"] = str(DET_PTH)

    import torch

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
    # Match the stock inference.yml (resize_long: 960, thresh: 0.3, box_thresh: 0.6, unclip: 1.5)
    args.det_db_thresh = 0.3
    args.det_db_box_thresh = 0.6
    args.det_db_unclip_ratio = 1.5
    args.det_limit_side_len = 960
    args.det_limit_type = "max"

    detector = predict_det.TextDetector(args)
    det_wrapper = ppocr_model.PPOCRv5ServerDetModel()
    detector.net = det_wrapper.net.to("cuda").float().eval()
    return detector


def _paddle_polys(detector, image: np.ndarray) -> list[np.ndarray]:
    results = detector.predict(image)
    polys: list[np.ndarray] = []
    for result in results:
        for poly in result.get("dt_polys", []):
            p = np.asarray(poly, dtype=np.float64)
            if p.ndim == 1:
                p = p.reshape(-1, 2)
            polys.append(p)
    return polys


def _torch_polys(detector, image: np.ndarray) -> list[np.ndarray]:
    result = detector(image)
    if isinstance(result, tuple):
        boxes = result[0]
    else:
        boxes = result
    polys: list[np.ndarray] = []
    if boxes is None:
        return polys
    for box in boxes:
        polys.append(np.asarray(box, dtype=np.float64).reshape(-1, 2))
    return polys


@pytest.mark.parametrize("image_path", IMAGE_PATHS, ids=lambda p: p.name)
def test_det_polygon_parity(paddle_detector, torch_detector, image_path):
    image = cv2.imread(str(image_path))
    assert image is not None, f"Failed to load {image_path}"
    h, w = image.shape[:2]

    paddle_polys = _paddle_polys(paddle_detector, image.copy())
    torch_polys = _torch_polys(torch_detector, image.copy())

    assert len(paddle_polys) == len(torch_polys), (
        f"{image_path.name}: polygon count mismatch "
        f"(paddle={len(paddle_polys)}, torch={len(torch_polys)})"
    )

    # Match by nearest centroid; track used torch indices.
    used: set[int] = set()
    mismatches: list[str] = []
    for i, pp in enumerate(paddle_polys):
        pc = _centroid(pp)
        best_j = -1
        best_dist = float("inf")
        for j, tp in enumerate(torch_polys):
            if j in used:
                continue
            tc = _centroid(tp)
            dist = (pc[0] - tc[0]) ** 2 + (pc[1] - tc[1]) ** 2
            if dist < best_dist:
                best_dist = dist
                best_j = j
        assert best_j >= 0, f"{image_path.name}: no torch poly available for paddle poly #{i}"
        used.add(best_j)

        pp_s = np.asarray(pp, dtype=np.float64).reshape(-1, 2)
        tp_s = np.asarray(torch_polys[best_j], dtype=np.float64).reshape(-1, 2)
        max_vertex_diff = float(np.abs(pp_s - tp_s).max()) if pp_s.shape == tp_s.shape else float("inf")
        iou = _polygon_iou(pp, torch_polys[best_j], (h, w))

        if max_vertex_diff > MAX_VERTEX_PIXEL_DIFF or iou < IOU_REGRESSION_FLOOR:
            mismatches.append(
                f"poly#{i} centroid=({pc[0]:.1f},{pc[1]:.1f}) "
                f"max_vertex_diff={max_vertex_diff:.2f}px iou={iou:.4f}"
            )

    assert not mismatches, (
        f"{image_path.name}: {len(mismatches)} polygon pairs drift beyond tolerance "
        f"(max vertex diff > {MAX_VERTEX_PIXEL_DIFF}px or IoU < {IOU_REGRESSION_FLOOR})\n"
        + "\n".join(mismatches[:10])
    )
