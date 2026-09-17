"""Unit tests for OCR backend selection and result conversion."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

import src.services.ocr_backends as ocr_backends
from src.services.ocr_backends import (
    AutoKernelPPOCRv5Backend,
    FullPyTorchBackend,
    HybridOCRBackend,
    PaddleOCRBackend,
    create_ocr_backend,
    load_verified_replacement_specs,
    resolve_optimized_kernel_path,
)


class FakePaddleOCR:
    def __init__(self, *, paddlex_config):
        self.paddlex_config = paddlex_config
        self.predict = MagicMock(return_value=[{"rec_texts": ["ok"]}])


def test_paddle_backend_keeps_paddle_predict_contract():
    backend = PaddleOCRBackend("config.yaml", paddle_ocr_cls=FakePaddleOCR)
    image = np.zeros((2, 3, 3), dtype=np.uint8)

    result = backend.predict(image)

    assert backend.name == "paddle"
    assert backend.engine.paddlex_config == "config.yaml"
    assert result == [{"rec_texts": ["ok"]}]
    backend.engine.predict.assert_called_once_with(image)


def test_auto_backend_predict_keeps_rgb_and_paddle_result_shape():
    backend = object.__new__(AutoKernelPPOCRv5Backend)
    captured = {}

    def fake_system(image):
        captured["image"] = image.copy()
        boxes = [
            np.array([[1, 2], [3, 2], [3, 4], [1, 4]], dtype=np.float32),
        ]
        return boxes, [("hello", 0.987)], {"det": 0.01}

    backend.system = fake_system
    rgb = np.array([[[10, 20, 30], [1, 2, 3]]], dtype=np.uint8)

    result = backend.predict(rgb)

    assert captured["image"][0, 0].tolist() == [10, 20, 30]
    assert captured["image"][0, 1].tolist() == [1, 2, 3]
    assert result[0]["rec_texts"] == ["hello"]
    assert result[0]["rec_scores"] == [pytest.approx(0.987)]
    np.testing.assert_array_equal(
        result[0]["rec_polys"][0],
        np.array([[1, 2], [3, 2], [3, 4], [1, 4]], dtype=np.float32),
    )


def test_auto_backend_predict_rejects_non_rgb_input():
    backend = object.__new__(AutoKernelPPOCRv5Backend)

    with pytest.raises(ValueError):
        backend.predict(np.zeros((10, 10), dtype=np.uint8))


def test_resolve_optimized_kernel_path_handles_stale_absolute_paths(tmp_path):
    autokernel_root = tmp_path / "autokernel"
    kernel = autokernel_root / "workspace" / "kernel_graph_capture_900_optimized.py"
    kernel.parent.mkdir(parents=True)
    kernel.write_text("def kernel_fn(x): return x\n", encoding="utf-8")
    workspace = autokernel_root / "workspace" / "graph_capture_eval"
    workspace.mkdir()

    resolved = resolve_optimized_kernel_path(
        "/workspace/ppocr-triton/autokernel/workspace/kernel_graph_capture_900_optimized.py",
        autokernel_root,
        workspace,
    )

    assert resolved == kernel


def test_load_verified_replacement_specs_uses_verified_pass_stack(tmp_path):
    autokernel_root = tmp_path / "autokernel"
    kernel = autokernel_root / "workspace" / "kernel_graph_capture_900_optimized.py"
    kernel.parent.mkdir(parents=True)
    kernel.write_text("def kernel_fn(x): return x\n", encoding="utf-8")
    workspace = autokernel_root / "workspace" / "graph_capture_eval"
    workspace.mkdir()
    verification = {
        "verification": {"correctness": "PASS"},
        "optimized": {
            "kernels_replaced": [
                {
                    "type": "graph_capture",
                    "rank": 900,
                    "speedup": 6.998,
                    "path": "/workspace/ppocr-triton/autokernel/workspace/kernel_graph_capture_900_optimized.py",
                }
            ]
        },
    }
    (workspace / "verification_result.json").write_text(
        json.dumps(verification),
        encoding="utf-8",
    )

    specs = load_verified_replacement_specs(workspace, autokernel_root)

    assert len(specs) == 1
    assert specs[0].kernel_type == "graph_capture"
    assert specs[0].rank == 900
    assert specs[0].speedup == pytest.approx(6.998)
    assert specs[0].optimized_path == kernel


def test_load_verified_replacement_specs_rejects_failed_verification(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "verification_result.json").write_text(
        json.dumps({"verification": {"correctness": "FAIL"}}),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError):
        load_verified_replacement_specs(workspace, tmp_path / "autokernel")


def test_create_backend_auto_cpu_keeps_existing_paddle_runtime():
    settings = SimpleNamespace(
        ocr_backend="auto",
        ocr_autokernel_enabled=True,
        ocr_server_config_path="server.yaml",
        ocr_mobile_config_path="mobile.yaml",
    )

    backend = create_ocr_backend(
        settings,
        use_gpu=False,
        paddle_ocr_cls=FakePaddleOCR,
    )

    assert isinstance(backend, PaddleOCRBackend)
    assert backend.engine.paddlex_config == "mobile.yaml"


def test_create_backend_strict_disabled_autokernel_raises():
    settings = SimpleNamespace(
        ocr_backend="autokernel",
        ocr_autokernel_enabled=False,
        ocr_server_config_path="server.yaml",
        ocr_mobile_config_path="mobile.yaml",
    )

    with pytest.raises(RuntimeError):
        create_ocr_backend(settings, use_gpu=True, paddle_ocr_cls=FakePaddleOCR)


def test_hybrid_backend_predict_detects_with_paddle_and_recognises_with_pytorch():
    """Unit test: verify detect→crop→recognise flow without real models."""
    backend = object.__new__(HybridOCRBackend)

    # Stub detection-only model — returns one rectangular box
    box = np.array([[10, 10], [110, 10], [110, 50], [10, 50]], dtype=np.float64)
    backend._paddle_det = MagicMock()
    backend._paddle_det.predict = MagicMock(
        return_value=[{"dt_polys": [box]}]
    )

    # Stub the fast-path recognizer
    backend._fast_recognize = MagicMock(return_value=[("HELLO", 0.95)])

    rgb = np.zeros((100, 200, 3), dtype=np.uint8)
    result = backend.predict(rgb)

    assert backend.name == "hybrid"
    assert result[0]["rec_texts"] == ["HELLO"]
    assert result[0]["rec_scores"] == [pytest.approx(0.95)]
    np.testing.assert_array_equal(result[0]["rec_polys"][0], box)
    # Detection model was called with the original RGB image
    backend._paddle_det.predict.assert_called_once()
    # Fast recognizer was called with the crops
    backend._fast_recognize.assert_called_once()


def test_hybrid_backend_predict_returns_raw_results_in_reading_order(monkeypatch):
    backend = object.__new__(HybridOCRBackend)

    top_left = np.array([[10, 14], [40, 14], [40, 24], [10, 24]], dtype=np.float64)
    top_right = np.array([[60, 10], [90, 10], [90, 20], [60, 20]], dtype=np.float64)
    bottom = np.array([[20, 50], [80, 50], [80, 62], [20, 62]], dtype=np.float64)

    backend._paddle_det = MagicMock()
    backend._paddle_det.predict = MagicMock(
        return_value=[{"dt_polys": [bottom, top_right, top_left]}]
    )

    def fake_crop(_image, box):
        marker = int(np.asarray(box)[0][0])
        return np.full((2, 2, 3), marker, dtype=np.uint8)

    def fake_recognize(crops):
        return [(f"x{int(crop[0, 0, 0])}", 0.9) for crop in crops]

    monkeypatch.setattr(ocr_backends, "_crop_text_region", fake_crop)
    backend._fast_recognize = MagicMock(side_effect=fake_recognize)

    result = backend.predict(np.zeros((100, 120, 3), dtype=np.uint8))

    assert result[0]["rec_texts"] == ["x10", "x60", "x20"]
    np.testing.assert_array_equal(result[0]["rec_polys"][0], top_left)
    np.testing.assert_array_equal(result[0]["rec_polys"][1], top_right)
    np.testing.assert_array_equal(result[0]["rec_polys"][2], bottom)


def test_sort_text_boxes_uses_upstream_insertion_same_row_ordering():
    left = np.array([[100, 102], [140, 102], [140, 120], [100, 120]], dtype=np.float64)
    middle = np.array([[220, 101], [260, 101], [260, 119], [220, 119]], dtype=np.float64)
    right = np.array([[340, 100], [380, 100], [380, 118], [340, 118]], dtype=np.float64)

    ordered = ocr_backends._sort_text_boxes([right, middle, left])

    np.testing.assert_array_equal(ordered[0], left)
    np.testing.assert_array_equal(ordered[1], middle)
    np.testing.assert_array_equal(ordered[2], right)


def test_compute_det_resize_shape_matches_paddle_max_limit_rounding():
    resize_h, resize_w, ratio_h, ratio_w = ocr_backends._compute_det_resize_shape(
        997,
        1001,
        limit_side_len=1280,
        limit_type="max",
    )

    assert (resize_h, resize_w) == (992, 992)
    assert ratio_h == pytest.approx(992 / 997)
    assert ratio_w == pytest.approx(992 / 1001)


def test_compute_det_resize_shape_matches_paddle_min_limit_rounding():
    resize_h, resize_w, ratio_h, ratio_w = ocr_backends._compute_det_resize_shape(
        480,
        1280,
        limit_side_len=960,
        limit_type="min",
    )

    assert (resize_h, resize_w) == (960, 2560)
    assert ratio_h == pytest.approx(2.0)
    assert ratio_w == pytest.approx(2.0)


def test_hybrid_backend_predict_empty_when_no_boxes():
    backend = object.__new__(HybridOCRBackend)
    backend._paddle_det = MagicMock()
    backend._paddle_det.predict = MagicMock(return_value=[{"dt_polys": []}])
    backend._fast_recognize = MagicMock()

    result = backend.predict(np.zeros((100, 200, 3), dtype=np.uint8))

    assert result[0]["rec_texts"] == []
    assert result[0]["rec_scores"] == []
    assert result[0]["rec_polys"] == []
    backend._fast_recognize.assert_not_called()


def test_hybrid_backend_predict_rejects_non_rgb_input():
    backend = object.__new__(HybridOCRBackend)
    with pytest.raises(ValueError):
        backend.predict(np.zeros((10, 10), dtype=np.uint8))


def test_fullpytorch_backend_predict_detects_crops_and_fast_recognizes(monkeypatch):
    backend = object.__new__(FullPyTorchBackend)
    box = np.array([[10, 10], [110, 10], [110, 50], [10, 50]], dtype=np.float64)
    backend._detect_boxes = MagicMock(return_value=[box])
    backend._fast_recognize = MagicMock(return_value=[("HELLO", 0.95)])

    def fake_crop(_image, crop_box):
        assert crop_box is box
        return np.full((4, 8, 3), 7, dtype=np.uint8)

    monkeypatch.setattr(ocr_backends, "_crop_text_region", fake_crop)

    result = backend.predict(np.zeros((100, 200, 3), dtype=np.uint8))

    assert result[0]["rec_texts"] == ["HELLO"]
    assert result[0]["rec_scores"] == [pytest.approx(0.95)]
    np.testing.assert_array_equal(result[0]["rec_polys"][0], box)
    backend._detect_boxes.assert_called_once()
    backend._fast_recognize.assert_called_once()
    assert backend._fast_recognize.call_args.args[0][0].shape == (4, 8, 3)


def test_fullpytorch_backend_predict_empty_when_no_boxes():
    backend = object.__new__(FullPyTorchBackend)
    backend._detect_boxes = MagicMock(return_value=[])
    backend._fast_recognize = MagicMock()

    result = backend.predict(np.zeros((100, 200, 3), dtype=np.uint8))

    assert result[0]["rec_texts"] == []
    assert result[0]["rec_scores"] == []
    assert result[0]["rec_polys"] == []
    backend._fast_recognize.assert_not_called()


def test_fullpytorch_backend_predict_rejects_non_rgb_input():
    backend = object.__new__(FullPyTorchBackend)
    with pytest.raises(ValueError):
        backend.predict(np.zeros((10, 10), dtype=np.uint8))


def test_fullpytorch_recognition_batch_plan_limits_padding():
    backend = object.__new__(FullPyTorchBackend)
    backend._rec_imgH = 48
    backend._rec_imgW = 320
    backend.rec_batch_size = 8
    backend.rec_bucket_max_width_ratio = 1.30
    backend.system = SimpleNamespace(
        text_recognizer=SimpleNamespace(
            limited_min_width=16,
            limited_max_width=4000,
        )
    )

    target_widths = [
        *([320] * 17),
        331,
        333,
        338,
        381,
        401,
        406,
        414,
        424,
        452,
        530,
        538,
        544,
        583,
        616,
    ]
    crops = [np.zeros((48, width, 3), dtype=np.uint8) for width in target_widths]

    batches, resized_widths, truncated = backend._recognition_batch_plan(crops)

    # target_w is snapped to REC_W_BUCKETS = (320, 416, 512, 640, 800, 1024, 1280).
    # The 17 width-320 crops stay in bucket 320; widths 331-414 snap to 416;
    # 424, 452 snap to 512; 530-616 snap to 640. The planner packs into
    # batches of 8 with cross-bucket merge allowed under the 1.30 ratio.
    assert [len(indices) for _target_w, indices in batches] == [8, 8, 8, 7]
    assert [target_w for target_w, _indices in batches] == [320, 320, 416, 640]
    assert resized_widths[0] == 320
    assert resized_widths[len(target_widths) - 1] == 616
    assert truncated == set()


def test_fullpytorch_recognition_workspace_reuses_and_grows():
    torch = pytest.importorskip("torch")

    backend = object.__new__(FullPyTorchBackend)
    backend._rec_imgC = 3
    backend._rec_imgH = 48
    backend.rec_batch_size = 8

    first = backend._get_rec_batch_workspace(
        torch,
        slot=0,
        batch_size=2,
        target_w=64,
        dtype=torch.float32,
        device="cpu",
    )
    second = backend._get_rec_batch_workspace(
        torch,
        slot=0,
        batch_size=1,
        target_w=32,
        dtype=torch.float32,
        device="cpu",
    )

    assert first.shape == (2, 3, 48, 64)
    assert second.shape == (1, 3, 48, 32)
    assert (
        first.untyped_storage().data_ptr()
        == second.untyped_storage().data_ptr()
    )

    grown = backend._get_rec_batch_workspace(
        torch,
        slot=0,
        batch_size=4,
        target_w=128,
        dtype=torch.float32,
        device="cpu",
    )

    assert grown.shape == (4, 3, 48, 128)
    assert grown.untyped_storage().data_ptr() != first.untyped_storage().data_ptr()


def test_create_backend_hybrid_disabled_autokernel_raises():
    settings = SimpleNamespace(
        ocr_backend="hybrid",
        ocr_autokernel_enabled=False,
        ocr_server_config_path="server.yaml",
        ocr_mobile_config_path="mobile.yaml",
    )
    with pytest.raises(RuntimeError):
        create_ocr_backend(settings, use_gpu=True, paddle_ocr_cls=FakePaddleOCR)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.integration
def test_autokernel_backend_loads_converted_ppocrv5_stack_when_artifacts_exist():
    autokernel_root = Path(
        os.getenv("AUTOKERNEL_ROOT", str(PROJECT_ROOT / "autokernel"))
    ).resolve()
    ppocr_root = Path(
        os.getenv("AUTOKERNEL_PPOCR_ROOT", str(PROJECT_ROOT / "PaddleOCR2Pytorch"))
    ).resolve()
    det_weights = Path(
        os.getenv(
            "AUTOKERNEL_PPOCRV5_SERVER_DET_PTH",
            str(autokernel_root / "workspace/ppocrv5/server_det.pth"),
        )
    ).resolve()
    rec_weights = Path(
        os.getenv(
            "AUTOKERNEL_PPOCRV5_SERVER_REC_PTH",
            str(autokernel_root / "workspace/ppocrv5/server_rec.pth"),
        )
    ).resolve()
    workspace = Path(
        os.getenv(
            "AUTOKERNEL_WORKSPACE",
            str(autokernel_root / "workspace/graph_capture_eval"),
        )
    ).resolve()

    missing = [
        path
        for path in (autokernel_root, ppocr_root, det_weights, rec_weights, workspace)
        if not path.exists()
    ]
    if missing:
        pytest.skip("AutoKernel PP-OCRv5 artifacts are not available: " + ", ".join(map(str, missing)))

    torch = pytest.importorskip("torch")
    pytest.importorskip("cv2")
    pytest.importorskip("shapely")
    pytest.importorskip("pyclipper")
    if not torch.cuda.is_available():
        pytest.skip("AutoKernel optimized recognizer integration requires CUDA")

    specs = load_verified_replacement_specs(workspace, autokernel_root)
    assert specs, "expected verified AutoKernel replacement specs"

    backend = AutoKernelPPOCRv5Backend(
        autokernel_root=autokernel_root,
        ppocr_root=ppocr_root,
        det_weights_path=det_weights,
        rec_weights_path=rec_weights,
        workspace_path=workspace,
        use_gpu=True,
        optimize_recognizer=True,
        optimize_detector=False,
    )
    try:
        assert backend.name == "autokernel"
        assert backend.system is not None
    finally:
        backend.close()
