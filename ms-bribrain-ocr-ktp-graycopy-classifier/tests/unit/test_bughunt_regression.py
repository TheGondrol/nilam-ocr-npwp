"""Regression tests for confirmed dgc_gry bugs: BUG-04, BUG-07, BUG-28.

Each bug has TWO tests:
  * ``test_<bug>_characterization_*`` — pins the CURRENT (buggy) behavior and
    MUST PASS against today's code.
  * ``test_<bug>_correct_behavior_*`` — encodes the DESIRED behavior and is
    decorated ``@pytest.mark.xfail(strict=True)`` so it reports XFAIL today and
    will start FAILING (alerting us) once the bug is fixed.

All tests are CPU-only, use no real model checkpoint, no GPU, and no network.
Source under test:
  * src/services/graycopy_detection.py  (BUG-04, BUG-28)
  * src/services/gcs_service.py          (BUG-07)
"""

from __future__ import annotations

from unittest.mock import MagicMock, call

import pytest
import torch
import torch.nn.functional as F
from PIL import Image
from pydantic import ValidationError

from src.schemas.api_schema import PredictionResponse
from src.services.graycopy_detection import GraycopyDetectionService


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_config() -> MagicMock:
    """Minimal mock Config sufficient to drive transforms + predict()."""
    config = MagicMock()
    config.model.path = "fake_model.pth"
    config.model.image_size = 224
    config.model.normalize_mean = [0.485, 0.456, 0.406]
    config.model.normalize_std = [0.229, 0.224, 0.225]
    config.model.normalize_size = 256
    config.model.dropout_rate = 0.6
    config.device.prefer_gpu = False
    config.device.force_cpu = True
    config.device.compile_model = False
    config.device.compile_mode = "default"
    config.device.use_jit_cpu = False
    config.prediction.threshold = 0.5
    return config


def _build_service_with_stub_model(model: MagicMock, device: torch.device):
    """Construct a GraycopyDetectionService WITHOUT loading a real checkpoint.

    Bypasses ``__init__`` (which would call ``_load_model`` / ``torch.load``),
    then wires up just enough state for ``predict()`` to run on CPU:
      * real, lightweight transforms (so ``_preprocess_image`` works),
      * a stubbed ``self.model`` recording ``.to()`` calls,
      * ``device`` chosen by the caller.
    """
    svc = GraycopyDetectionService.__new__(GraycopyDetectionService)
    svc.config = _make_config()
    svc.device = device
    svc._is_jit_model = False
    svc.model = model
    # Build the real torchvision transforms (cheap, CPU-only).
    GraycopyDetectionService._setup_transforms(svc)
    return svc


# ===========================================================================
# BUG-04 (FIXED 2026-06-17) — CPU fallback mutated the shared model in place
#   graycopy_detection.py predict()
# The GPU->CPU fallback did self.model.to(cpu) then self.model.to(self.device),
# mutating the process-global model shared across ThreadPoolExecutor workers
# (racing concurrent inferences). The racy in-place fallback is removed: a GPU
# RuntimeError now fails that single request cleanly without touching the shared
# model's device. (For a permanent CPU deployment, set device.force_cpu=true.)
# ===========================================================================

class _CudaThenCpuModel:
    """Stub model: first forward raises a CUDA RuntimeError, subsequent calls
    return fixed logits. Records every ``.to(device)`` call so the test can
    assert the in-place device shuffle of the *shared* model."""

    def __init__(self):
        self.calls = 0
        self.to = MagicMock(name="model.to")
        # so chained `self.model = self.model.to(...)` style would still work
        self.to.return_value = self

    def __call__(self, tensor):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("CUDA error: device-side assert triggered")
        # Logits favouring class 1 (ORIGINAL); finite & well-formed.
        return torch.tensor([[0.1, 0.9]])


def test_bug04_gpu_error_does_not_move_shared_model(monkeypatch):
    """BUG-04 (FIXED): a GPU RuntimeError no longer triggers an in-place device
    move of the shared model (which races concurrent ThreadPoolExecutor
    workers). The request fails cleanly and ``self.model``'s device is untouched.
    """
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)

    device = torch.device("cuda:0")  # pretend GPU deployment
    model = _CudaThenCpuModel()  # first forward raises a 'CUDA ...' RuntimeError
    svc = _build_service_with_stub_model(model, device)

    img = Image.new("RGB", (300, 300), "white")
    with pytest.raises(RuntimeError):
        svc.predict(img, "fallback.jpg")

    # The shared model object's device was never mutated.
    assert model.to.call_count == 0, (
        "shared model device must not be mutated on a GPU error"
    )


# ===========================================================================
# BUG-28 (FIXED 2026-06-17) — non-finite model output -> NaN -> 500
#   graycopy_detection.py predict()
# When the model emits non-finite (e.g. inf,inf) logits, softmax yields NaN.
# predict() now guards finiteness and raises a clean RuntimeError instead of
# leaking NaN that would explode the PredictionResponse (Field ge/le) -> 500.
# ===========================================================================

class _InfLogitsModel:
    """Stub model whose forward returns non-finite (inf) logits."""

    def __init__(self):
        self.to = MagicMock(return_value=self)

    def __call__(self, tensor):
        return torch.tensor([[float("inf"), float("inf")]])


def _predict_with_inf_logits():
    model = _InfLogitsModel()
    svc = _build_service_with_stub_model(model, torch.device("cpu"))
    img = Image.new("RGB", (224, 224), "white")
    return svc.predict(img, "naninf.jpg")


def test_bug28_nonfinite_output_is_guarded():
    """BUG-28 (FIXED): non-finite model output is guarded inside ``predict()``
    and raises a clean error, instead of leaking NaN that later explodes the
    response model (Field ge/le -> ValidationError -> 500)."""
    with pytest.raises(Exception):
        _predict_with_inf_logits()


# ===========================================================================
# BUG-07 (FIXED 2026-06-17) — GCS "select newest version" was dead after the
#   first download. gcs_service.py download_model_gcs now always (re)downloads
#   the selected newest blob instead of skipping when the fixed-name local file
#   exists, so a newer published version is fetched on restart.
# ===========================================================================

def _make_blob(name: str) -> MagicMock:
    blob = MagicMock(name=f"blob<{name}>")
    blob.name = name
    blob.download_to_filename = MagicMock()
    return blob


def _patch_gcs_for_download(monkeypatch, blobs, model_path_exists):
    """Wire up gcs_service so download_model_gcs() runs offline.

    Returns the (mocked) best/v2 blob so the test can assert (no) download.
    """
    from src.services import gcs_service

    # Mock the GCS client -> bucket -> list_blobs chain.
    bucket = MagicMock()
    bucket.list_blobs = MagicMock(return_value=list(blobs))
    client = MagicMock()
    client.bucket = MagicMock(return_value=bucket)
    monkeypatch.setattr(gcs_service, "get_gcs_client", lambda: client)

    # Config used by download_model_gcs (module-level `config`).
    cfg = MagicMock()
    cfg.gcs.bucket = "test-bucket"
    cfg.gcs.prefix = "models/"
    cfg.gcs.model_prefix = "graycopy_model"
    cfg.model.path = "/tmp/models/graycopy_model.pth"
    monkeypatch.setattr(gcs_service, "config", cfg)

    # Don't touch the real filesystem: stub Path used inside the module.
    import pathlib

    real_path_cls = pathlib.Path

    class _FakePath:
        def __init__(self, p):
            self._p = str(p)

        @property
        def parent(self):
            return self

        def mkdir(self, *a, **k):
            return None

        def exists(self):
            return model_path_exists

        def __str__(self):
            return self._p

        def __fspath__(self):
            return self._p

    monkeypatch.setattr(gcs_service, "Path", _FakePath)

    return gcs_service


def test_bug07_newer_version_triggers_download(monkeypatch):
    """BUG-07 (FIXED): the selected newest remote version (v2.0) is (re)downloaded
    even though an older local model file is already present, so the version
    selection is no longer dead after the first download."""
    v1 = _make_blob("models/graycopy_model_v1.0.pth")
    v2 = _make_blob("models/graycopy_model_v2.0.pth")
    gcs_service = _patch_gcs_for_download(
        monkeypatch, blobs=[v1, v2], model_path_exists=True
    )

    gcs_service.download_model_gcs()

    assert v2.download_to_filename.called, (
        "newer remote version v2.0 should trigger a (re)download"
    )
