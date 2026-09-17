"""Tests for the AutoKernel PP-OCRv5 logits verifier script."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def _load_verifier_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "verify_autokernel_ppocrv5_logits.py"
    spec = importlib.util.spec_from_file_location("verify_autokernel_ppocrv5_logits", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_synthetic_recognizer_input_is_deterministic_and_normalized():
    verifier = _load_verifier_module()
    first_rng = np.random.default_rng(123)
    second_rng = np.random.default_rng(123)

    first = verifier._make_synthetic_recognizer_input(7, first_rng, "3,48,320")
    second = verifier._make_synthetic_recognizer_input(7, second_rng, "3,48,320")

    assert first.shape == (3, 48, 320)
    assert first.dtype == np.float32
    assert float(first.min()) >= -1.0
    assert float(first.max()) <= 1.0
    np.testing.assert_array_equal(first, second)


def test_synthetic_recognizer_input_rejects_bad_shape():
    verifier = _load_verifier_module()

    with pytest.raises(ValueError):
        verifier._make_synthetic_recognizer_input(0, np.random.default_rng(1), "1,48,320")


def test_logits_verifier_reports_missing_artifacts_before_importing_torch(tmp_path):
    verifier = _load_verifier_module()
    args = SimpleNamespace(
        autokernel_root=str(tmp_path / "missing_autokernel"),
        ppocr_root=str(tmp_path / "missing_ppocr"),
        det_weights=str(tmp_path / "server_det.pth"),
        rec_weights=str(tmp_path / "server_rec.pth"),
        workspace=str(tmp_path / "workspace"),
        device="cuda",
        dtype="float16",
        image_shape="3,48,320",
        samples=100,
        seed=20260410,
        atol=5e-3,
        rtol=5e-3,
    )

    with pytest.raises(FileNotFoundError, match="Missing required artifacts"):
        verifier.verify_logits(args)
