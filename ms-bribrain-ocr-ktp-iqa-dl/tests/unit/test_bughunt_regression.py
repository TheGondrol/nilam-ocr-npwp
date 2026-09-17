"""Regression tests for confirmed bug-hunt findings in dgc_idl.

Covers BUG-17, BUG-22 and BUG-24 from BUG_HUNT_REPORT.md.

For every bug there are two tests:

* ``test_<bugid>_characterization_*`` pins the CURRENT (buggy) behaviour and
  MUST PASS against today's code. It documents the defect so an accidental
  change is noticed.
* ``test_<bugid>_correct_behavior_*`` asserts the DESIRED behaviour and is
  decorated with ``@pytest.mark.xfail(strict=True)``. It reports as XFAIL today
  (the bug is present) and will turn into a failure -- prompting removal of the
  xfail marker -- once the bug is fixed.

The tests exercise the REAL symbols from ``src/`` with crafted in-memory inputs
and mocks: no GPU, no real model checkpoint, no network/GCS, no live DB.
"""

from __future__ import annotations

import io
from unittest.mock import MagicMock, patch

import pytest
import torch
from PIL import Image

from src.schemas.api_schema import Crop
from src.services.quality_service import process_and_classify_sync


# ---------------------------------------------------------------------------
# Shared helpers (mirrors the patterns in tests/unit/test_quality_service.py)
# ---------------------------------------------------------------------------

def _mock_transform(img):
    """Return a fixed-shape tensor without needing torchvision/a real image."""
    return torch.randn(3, 64, 320)


def _make_crop(x1, y1, x2, y2, text="t"):
    return Crop(
        bbox=[[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
        text=text,
        confidence=0.9,
    )


def _image_bytes(width=200, height=200):
    img = Image.new("RGB", (width, height), color=(128, 128, 128))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ===========================================================================
# BUG-24 (FIXED 2026-06-17) — forced-good crop kept the bad-class probability
# Location: src/services/quality_service.py
#
# When a crop's top class is "bad" but below confidence_threshold, the code
# forces pred_class=1/label="good"; it now also recomputes pred_score to the
# good-class probability so score and label are consistent.
# ===========================================================================

def _drive_single_crop(mock_config, probs):
    """Run process_and_classify_sync for one crop with a stubbed model.

    ``probs`` is the 2-element per-crop probability vector [bad, good] that the
    code under test should end up with AFTER its internal ``torch.softmax``
    (quality_service.py:180). Because the real model emits raw logits that the
    service softmaxes, the fake model must emit logits whose softmax equals
    ``probs``. Using ``log(p)`` as the logit achieves exactly that:
    ``softmax([log(p0), log(p1)]) == [p0, p1]`` when ``p0 + p1 == 1``.

    Returns the single CropPrediction produced.
    """
    import math

    import src.services.quality_service as qs

    mock_config.min_width_ratio = 1.0
    mock_config.min_width = 0
    mock_config.bad_crop_threshold = 4
    mock_config.debug_mode = False
    # threshold_provider.get_provider().get(...) is patched separately

    # Convert target probabilities to logits so that the service's softmax
    # reproduces ``probs`` exactly.
    logits = [math.log(p) for p in probs]

    model = MagicMock()
    model.side_effect = lambda x: torch.tensor([list(logits)] * x.shape[0])

    orig = (qs._model, qs._device, qs._transform)
    qs._model = model
    qs._device = torch.device("cpu")
    qs._transform = _mock_transform
    try:
        result = process_and_classify_sync(_image_bytes(), [_make_crop(0, 0, 150, 30)])
    finally:
        qs._model, qs._device, qs._transform = orig
    return result.predictions[0]


@patch("src.services.quality_service.get_provider")
@patch("src.services.quality_service.config")
def test_bug24_forced_good_uses_good_score(mock_config, mock_get_provider):
    """BUG-24 (FIXED): when a crop is forced to 'good' (bad-class conf 0.6 < 0.67),
    the reported score reflects the GOOD-class probability (0.4), not the
    bad-class one."""
    mock_get_provider.return_value.get.side_effect = lambda key, *a, **k: (
        0.67 if key == "confidence_threshold" else 4
    )

    pred = _drive_single_crop(mock_config, probs=(0.6, 0.4))

    assert pred.label == "good"
    assert pred.prediction == 1
    assert pred.score == pytest.approx(0.4)


# ===========================================================================
# BUG-17 (FIXED 2026-06-17) — GCS download raised on empty listing even when the
# model was already local. Location: src/services/gcs_service.py (also dgc_ext)
#
# The local-file short-circuit (`if model_path.exists(): return`) now runs
# BEFORE list_blobs, so an already-present local model skips the network call.
# ===========================================================================

def _make_gcs_mocks(mock_config, model_path, blobs):
    """Wire up config + a mocked GCS client whose bucket.list_blobs -> blobs."""
    mock_config.gcs_bucket_name = "bucket"
    mock_config.gcs_prefix = "models/"
    mock_config.gcs_model_prefix = "qualitydl"
    mock_config.get.return_value = str(model_path)

    mock_bucket = MagicMock()
    mock_bucket.list_blobs.return_value = list(blobs)
    mock_client = MagicMock()
    mock_client.bucket.return_value = mock_bucket
    return mock_client, mock_bucket


@patch("src.services.gcs_service.get_gcs_client")
@patch("src.services.gcs_service.config")
def test_bug17_local_model_short_circuits(mock_config, mock_get_client, tmp_path):
    """BUG-17 (FIXED): an existing local model short-circuits the download BEFORE
    any network call — no RuntimeError and no list_blobs invocation, mirroring
    download_model_minio's exists-first ordering."""
    from src.services.gcs_service import download_model_gcs

    model_file = tmp_path / "qualitydl_model.pt"
    model_file.write_text("already-on-disk")

    mock_client, mock_bucket = _make_gcs_mocks(mock_config, model_file, blobs=[])
    mock_get_client.return_value = mock_client

    # Returns cleanly (no RuntimeError) ...
    download_model_gcs()
    # ... and must not have touched the network listing at all.
    mock_bucket.list_blobs.assert_not_called()


# ===========================================================================
# BUG-22 (FIXED 2026-06-17) — created_at was TIMESTAMP(timezone=False)
# Location: src/schemas/database_schema.py (also dgc_irl)
#
# The timestamp column was timezone-naive (Postgres TIMESTAMP WITHOUT TIME
# ZONE, losing tz info). It is now timezone-aware (timezone=True).
# ===========================================================================

def test_bug22_created_at_is_timezone_aware():
    """BUG-22 (FIXED): OcrKtpQualityLog.created_at is a timezone-aware TIMESTAMP
    (timezone=True) so request timestamps carry tz information."""
    from src.schemas.database_schema import OcrKtpQualityLog

    created_at_type = OcrKtpQualityLog.__table__.c.created_at.type
    assert created_at_type.timezone is True
