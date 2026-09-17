"""
Regression tests for confirmed bugs from BUG_HUNT_REPORT.md (dgc_irl).

Covers:
  - BUG-08: InsightFace unavailability / transient detector errors are reported
            identically as `is_rotated=True`, and each error resets the global
            singleton `_face_detector` to None.
            Location: src/services/rotation_detection.py:251-285

Each bug has TWO tests:
  * `test_<bugid>_characterization_*`  -> asserts the CURRENT (buggy) behavior;
                                          PASSES against today's code.
  * `test_<bugid>_correct_behavior_*`  -> asserts the DESIRED behavior; marked
                                          strict-xfail so it reports XFAIL today
                                          and will FAIL (alerting us) once fixed.

All tests are deterministic and self-contained: insightface / ONNX / the real
detector are never exercised. We monkeypatch `get_face_detector` to simulate the
infrastructure-error path (e.g. insightface not installed -> RuntimeError, or a
transient ONNX session error).

The autouse `_init_threshold_provider` fixture (tests/unit/conftest.py) seeds the
ThresholdProvider that `detect_image_rotation` reads via
`get_provider().get("rotation_min_face_proportion")`, so these tests do not need a
real threshold backend. The `landscape_image` fixture is also provided there.
"""
import numpy as np
import pytest
from unittest.mock import patch

import src.services.rotation_detection as rd
from src.services.rotation_detection import detect_image_rotation


@pytest.fixture
def _restore_singleton():
    """Snapshot and restore the module-level `_face_detector` singleton so a test
    that resets it to None (the bug under test) does not leak into other tests."""
    saved = rd._face_detector
    yield
    rd._face_detector = saved


# ---------------------------------------------------------------------------
# BUG-08 (FIXED 2026-06-17) — infra error was coerced to is_rotated=True
# ---------------------------------------------------------------------------
#
# `rotated_detection()` used to swallow ALL exceptions and return
# `(False, "Error ...")`, which `detect_image_rotation()` mapped to
# `is_rotated=True` on a 200 — masking an infrastructure failure (insightface
# missing, ONNX error) as a business verdict. Both now re-raise so the API can
# surface a 5xx. We exercise the public `detect_image_rotation` entry point.


def test_bug08_infra_error_propagates(landscape_image, _restore_singleton):
    """BUG-08 (FIXED): an infrastructure error (insightface missing / ONNX
    session failure) propagates to the caller (so the API returns a 5xx) instead
    of being masked as the business verdict `is_rotated=True`."""
    with patch(
        "src.services.rotation_detection.get_face_detector",
        side_effect=RuntimeError(
            "insightface is not installed in this environment."
        ),
    ):
        with pytest.raises(RuntimeError):
            detect_image_rotation(landscape_image)
