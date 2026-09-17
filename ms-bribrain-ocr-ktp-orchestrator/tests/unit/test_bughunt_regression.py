"""
Regression tests for confirmed bugs in the dgc_oct microservice.

Covers BUG-03, BUG-09, BUG-10, BUG-15 from BUG_HUNT_REPORT.md.

For each bug there are two tests:

1. ``test_<bugid>_characterization_<short>`` pins the CURRENT (buggy) behavior.
   These MUST PASS against today's code — they document the defect as it
   exists now.
2. ``test_<bugid>_correct_behavior_<short>`` asserts the DESIRED/FIXED
   behavior and is decorated with ``@pytest.mark.xfail(strict=True, ...)``.
   These MUST report as XFAIL today (the bug is present) and will start
   XPASS-ing — failing the strict-xfail and alerting maintainers — once the
   bug is fixed, at which point the xfail decorator should be removed.

All tests exercise the REAL symbols from ``src/`` with tiny crafted inputs and
stubs. No GPU, network, DB, or model checkpoints are touched.
"""

import asyncio

import cv2
import numpy as np
import pytest

from src.services.crop_helper import crop_image
from src.services.quality_service import get_lowest_confidence
from src.middleware.rate_limiter import SlidingWindowRateLimiter
from src.schemas.database_schema import OcrStatus


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _valid_jpeg_bytes() -> bytes:
    """A tiny but genuinely decodable JPEG so cv2.imdecode succeeds.

    This forces execution past crop_image's image-decode/empty guard and into
    the bbox-shape handling that the bugs concern.
    """
    img = np.ones((100, 100, 3), dtype=np.uint8) * 255
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


class _FakeClient:
    """Minimal stand-in for ``request.client`` (only ``.host`` is read)."""

    def __init__(self, host: str):
        self.host = host


class _FakeRequest:
    """Minimal Request stub exposing ``.headers`` (dict) and ``.client``.

    ``_get_client_id`` only calls ``request.headers.get(...)`` and reads
    ``request.client.host``, so a plain object with a real dict is sufficient
    and avoids constructing a full Starlette Request/ASGI scope.
    """

    def __init__(self, headers: dict, client_host: str | None):
        self.headers = dict(headers)
        self.client = _FakeClient(client_host) if client_host is not None else None


# ===========================================================================
# BUG-03 (FIXED 2026-06-17) — Malformed NIK bbox from postprocess crashed crop
#   src/services/crop_helper.py :: crop_image  +  manage_service.ocr_process
#
# crop_image iterated ``for point in bbox: if len(point) != 2`` after only
# checking ``len(bbox) == 4``; a flat 4-scalar list ``[x1, y1, x2, y2]`` passed
# the length-4 check, then ``len(0)`` raised TypeError, nuking an
# already-successful OCR result (HTTP 500). Fixed two ways: crop_image now
# normalizes a flat axis-aligned box to 4 points, and ocr_process wraps the
# crop+tamper step in try/except so ANY malformed box just skips the optional
# tamper check instead of failing the request.
# ===========================================================================

def test_bug03_flat_bbox_is_normalized_not_fatal():
    """BUG-03 (FIXED): a flat 4-scalar bbox is normalized and cropped.

    ``[0, 0, 100, 50]`` (an axis-aligned box given as two corners) is normalized
    to 4 corner points and cropped, returning JPEG bytes — instead of raising
    ``TypeError`` on ``len(point)`` and discarding an already-successful OCR
    result. Regression guard.
    """
    image_bytes = _valid_jpeg_bytes()

    result = crop_image(image_bytes, [0, 0, 100, 50])

    assert isinstance(result, bytes)
    assert len(result) > 0


def test_bug03_two_point_bbox_rejected_but_caller_guards():
    """crop_image still rejects an ambiguous 2-point bbox with ValueError.

    That is intentional: the orchestrator (manage_service.ocr_process) now wraps
    the crop+tamper step in try/except, so a 2-point (or otherwise malformed)
    box skips the optional tamper check rather than nuking a good OCR result.
    """
    image_bytes = _valid_jpeg_bytes()

    with pytest.raises(ValueError) as exc_info:
        crop_image(image_bytes, [[0, 0], [100, 0]])

    assert "exactly 4 points" in str(exc_info.value)


# ===========================================================================
# BUG-09 (FIXED 2026-06-17) — get_lowest_confidence indexed line[1][1] blindly
#   src/services/quality_service.py :: get_lowest_confidence
#
# It assumed every OCR line was shaped ``[bbox, [text, confidence]]``. A
# dict-shaped or too-short line made ``line[1][1]`` raise (KeyError/IndexError);
# since it is called OUTSIDE the try block in request_quality_dl, that bubbled
# to a 500 for an otherwise-valid image. Fixed by skipping malformed entries.
# get_lowest_confidence is an async coroutine; we drive it directly.
# ===========================================================================

def test_bug09_dict_line_is_skipped():
    """BUG-09 (FIXED): a dict-shaped OCR line is skipped, not fatal."""
    line = {"text": "NIK", "bbox": [1, 2, 3, 4]}

    assert asyncio.run(get_lowest_confidence([line])) == []


def test_bug09_short_line_is_skipped():
    """BUG-09 (FIXED): an OCR line whose element [1] is too short is skipped."""
    line = ["bbox", ["only-text"]]

    assert asyncio.run(get_lowest_confidence([line])) == []


def test_bug09_skips_malformed_keeps_valid_lines():
    """BUG-09 (FIXED): malformed lines are skipped while valid ones are kept.

    Given one malformed (dict) line and one well-formed line, only the valid
    line is returned. Regression guard.
    """
    malformed = {"text": "NIK", "bbox": [1, 2, 3, 4]}
    valid = [[10, 20, 30, 40], ["3201xxxx", 0.42]]

    result = asyncio.run(get_lowest_confidence([malformed, valid], k=5))

    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0]["text"] == "3201xxxx"
    assert result[0]["confidence"] == pytest.approx(0.42)


# ===========================================================================
# BUG-10 (FIXED 2026-06-17) — Rate limiter trusted client-supplied
#   X-Forwarded-For. src/middleware/rate_limiter.py :: _get_client_id now only
#   honors XFF when the immediate peer is a configured trusted proxy (empty by
#   default), and otherwise keys on the real peer (request.client.host) — so a
#   client can no longer rotate XFF to mint a fresh rate-limit bucket.
# ===========================================================================

def test_bug10_untrusted_xforwarded_for_ignored():
    """BUG-10 (FIXED): with no trusted-proxy allowlist, key on the real peer.

    The direct peer is ``10.0.0.1``; the spoofed ``X-Forwarded-For: 1.2.3.4``
    must NOT be trusted, so the client id is the real peer.
    """
    limiter = SlidingWindowRateLimiter()
    request = _FakeRequest(
        headers={"X-Forwarded-For": "1.2.3.4"},
        client_host="10.0.0.1",
    )

    assert limiter._get_client_id(request) == "10.0.0.1"


def test_bug10_trusted_proxy_xff_is_honored():
    """When the peer IS a trusted proxy, the right-most XFF hop is used."""
    limiter = SlidingWindowRateLimiter(trusted_proxies={"10.0.0.1"})
    request = _FakeRequest(
        headers={"X-Forwarded-For": "1.2.3.4"},
        client_host="10.0.0.1",
    )

    assert limiter._get_client_id(request) == "1.2.3.4"


# ===========================================================================
# BUG-15 (FIXED 2026-06-17) — Business 4xx rejections were persisted as
#   OcrStatus.FAILED, indistinguishable from server 5xx errors. OcrStatus now
#   has a distinct REJECTED member, and routes.py sets final_status=REJECTED for
#   pipeline errors with http_status < 500 (FAILED only for >= 500).
# ===========================================================================

def test_bug15_rejected_status_exists():
    """BUG-15 (FIXED): a distinct REJECTED status exists, separate from FAILED.

    Business rejections (HTTP 4xx) are persisted as REJECTED so they are not
    conflated with genuine server errors (FAILED).
    """
    assert "REJECTED" in OcrStatus.__members__
    assert OcrStatus.REJECTED is not OcrStatus.FAILED
    assert OcrStatus.REJECTED.value == "rejected"
