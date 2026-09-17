"""
Regression tests for confirmed bug-hunt findings in dgc_tpr.

Covers:
  - BUG-18: Blocking image decode runs on the event loop in async
            `load_image_from_bytes` (src/services/image_preprocessing.py:69-87).
  - BUG-19: No image dimension / content-type validation; a truncated/malformed
            already-RGB image is decoded lazily so the failure surfaces later as
            a non-ValueError (OSError) -> 500 instead of 400
            (src/api/routes.py:240-258 + src/services/image_preprocessing.py).
  - BUG-29: Root endpoint advertises "/predict" but the real route is
            "/v1/ocr_tamper" (src/api/routes.py:153-159).

Each bug has:
  * a CHARACTERIZATION test asserting the CURRENT (buggy) behavior — must PASS today.
  * a strict-XFAIL test asserting the CORRECT/desired behavior — must XFAIL today.

All tests import the REAL symbols from src/ and are self-contained (no GPU,
no model checkpoint, no network/DB). Heavy paths are exercised at the tightest
level where the defect actually manifests.

NOTE on async: the production symbols are `async def`. To avoid depending on the
pytest-asyncio plugin (which lives only in the optional `test` extra and is not
guaranteed installed in every venv), these tests are plain synchronous functions
that drive the coroutines on an event loop they own via `_run_on_loop`. The loop
runs on the calling (test) thread, so "the event-loop thread" == the test thread
— which is exactly what the BUG-18 thread-affinity assertion relies on.
"""

import asyncio
import io
import threading

import pytest
from PIL import Image

from src.services.image_preprocessing import load_image_from_bytes
import src.services.image_preprocessing as image_preprocessing


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_on_loop(coro):
    """Drive a coroutine to completion on a fresh event loop on THIS thread.

    The loop executes on the calling thread, so any coroutine code that is NOT
    offloaded to an executor also runs on this thread. That property is what the
    BUG-18 characterization checks.
    """
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _valid_rgb_png_bytes(color=(123, 45, 67), size=(64, 64)) -> bytes:
    """A tiny, valid, already-RGB PNG produced in-test via PIL."""
    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _truncated_rgb_png_bytes() -> bytes:
    """A valid-header RGB PNG with its pixel data chopped off.

    `Image.open` parses the header lazily and succeeds; the actual pixel
    decode (`.load()` / `convert`) only fails later. This is the exact shape
    that drives BUG-19's mode-dependent misclassification. Verified offline:
    the full PNG is ~185 bytes; keeping the first half preserves the PNG
    signature + IHDR while corrupting the IDAT stream, so `.load()` raises
    `OSError: image file is truncated`.
    """
    full = _valid_rgb_png_bytes()
    return full[: len(full) // 2]


# ===========================================================================
# BUG-18 — blocking PIL decode on the event loop in async load_image_from_bytes
# ===========================================================================
#
# `load_image_from_bytes` is `async def` but calls `Image.open` / `convert`
# directly — it never delegates the CPU-bound decode to a thread pool
# (`run_in_executor`). We monkeypatch the `Image.open` symbol used by the
# module to record WHICH thread the decode runs on. Because the decode is not
# offloaded, it runs on the very same thread that drives the event loop.


def test_bug18_decode_offloaded_to_thread(monkeypatch):
    """BUG-18 (FIXED): the CPU-bound decode runs OFF the event-loop thread.

    load_image_from_bytes offloads via loop.run_in_executor, so Image.open runs
    on a worker thread, not the loop-driving thread.
    """
    loop_thread_ident = threading.get_ident()
    captured = {}

    real_open = Image.open

    def recording_open(fp, *args, **kwargs):
        captured["thread_ident"] = threading.get_ident()
        return real_open(fp, *args, **kwargs)

    monkeypatch.setattr(image_preprocessing.Image, "open", recording_open)

    img = _run_on_loop(load_image_from_bytes(_valid_rgb_png_bytes()))

    assert isinstance(img, Image.Image)
    assert "thread_ident" in captured, "Image.open was never invoked"
    # Offloaded -> runs on a DIFFERENT (worker) thread.
    assert captured["thread_ident"] != loop_thread_ident


# ===========================================================================
# BUG-19 (FIXED 2026-06-17) — truncated already-RGB image decoded lazily, so the
#          failure escaped later as OSError -> 500 instead of 400. Fixed by
#          force-loading pixels inside the try in _decode_image_sync (plus a
#          max-megapixels guard), so decode failures map to ValueError -> 400.
# ===========================================================================


def test_bug19_truncated_rgb_raises_valueerror_at_load():
    """BUG-19 (FIXED): a malformed/truncated image consistently raises ValueError.

    load_image_from_bytes now force-loads the pixels inside the try, so a
    truncated already-RGB image is rejected as ValueError at load time (mapping
    to a clean 400) instead of surfacing later as OSError -> 500.
    """
    truncated = _truncated_rgb_png_bytes()

    with pytest.raises(ValueError):
        _run_on_loop(load_image_from_bytes(truncated))


# ===========================================================================
# BUG-29 — root endpoint advertises "/predict" but the real route is
#          "/v1/ocr_tamper"
# ===========================================================================
#
# The root handler returns a plain dict, so we call it directly (no torch/app
# wiring needed) and inspect the advertised endpoint map.


def test_bug29_root_advertises_real_predict_path():
    """BUG-29 (FIXED): root advertises the real prediction route "/v1/ocr_tamper"."""
    from src.api.routes import root

    body = _run_on_loop(root())
    assert isinstance(body, dict)
    assert body["endpoints"]["predict"] == "/v1/ocr_tamper"
