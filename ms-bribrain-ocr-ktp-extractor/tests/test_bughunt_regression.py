"""Regression tests for confirmed bug-hunt findings in the dgc_ext service.

Covers:
  - BUG-16: cleanup_ocr leaks the dedicated OCR model thread pool
            (``_ocr_executor``) and never invokes ``_ocr_instance.close()``,
            so backend GPU/CUDA/Paddle resources are never released on
            shutdown / in-process re-init (uvicorn --reload, tests, atexit).

Each bug has two tests:
  1. ``test_<bug>_characterization_*`` — asserts the CURRENT (buggy) behavior.
     It MUST PASS against today's code.
  2. ``test_<bug>_correct_behavior_*`` — asserts the DESIRED behavior.
     Decorated with ``@pytest.mark.xfail(strict=True, ...)`` so it reports
     XFAIL today and will flip to a real failure (alerting us to remove the
     marker) once the bug is fixed.

These tests manipulate the module-level globals of
``src.services.ocr_service`` directly (their exact names are read from the
source) and use ``unittest.mock`` stubs — no GPU, no real OCR backend, no IO.
"""

import pytest
from unittest.mock import MagicMock

import src.services.ocr_service as ocr_service
from src.services.ocr_service import cleanup_ocr


@pytest.fixture
def reset_ocr_globals():
    """Snapshot and restore the ocr_service module globals around a test.

    cleanup_ocr() mutates module-level globals (and atexit also registers it),
    so we save/restore them to keep tests isolated and avoid leaking the Mocks
    we install into other tests or the interpreter shutdown hook.
    """
    saved = {
        "_ocr_instance": ocr_service._ocr_instance,
        "_executor": ocr_service._executor,
        "_ocr_executor": ocr_service._ocr_executor,
        "_ocr_initialized": ocr_service._ocr_initialized,
    }
    try:
        yield
    finally:
        ocr_service._ocr_instance = saved["_ocr_instance"]
        ocr_service._executor = saved["_executor"]
        ocr_service._ocr_executor = saved["_ocr_executor"]
        ocr_service._ocr_initialized = saved["_ocr_initialized"]


# ---------------------------------------------------------------------------
# BUG-16 — cleanup_ocr leaks the OCR model thread pool and never closes backend
# ---------------------------------------------------------------------------

def test_bug16_cleanup_shuts_executor_and_closes_backend(reset_ocr_globals):
    """BUG-16 (FIXED): cleanup_ocr() shuts down ``_ocr_executor`` AND calls
    ``_ocr_instance.close()`` before clearing the reference, so the OCR model
    thread is joined and backend GPU/Paddle resources are released."""
    mock_instance = MagicMock()
    mock_ocr_executor = MagicMock()
    mock_executor = MagicMock()

    ocr_service._ocr_instance = mock_instance
    ocr_service._ocr_executor = mock_ocr_executor
    ocr_service._executor = mock_executor
    ocr_service._ocr_initialized = True

    cleanup_ocr()

    # Dedicated OCR model executor is shut down and its global cleared.
    mock_ocr_executor.shutdown.assert_called_once()
    assert ocr_service._ocr_executor is None

    # Backend close() is invoked to release CUDA/Paddle resources.
    mock_instance.close.assert_called_once()

    # And the existing teardown still holds.
    mock_executor.shutdown.assert_called_once()
    assert ocr_service._ocr_instance is None
    assert ocr_service._ocr_initialized is False
