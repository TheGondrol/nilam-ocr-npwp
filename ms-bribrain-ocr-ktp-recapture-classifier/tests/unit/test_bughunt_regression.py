"""Regression tests for confirmed bug-hunt findings in dgc_rct.

Each bug gets two tests:

* ``test_<bugid>_characterization_*`` — asserts the CURRENT (buggy) behavior.
  It MUST PASS against today's code, pinning the defect in place so a future
  change that alters the behavior is noticed.
* ``test_<bugid>_correct_behavior_*`` — asserts the CORRECT / desired behavior
  and is decorated with ``@pytest.mark.xfail(strict=True, ...)``.  It MUST
  report as XFAIL today (because the bug is present) and will start XPASSing
  (failing the strict marker) once the bug is fixed, prompting removal of the
  marker.

All tests are deterministic and self-contained: credentials and the
``requests.Session`` are mocked; there is no real network / GCS access.
"""

import pytest
from unittest.mock import MagicMock, patch

from google.auth.exceptions import TransportError


# ---------------------------------------------------------------------------
# BUG-23 (FIXED 2026-06-17) — GCS credential refresh only handled
#          ``RefreshError``; other refresh failures escaped unwrapped and leaked
#          the ``requests.Session``.
#
# Location: src/services/gcs_service.py (_get_credentials, refresh path)
#
# The refresh now runs the Session as a context manager (always closed) and
# wraps ANY refresh failure (TransportError / Timeout / etc.) in ``RuntimeError``.
# ---------------------------------------------------------------------------


def _make_expired_creds(refresh_exc):
    """Build a mock credentials object whose refresh() raises ``refresh_exc``."""
    creds = MagicMock()
    creds.valid = False  # forces the refresh branch
    creds.refresh.side_effect = refresh_exc
    return creds


def test_bug23_wraps_transport_error_and_closes_session():
    """BUG-23 (FIXED): ANY refresh failure -> RuntimeError, and Session closed.

    A ``TransportError`` (or any non-RefreshError) raised by ``refresh()`` is
    wrapped in ``RuntimeError`` with the explanatory message, and the
    ``requests.Session`` created for the refresh is closed (via context-manager
    ``__exit__`` or an explicit ``.close()``).
    """
    import src.services.gcs_service as gcs_mod

    original_creds = gcs_mod._credentials
    gcs_mod._credentials = _make_expired_creds(
        TransportError("connection reset during token refresh")
    )

    mock_session_instance = MagicMock()
    # Support both `with requests.Session() as s:` and explicit close styles.
    mock_session_instance.__enter__.return_value = mock_session_instance

    try:
        with patch(
            "src.services.gcs_service.requests.Session",
            return_value=mock_session_instance,
        ), patch("src.services.gcs_service.google.auth.transport.requests.Request"):
            with pytest.raises(RuntimeError, match="Failed to refresh GCS credentials"):
                gcs_mod._get_credentials()

        closed = (
            mock_session_instance.close.called
            or mock_session_instance.__exit__.called
        )
        assert closed, "refresh Session was never closed (leaked socket)"
    finally:
        gcs_mod._credentials = original_creds
