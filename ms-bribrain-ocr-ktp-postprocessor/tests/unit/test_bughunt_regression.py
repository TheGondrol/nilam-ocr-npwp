"""Regression tests for confirmed bugs in the ``dgc_pps`` microservice.

Each assigned bug gets TWO tests:

1. ``test_<bugid>_characterization_*`` — asserts the CURRENT (buggy) behavior.
   It PASSES against today's code and acts as a tripwire: if a future change
   alters the buggy behavior (e.g. someone applies the fix), this test fails
   loudly so the paired xfail can be revisited.

2. ``test_<bugid>_correct_behavior_*`` — asserts the CORRECT/desired behavior,
   decorated with ``@pytest.mark.xfail(strict=True)``. It reports as XFAIL
   today (bug present) and will XPASS — failing the strict gate — the moment
   the bug is fixed, prompting removal of the marker.

Bugs covered (see BUG_HUNT_REPORT.md): BUG-05, BUG-11, BUG-12, BUG-13,
BUG-14, BUG-26.

All tests exercise the REAL symbols from ``src/`` with minimal crafted inputs.
No GPU / model / network / DB is touched; the few heavy collaborators
(``insert_log``, ``remapping``, field matchers) are stubbed via mock/patch.
"""

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Confidence/ratio thresholds the production code compares against. With the
# shipped config these resolve to confidence=0.8 and ratio=80, so the crafted
# cells below (confidence 0.95-0.99, exact label text) clear both gates.
from src.services.constants import THRESHOLD_CONFIDENCE


def _box():
    """A throwaway 4-point bounding box; its exact value is irrelevant here."""
    return [[0, 0], [1, 0], [1, 1], [0, 1]]


# ---------------------------------------------------------------------------
# BUG-05 (FIXED 2026-06-17) — Unguarded ``data_cleaned[i + 2]`` in the ``nama``
# (and ``tempat/tgl lahir``) branch of ``mappingnext`` raised IndexError when
# the label was the last cell, which the route's broad ``except Exception``
# turned into an HTTP 500 — discarding every already-extracted field. Fixed by
# guarding both branches with ``i + 2 < len(data_cleaned)`` like ``alamat``.
# We drive ``mappingnext`` directly and stub ``remapping``.
# ---------------------------------------------------------------------------

def _bug05_data_nama_last():
    """Cleaned OCR list whose fuzzy-matched ``nama`` label lands at index len-1.

    The loop reaches ``i = len - 1`` for the ``nama`` cell and evaluates
    ``data_cleaned[i + 2]`` (== ``data_cleaned[len + 1]``) with no bounds check.
    """
    return [
        [_box(), ("nik", 0.99)],
        [_box(), ("3174012345678901", 0.99)],
        [_box(), ("nama", 0.95)],  # last cell -> i+2 is out of range
    ]


def test_bug05_nama_label_at_last_cell_does_not_raise():
    """BUG-05 (FIXED): ``nama`` label as the last cell no longer raises.

    The branch is now guarded with ``i + 2 < len(data_cleaned)`` (like alamat),
    so ``mappingnext`` returns a dict cleanly instead of an IndexError that the
    route would surface as a 500. Regression guard.
    """
    from src.services.ocr_processor import mappingnext

    with patch("src.services.ocr_processor.remapping", side_effect=lambda d, r: r):
        result, _ = mappingnext(_bug05_data_nama_last())

    assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# BUG-11 (FIXED 2026-06-17) — ``matching_nik`` accepted ANY cleanable-to-16-digit
# cell with no structural validation. It now rejects strings whose province code
# (digits 1-2) or birth day (digits 7-8, +40 for women) are out of range, so a
# non-NIK 16-digit number is no longer stamped as the NIK.
# ---------------------------------------------------------------------------

def test_bug11_implausible_nik_is_rejected():
    """BUG-11 (FIXED): a structurally-impossible NIK (region ``99``) is rejected."""
    from src.services.field_matchers.nik import matching_nik

    implausible = ["9999999999999999", 0.99, _box()]
    assert matching_nik(implausible) == ""


def test_bug11_plausible_nik_still_accepted():
    """A structurally-valid NIK is still accepted (no over-strict rejection)."""
    from src.services.field_matchers.nik import matching_nik

    valid = ["3174012801900001", 0.99, _box()]
    assert matching_nik(valid) == "3174012801900001"


# ---------------------------------------------------------------------------
# BUG-12 (FIXED 2026-06-17) — ``cleaning_tempat_meet_tgllength`` raised
# UnboundLocalError when the ``AMBIL_HURUF_SPASI`` match failed (``tempat_clear``
# was only bound inside the ``if match:`` branch). Fixed by initializing
# ``tempat_clear = False`` at the top.
# ---------------------------------------------------------------------------

def test_bug12_no_letter_match_returns_false_tuple():
    """BUG-12 (FIXED): returns ``(False, tempat)`` without raising when the
    letter/space pattern does not match (e.g. ``"12"`` -> ``""``)."""
    from src.services.field_matchers.ttl import cleaning_tempat_meet_tgllength

    cleared, tempat = cleaning_tempat_meet_tgllength("12")
    assert cleared is False
    assert isinstance(tempat, str)


# ---------------------------------------------------------------------------
# BUG-13 (FIXED 2026-06-17) — ``format_ttl_new_sesuai`` passed the raw ``data``
# LIST to ``re.findall`` in its ``else`` branch, raising TypeError (re.findall
# needs str/bytes). Fixed by operating on a joined string. The caller masked it
# via try/except, silently dropping birthplace/DOB.
# ---------------------------------------------------------------------------

def test_bug13_short_first_token_returns_tuple():
    """BUG-13 (FIXED): returns a ``(tempat, number)`` 2-tuple of strings
    without raising, even when the else branch is taken."""
    from src.services.field_matchers.ttl import format_ttl_new_sesuai

    tempat, number = format_ttl_new_sesuai(["AB", "12-1990"])
    assert isinstance(tempat, str)
    assert isinstance(number, str)


# ---------------------------------------------------------------------------
# BUG-14 (FIXED 2026-06-17) — Operator-precedence in the nested ``update_result``
# closure (``field not in result or result[field][1] <= score and value != ""``)
# let an empty first-seen value be stored. Fixed by parenthesizing and applying
# the ``value != ""`` guard to all inserts (same fix in utils/remapper.py).
#
# We drive the REAL code path: ``mappingnext`` with the ``agama`` matcher forced
# to return ``""`` for a first-seen ``agama`` field.
# ---------------------------------------------------------------------------

def _bug14_data():
    """OCR list whose ``agama`` label matches; its value matcher will be
    stubbed to return ``""``. A trailing cell avoids any i+2 index issue."""
    return [
        [_box(), ("agama", 0.95)],
        [_box(), ("???", 0.95)],
        [_box(), ("xyz", 0.95)],
    ]


def test_bug14_empty_value_not_stored_on_first_insert():
    """BUG-14 (FIXED): an empty first-seen value is NOT stored, so the field
    key is absent from the result."""
    from src.services.ocr_processor import mappingnext

    with patch("src.services.ocr_processor.matching_agama", return_value=""), patch(
        "src.services.ocr_processor.remapping", side_effect=lambda d, r: r
    ):
        result, _ = mappingnext(_bug14_data())

    assert "agama" not in result


# ---------------------------------------------------------------------------
# BUG-26 (FIXED 2026-06-17) — The postprocess route logged ``raw_text[:100]``
# (raw OCR JSON carrying PII: NIK / name / DOB / address) in plaintext at INFO.
# It now logs only the payload length. We drive the REAL route handler and
# assert the request log line no longer echoes the raw OCR content.
# ---------------------------------------------------------------------------

# NIK appears within the first 100 chars so raw_text[:100] captures it.
_BUG26_NIK = "3174012345678901"
_BUG26_RAW = (
    '[[[[0,0],[1,0],[1,1],[0,1]],["NIK",0.99]],'
    '[[[0,0],[1,0],[1,1],[0,1]],["' + _BUG26_NIK + '",0.99]]]'
)


class _FakePayload:
    """Minimal stand-in for the OCRExtract schema (only ``.ocr_text`` used)."""

    def __init__(self, ocr_text):
        self.ocr_text = ocr_text


def _drive_postprocess_route(caplog):
    """Invoke the real route handler and capture its log records.

    Returns the list of captured log messages from the route logger.
    """
    import src.api.routes as routes

    fake_request = MagicMock()
    payload = _FakePayload(_BUG26_RAW)

    async def _run():
        with patch.object(routes, "insert_log", new=AsyncMock(return_value=None)), patch.object(
            routes,
            "mappingnext",
            return_value=({"nik": _BUG26_NIK}, [[0, 0]]),
        ):
            await routes.extract_text_lines(fake_request, payload)

    with caplog.at_level(logging.INFO, logger=routes.logger.name):
        asyncio.run(_run())

    return [rec.getMessage() for rec in caplog.records]


def test_bug26_request_log_omits_raw_ocr_content(caplog):
    """BUG-26 (FIXED): the request log line does NOT contain raw OCR content
    (no NIK / raw payload prefix) — only non-PII metadata such as length."""
    messages = _drive_postprocess_route(caplog)

    received = [m for m in messages if m.startswith("Received postprocess request")]
    assert received, f"expected a 'Received postprocess request' log line, got: {messages}"
    assert not any(_BUG26_NIK in m for m in received), (
        f"request log line must not echo raw OCR content, got: {received}"
    )


# Reference the import so linters don't flag it as unused; THRESHOLD_CONFIDENCE
# documents the live confidence gate the crafted cells are designed to clear.
assert float(THRESHOLD_CONFIDENCE) <= 0.95
