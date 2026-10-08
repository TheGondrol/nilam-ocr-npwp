from typing import cast

import pytest

from ocr_common.npwp import (
    auto_accept,
    column_thresholds_from_json,
    contract_fields,
    guardrails_threshold_from_json,
    guardrails_value,
    parse_column_thresholds,
    parse_guardrails_threshold,
)
from ocr_common.types import FinalResult

RAW = {
    "fields": {
        "nomor_npwp": {"value": "12.345.678.9-012.345", "confidence": 0.99},
        "nama": {"value": "BUDI SANTOSO", "confidence": 0.97},
        "nama_badan": {"value": None, "confidence": 0.0},
    },
    "scoring": {"npwp_confidence": 0.7296, "name_confidence": 0.9471},
}
RESULT = cast(FinalResult, RAW)


def test_the_central_orchestrators_thresholds_are_read():
    assert column_thresholds_from_json('{"nomor_npwp": 0.9, "nama": 0.5}') == {"nomor_npwp": 0.9, "nama": 0.5}
    assert parse_column_thresholds({"nama": 1}) == {"nama": 1.0}


def test_all_field_is_spread_over_every_field_and_a_fields_own_key_wins():
    assert column_thresholds_from_json('{"all_field": 0.8}') == {"nomor_npwp": 0.8, "nama": 0.8}
    assert parse_column_thresholds({"all_field": 0.8, "nama": 0.5}) == {"nomor_npwp": 0.8, "nama": 0.5}
    assert parse_column_thresholds({"nama": 0.5, "all_field": 0.8}) == {"nomor_npwp": 0.8, "nama": 0.5}


@pytest.mark.parametrize("raw", [None, "", "   ", "{}"])
def test_nothing_given_means_the_default_for_every_field(raw):
    assert column_thresholds_from_json(raw) is None


@pytest.mark.parametrize(
    ("raw", "reason"),
    [
        ("{not json", "valid JSON"),
        ("[0.9, 0.5]", "JSON object"),
        ('{"npwp": 0.9}', "unknown field(s) npwp"),
        ('{"all_fields": 0.9}', "unknown field(s) all_fields"),
        ('{"all_field": 0.9, "npwp": 0.9}', "unknown field(s) npwp"),
        ('{"all_field": 1.5}', "all_field must be between 0 and 1"),
        ('{"nama": 1.5}', "between 0 and 1"),
        ('{"nama": -0.1}', "between 0 and 1"),
        ('{"nama": "tinggi"}', "must be a number"),
        ('{"nama": true}', "must be a number"),
    ],
)
def test_anything_else_is_refused_with_the_reason(raw, reason):
    with pytest.raises(ValueError, match=reason.replace("(", r"\(").replace(")", r"\)")):
        column_thresholds_from_json(raw)


def test_the_guardrails_threshold_is_read_from_its_object_keyed_by_guardrails_name():
    assert guardrails_threshold_from_json('{"acc_rej": 0.8}') == 0.8
    assert parse_guardrails_threshold({"acc_rej": 0.3}) == 0.3


@pytest.mark.parametrize("raw", [None, "", "   ", "{}"])
def test_no_guardrails_threshold_means_none_was_sent(raw):
    assert guardrails_threshold_from_json(raw) is None


@pytest.mark.parametrize(
    ("raw", "reason"),
    [
        ("0.3", "JSON object"),
        ("{not json", "valid JSON"),
        ("[0.3]", "JSON object"),
        ('{"accept": 0.3}', "unknown guardrails accept"),
        ('{"acc_rej": 0.3, "blur": 0.3}', "unknown guardrails blur"),
        ('{"acc_rej": "tinggi"}', "must be a number"),
        ('{"acc_rej": true}', "must be a number"),
        ('{"acc_rej": 0}', "between 0 and 1 \\(exclusive\\)"),
        ('{"acc_rej": 1}', "between 0 and 1 \\(exclusive\\)"),
        ('{"acc_rej": 1.5}', "between 0 and 1"),
    ],
)
def test_any_other_guardrails_threshold_is_refused_with_the_reason(raw, reason):
    with pytest.raises(ValueError, match=reason):
        guardrails_threshold_from_json(raw)


def test_each_field_uses_its_own_threshold_else_the_default():
    fields = contract_fields(RESULT, 0.5, {"nomor_npwp": 0.9})

    assert (fields["nomor_npwp"]["confidence"], fields["nama"]["confidence"]) == (0, 1)
    assert contract_fields(RESULT, 0.5)["nomor_npwp"]["confidence"] == 1


# --- without thresholds: auto accept and probabilities ---------------------------------------------------------

TWO_PAGES = {
    "passed": False,
    "reason": "Document rejected by guardrails: 1/2 page(s) rejected (confidence 0.70)",
    "document": {"verdict": "reject", "confidence": 0.7, "n_pages": 2, "n_approve": 1, "n_reject": 1},
    "pages": [
        {"page_index": 0, "proba_approve": 0.96123, "proba_reject": 0.03877, "verdict": "accepted"},
        {"page_index": 1, "proba_approve": 0.3, "proba_reject": 0.7, "verdict": "reject"},
    ],
}


def test_auto_accept_passes_the_document_with_its_lowest_accepted_probability():
    report = auto_accept(TWO_PAGES)

    assert (report["passed"], report["reason"], report["auto_accepted"], report["score"]) == (True, None, True, 0.3)
    assert report["document"] == TWO_PAGES["document"], "the service's own verdict is kept"
    assert guardrails_value(report) == 0.3


def test_guardrails_value_is_0_1_with_a_threshold_and_none_without_a_report():
    assert guardrails_value({**TWO_PAGES, "passed": True}) == 0
    assert guardrails_value(TWO_PAGES) == 1
    assert guardrails_value(None) is None


def test_without_a_threshold_each_field_has_its_trust_probability():
    assert contract_fields(RESULT) == {
        "nomor_npwp": {"value": "12.345.678.9-012.345", "confidence": 0.7296},
        "nama": {"value": "BUDI SANTOSO", "confidence": 0.9471},
    }
    # A threshold for one field: that one is 0/1, the other keeps its probability.
    assert contract_fields(RESULT, None, {"nama": 0.95}) == {
        "nomor_npwp": {"value": "12.345.678.9-012.345", "confidence": 0.7296},
        "nama": {"value": "BUDI SANTOSO", "confidence": 0},
    }
