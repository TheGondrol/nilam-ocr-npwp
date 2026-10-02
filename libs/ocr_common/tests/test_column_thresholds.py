from typing import cast

import pytest

from ocr_common.npwp import (
    column_thresholds_from_json,
    contract_fields,
    guardrails_threshold_from_json,
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
def test_no_guardrails_threshold_means_the_guardrails_services_own(raw):
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
