from typing import cast

import pytest

from ocr_common.npwp import column_thresholds_from_json, contract_fields, parse_column_thresholds
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


@pytest.mark.parametrize("raw", [None, "", "   ", "{}"])
def test_nothing_given_means_the_default_for_every_field(raw):
    assert column_thresholds_from_json(raw) is None


@pytest.mark.parametrize(
    ("raw", "reason"),
    [
        ("{not json", "valid JSON"),
        ("[0.9, 0.5]", "JSON object"),
        ('{"npwp": 0.9}', "unknown field(s) npwp"),
        ('{"nama": 1.5}', "between 0 and 1"),
        ('{"nama": -0.1}', "between 0 and 1"),
        ('{"nama": "tinggi"}', "must be a number"),
        ('{"nama": true}', "must be a number"),
    ],
)
def test_anything_else_is_refused_with_the_reason(raw, reason):
    with pytest.raises(ValueError, match=reason.replace("(", r"\(").replace(")", r"\)")):
        column_thresholds_from_json(raw)


def test_each_field_uses_its_own_threshold_else_the_default():
    fields = contract_fields(RESULT, 0.5, {"nomor_npwp": 0.9})

    assert (fields["nomor_npwp"]["confidence"], fields["nama"]["confidence"]) == (0, 1)
    assert contract_fields(RESULT, 0.5)["nomor_npwp"]["confidence"] == 1
