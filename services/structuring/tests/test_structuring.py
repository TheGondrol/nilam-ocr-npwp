import pytest

from ocr_common.errors import ServiceError
from ocr_common.types import OcrBlock

from app.ml.rule_based import RuleBasedNpwpStructurer
from app.ml.utils import normalize_npwp
from app.services.structuring_service import StructuringService

LINES: list[OcrBlock] = [
    {"text": "KEMENTERIAN KEUANGAN REPUBLIK INDONESIA", "confidence": 0.99},
    {"text": "NPWP : 12.345.678.9-012.345", "confidence": 0.96},
    {"text": "NAMA : BUDI SANTOSO", "confidence": 0.95},
    {"text": "NAMA BADAN : PT CIPTA KARYA MANDIRI", "confidence": 0.93},
]


def _values(document):
    return {name: f["value"] for name, f in document["fields"].items()}


def test_rule_based_parses_labeled_lines():
    document = RuleBasedNpwpStructurer().structure(LINES)
    assert _values(document) == {
        "nomor_npwp": "123456789012345",
        "nama": "BUDI SANTOSO",
        "nama_badan": "PT CIPTA KARYA MANDIRI",
    }
    fields = document["fields"]
    assert fields["nomor_npwp"]["confidence"] == 0.96
    assert fields["nomor_npwp"]["source"] == "NPWP : 12.345.678.9-012.345"
    assert (document["flag"], document["flag_reason"]) == (False, None)


def test_rule_based_handles_paddle_style_lines_without_spaces():
    fields = RuleBasedNpwpStructurer().structure([{"text": "NPWP:12.345.678.9-012.345", "confidence": 0.9}])["fields"]
    assert fields["nomor_npwp"]["value"] == "123456789012345"


def test_rule_based_falls_back_to_patterns_for_unlabeled_lines():
    document = RuleBasedNpwpStructurer().structure(
        [{"text": "123456789012345", "confidence": 0.8}, {"text": "PT SINAR ABADI SEJAHTERA", "confidence": 0.7}]
    )
    assert _values(document) == {
        "nomor_npwp": "123456789012345",
        "nama": None,
        "nama_badan": "PT SINAR ABADI SEJAHTERA",
    }
    assert document["fields"]["nama"] == {"value": None, "confidence": 0.0, "source": None}


def test_nama_badan_is_not_mistaken_for_nama():
    fields = RuleBasedNpwpStructurer().structure([{"text": "NAMA BADAN : PT X Y", "confidence": 0.9}])["fields"]
    assert fields["nama"]["value"] is None
    assert fields["nama_badan"]["value"] == "PT X Y"


def test_normalize_npwp_strips_15_digits_to_plain_digits():
    assert normalize_npwp("123456789012345") == "123456789012345"
    assert normalize_npwp("12.345.678.9-012.345") == "123456789012345"
    assert normalize_npwp("garbage") == "garbage"


def test_service_rejects_blank_input():
    with pytest.raises(ServiceError) as exc:
        StructuringService(RuleBasedNpwpStructurer()).structure([{"text": "   ", "confidence": 1.0}])
    assert exc.value.status_code == 400


def test_health_lists_backend(client):
    assert client.get("/health").json()["backends"] == {"structuring": "npwp_rules", "storage": "memory"}
