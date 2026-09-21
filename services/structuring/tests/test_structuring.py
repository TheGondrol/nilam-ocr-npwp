import pytest

from ocr_common.errors import ServiceError
from src.models.structuring import RuleBasedNpwpStructurer, normalize_npwp
from src.services.structuring_service import StructuringService

LINES = [
    {"text": "KEMENTERIAN KEUANGAN REPUBLIK INDONESIA", "confidence": 0.99},
    {"text": "NPWP : 12.345.678.9-012.345", "confidence": 0.96},
    {"text": "NAMA : BUDI SANTOSO", "confidence": 0.95},
    {"text": "NAMA BADAN : PT CIPTA KARYA MANDIRI", "confidence": 0.93},
]


def _values(fields):
    return {name: f["value"] for name, f in fields.items()}


def test_rule_based_parses_labeled_lines():
    fields = RuleBasedNpwpStructurer().structure(LINES)
    assert _values(fields) == {
        "nomor_npwp": "12.345.678.9-012.345",
        "nama": "BUDI SANTOSO",
        "nama_badan": "PT CIPTA KARYA MANDIRI",
    }
    assert fields["nomor_npwp"]["confidence"] == 0.96
    assert fields["nomor_npwp"]["source"] == "NPWP : 12.345.678.9-012.345"


def test_rule_based_handles_paddle_style_lines_without_spaces():
    fields = RuleBasedNpwpStructurer().structure([{"text": "NPWP:12.345.678.9-012.345", "confidence": 0.9}])
    assert fields["nomor_npwp"]["value"] == "12.345.678.9-012.345"


def test_rule_based_falls_back_to_patterns_for_unlabeled_lines():
    fields = RuleBasedNpwpStructurer().structure(
        [{"text": "123456789012345", "confidence": 0.8}, {"text": "PT SINAR ABADI SEJAHTERA", "confidence": 0.7}]
    )
    assert _values(fields) == {
        "nomor_npwp": "12.345.678.9-012.345",
        "nama": None,
        "nama_badan": "PT SINAR ABADI SEJAHTERA",
    }
    assert fields["nama"] == {"value": None, "confidence": 0.0, "source": None}


def test_nama_badan_is_not_mistaken_for_nama():
    fields = RuleBasedNpwpStructurer().structure([{"text": "NAMA BADAN : PT X Y", "confidence": 0.9}])
    assert fields["nama"]["value"] is None
    assert fields["nama_badan"]["value"] == "PT X Y"


def test_normalize_npwp_formats_15_digits():
    assert normalize_npwp("123456789012345") == "12.345.678.9-012.345"
    assert normalize_npwp("12.345.678.9-012.345") == "12.345.678.9-012.345"
    assert normalize_npwp("garbage") == "garbage"


def test_service_rejects_blank_input():
    with pytest.raises(ServiceError) as exc:
        StructuringService(RuleBasedNpwpStructurer()).structure([{"text": "   ", "confidence": 1.0}])
    assert exc.value.status_code == 400


def test_health_lists_backend(client):
    assert client.get("/health").json()["backends"] == {"structuring": "npwp_rules", "storage": "memory"}


def test_http_structure(client, auth):
    response = client.post("/v1/structuring/structure", json={"lines": LINES}, headers=auth)
    assert response.status_code == 200
    data = response.json()["data"]
    assert data["document_type"] == "npwp"
    assert data["fields"]["nama"]["value"] == "BUDI SANTOSO"


def test_http_structure_blank_lines_returns_400(client, auth):
    response = client.post("/v1/structuring/structure", json={"lines": [{"text": " "}]}, headers=auth)
    assert response.status_code == 400
    assert response.json()["message"] == "No text lines to structure"


def test_validation_error_uses_envelope_with_code(client, auth):
    response = client.post("/v1/structuring/structure", json={"lines": "not-a-list"}, headers=auth)
    assert response.status_code == 422
    body = response.json()
    assert body["errors"] == "VALIDATION_ERROR"
    assert body["message"].startswith("body.lines:")
