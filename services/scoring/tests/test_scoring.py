import pytest

from ocr_common.errors import ServiceError
from src.core.config import Settings
from src.models.scoring import HeuristicNpwpScorer
from src.services.scoring_service import ScoringService

SETTINGS = Settings(api_key="x", _env_file=None)  # approve >= 0.8, review >= 0.5

FULL = {
    "nomor_npwp": {"value": "12.345.678.9-012.345", "confidence": 0.96},
    "nama": {"value": "BUDI SANTOSO", "confidence": 0.95},
    "nama_badan": {"value": "PT CIPTA KARYA MANDIRI", "confidence": 0.93},
}


def _service():
    return ScoringService(HeuristicNpwpScorer(), SETTINGS)


def test_complete_valid_document_is_approved():
    report = _service().score("npwp", FULL)
    assert report["decision"] == "approve"
    assert report["reasons"] == []
    assert all(not s["issues"] for s in report["field_scores"])


def test_personal_npwp_without_nama_badan_is_still_approved():
    fields = {k: v for k, v in FULL.items() if k != "nama_badan"}
    report = _service().score("npwp", fields)
    assert report["decision"] == "approve"
    assert report["reasons"] == []


def test_missing_both_names_lowers_score_and_explains():
    report = _service().score("npwp", {"nomor_npwp": FULL["nomor_npwp"]})
    assert report["decision"] == "reject"
    assert any("nama, nama_badan" in reason for reason in report["reasons"])


def test_missing_nomor_npwp_is_reported():
    report = _service().score("npwp", {"nama": FULL["nama"]})
    assert any(reason.startswith("required field nomor_npwp") for reason in report["reasons"])


def test_invalid_npwp_format_scores_zero_for_that_field():
    report = _service().score("npwp", {**FULL, "nomor_npwp": {"value": "12.34", "confidence": 0.99}})
    npwp = next(s for s in report["field_scores"] if s["name"] == "nomor_npwp")
    assert npwp["score"] == 0.0
    assert npwp["issues"][0].startswith("invalid format")


def test_unsupported_document_type():
    with pytest.raises(ServiceError) as exc:
        _service().score("ktp", {"nik": {"value": "1", "confidence": 1.0}})
    assert exc.value.status_code == 400


def test_health_lists_backend(client):
    assert client.get("/health").json()["backends"] == {
        "scoring": "trust_model",
        "legacy_score": "heuristic",
        "storage": "memory",
    }


def test_http_score(client, auth):
    response = client.post("/v1/scoring/score", json={"document_type": "npwp", "fields": FULL}, headers=auth)
    assert response.status_code == 200
    data = response.json()["data"]
    assert data["decision"] == "approve"
    assert 0.9 <= data["score"] <= 1.0


def test_http_score_empty_fields_returns_400(client, auth):
    response = client.post("/v1/scoring/score", json={"fields": {}}, headers=auth)
    assert response.status_code == 400
