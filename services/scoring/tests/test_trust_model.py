"""
Trust model dari ML engineer (weights/trust_model.joblib) dan kontrak scoring-nya:

    payload 13 kunci (hasil berantai tahap sebelumnya) -> {"npwp_confidence", "name_confidence"}

Angka yang di-pin di sini adalah keluaran FILE MODEL YANG SEKARANG. Kalau test ini gagal setelah file
model diganti, itu memang yang seharusnya terjadi: periksa model barunya, lalu perbarui angkanya.
"""

import math
from typing import Any

import pytest

from src.api.v1.scoring import get_trust_model
from src.models.trust_model import FEATURES, UNDEFINED_FEATURES, TrustModel
from src.services.confidence_service import ConfidenceService

# Contoh payload persis seperti yang diberikan ML engineer.
EXAMPLE: dict[str, Any] = {
    "npwp": "123456789012000",
    "npwp_score": 0.98,
    "npwp_has_homoglyph": False,
    "npwp_candidate_count": 1,
    "name": "PT CONTOH INDONESIA",
    "name_score": 0.96,
    "name_corrected": True,
    "n_boxes": 34,
    "num_pages": 1,
    "avg_doc_score": 0.912,
    "min_doc_score": 0.62,
    "flag": False,
    "guardrail_probability": 0.9821,
}


@pytest.fixture(scope="module")
def model() -> TrustModel:
    return get_trust_model()


def test_bundle_is_what_the_file_says_it_is(model):
    assert model.metadata["steps"] == ["SimpleImputer", "StandardScaler", "LogisticRegression"]
    assert tuple(model.metadata["features"]) == FEATURES
    report = model.metadata["train_report"]
    assert (report["n"], report["n_correct"], report["n_incorrect"]) == (1760, 1640, 120)
    assert report["auc"] == pytest.approx(0.9637, abs=1e-4)


def test_payload_maps_to_one_feature_row_per_field(model):
    rows = model.feature_rows(EXAMPLE)
    # field_is_npwp, field_score, field_corrected, shape_confidence, field_multiple_candidates,
    # n_boxes_per_page, avg_doc_score, min_doc_score, flag, guardrail_probability
    npwp, name = rows["npwp"], rows["name"]
    assert npwp[:3] == [1.0, 0.98, 0.0]  # npwp_score, npwp_has_homoglyph
    assert name[:3] == [0.0, 0.96, 1.0]  # name_score, name_corrected
    assert npwp[4] == name[4] == 0.0  # npwp_candidate_count 1 -> bukan "multiple"; satu nilai untuk kedua baris
    assert npwp[5:] == name[5:] == [34.0, 0.912, 0.62, 0.0, 0.9821]  # n_boxes / num_pages, lalu kunci bernama sama


def test_shape_confidence_is_left_to_the_models_own_imputer(model):
    """Satu-satunya fitur yang tidak ada di payload dan tidak terdefinisi di file mana pun: dikirim kosong."""
    assert UNDEFINED_FEATURES == ("shape_confidence",)
    index = FEATURES.index("shape_confidence")
    rows = model.feature_rows(EXAMPLE)
    assert math.isnan(rows["npwp"][index]) and math.isnan(rows["name"][index])


def test_example_payload_gives_two_confidences(model):
    result = model.predict(EXAMPLE)
    assert set(result) == {"npwp_confidence", "name_confidence"}
    assert result["npwp_confidence"] == pytest.approx(0.9737, abs=1e-4)
    # name_corrected=true: di data training field yang dikoreksi hampir selalu salah, jadi model menjatuhkannya.
    assert result["name_confidence"] == pytest.approx(0.001, abs=1e-3)


def test_uncorrected_name_is_trusted(model):
    assert model.predict(EXAMPLE | {"name_corrected": False})["name_confidence"] == pytest.approx(0.9136, abs=1e-4)


def test_homoglyph_in_the_number_destroys_its_confidence(model):
    assert model.predict(EXAMPLE | {"npwp_has_homoglyph": True})["npwp_confidence"] < 0.01


def test_multiple_candidates_lower_the_confidence(model):
    one = model.predict(EXAMPLE)["npwp_confidence"]
    two = model.predict(EXAMPLE | {"npwp_candidate_count": 2})["npwp_confidence"]
    assert two == pytest.approx(0.8543, abs=1e-4)
    assert two < one


def test_missing_field_has_no_confidence(model):
    """Tanpa ini imputer mengisi skor OCR field KOSONG dengan median training (0.998) -> confidence tinggi."""
    result = model.predict(EXAMPLE | {"name": None, "name_score": None, "name_corrected": None})
    assert result["name_confidence"] is None
    assert result["npwp_confidence"] == pytest.approx(0.9737, abs=1e-4)
    assert model.predict(EXAMPLE | {"npwp": "  "})["npwp_confidence"] is None


def test_missing_inputs_are_imputed_not_an_error(model):
    sparse = {"npwp": "123456789012000", "npwp_score": 0.98, "name": "BUDI", "name_score": 0.97}
    result = model.predict(sparse)
    assert 0 <= result["npwp_confidence"] <= 1 and 0 <= result["name_confidence"] <= 1


def test_missing_model_file_fails_at_load():
    with pytest.raises(RuntimeError, match="Trust model not found"):
        TrustModel("weights/tidak-ada.joblib")


def test_foreign_bundle_is_refused(tmp_path):
    import joblib

    path = tmp_path / "lain.joblib"
    joblib.dump({"pipeline": object(), "feature_cols": ["a", "b"]}, path)
    with pytest.raises(RuntimeError, match="Unexpected feature_cols"):
        TrustModel(str(path))


# ---------------------------------------------------------------------------
# Payload dari hasil berantai (pipeline)
# ---------------------------------------------------------------------------

GUARDRAILS = {
    "passed": True,
    "reason": None,
    "document": {"verdict": "accepted", "confidence": 0.7595, "n_pages": 1, "n_approve": 1, "n_reject": 0},
    "pages": [{"page_index": 0, "proba_approve": 0.7595, "proba_reject": 0.2405, "verdict": "accepted"}],
}
OCR = {
    "engine": "paddle",
    "blocks": [
        {"text": "npwp", "confidence": 0.9847, "page": 0},
        {"text": "95.844.800.1-805.000", "confidence": 1.0, "page": 0},
        {"text": "RAHMAT HIDAYAT", "confidence": 0.9931, "page": 0},
        {"text": "NPWP16:4318 0856 07040052", "confidence": 0.9762, "page": 0},
    ],
}
STRUCTURING = {
    "document_type": "npwp",
    "fields": {
        "nomor_npwp": {
            "value": "95.844.800.1-805.000",
            "confidence": 1.0,
            "source": "95.844.800.1-805.000",
            "signals": {"has_homoglyph": False, "candidate_count": 2},
        },
        "nama": {
            "value": "RAHMAT HIDAYAT",
            "confidence": 0.9931,
            "source": "RAHMAT HIDAYAT",
            "signals": {"corrected": False},
        },
        "nama_badan": {"value": None, "confidence": 0.0, "source": None, "signals": None},
    },
}


def test_payload_is_built_from_the_chained_results():
    payload = ConfidenceService.payload_from_chain(GUARDRAILS, OCR, STRUCTURING)
    assert payload == {
        "npwp": "958448001805000",  # digit saja, seperti contoh ML engineer
        "npwp_score": 1.0,
        "npwp_has_homoglyph": False,
        "npwp_candidate_count": 2,
        "name": "RAHMAT HIDAYAT",
        "name_score": 0.9931,
        "name_corrected": False,
        "n_boxes": 4,
        "num_pages": 1,
        "avg_doc_score": pytest.approx(0.9885),
        "min_doc_score": 0.9762,
        "flag": None,  # belum ada tahap yang menghasilkannya
        "guardrail_probability": 0.7595,
    }
    assert set(payload) == set(EXAMPLE)  # persis kunci kontrak ML engineer, tidak lebih tidak kurang


def test_company_name_is_the_name_of_the_payload():
    structuring = {
        "fields": {
            "nomor_npwp": {"value": "01.234.567.8-091.000", "confidence": 0.99},
            "nama": {"value": None, "confidence": 0.0},
            "nama_badan": {"value": "PT CONTOH INDONESIA", "confidence": 0.96, "signals": {"corrected": False}},
        }
    }
    payload = ConfidenceService.payload_from_chain(None, None, structuring)
    assert (payload["name"], payload["name_score"]) == ("PT CONTOH INDONESIA", 0.96)
    # Tanpa hasil OCR / guardrails: kosong, bukan nol (nol adalah nilai yang sah dan berarti lain).
    assert payload["n_boxes"] is None and payload["avg_doc_score"] is None and payload["guardrail_probability"] is None


def test_structurer_without_signals_leaves_them_empty():
    """Backend structuring `rule_based` tidak mengeluarkan signals: kosong -> diisi imputer, bukan False."""
    structuring = {"fields": {"nomor_npwp": {"value": "01.234.567.8-091.000", "confidence": 0.99}}}
    payload = ConfidenceService.payload_from_chain(None, None, structuring)
    assert payload["npwp_has_homoglyph"] is None and payload["npwp_candidate_count"] is None


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------


def test_http_confidence_follows_the_ml_contract(client, auth):
    response = client.post("/v1/scoring/confidence", json=EXAMPLE, headers=auth)
    assert response.status_code == 200, response.text
    data = response.json()["data"]
    assert set(data) == {"npwp_confidence", "name_confidence"}
    assert data["npwp_confidence"] == pytest.approx(0.9737, abs=1e-4)


def test_http_confidence_accepts_nulls(client, auth):
    response = client.post("/v1/scoring/confidence", json={"npwp": "123456789012000", "npwp_score": 0.9}, headers=auth)
    assert response.status_code == 200
    assert response.json()["data"]["name_confidence"] is None


def test_http_confidence_rejects_bad_types(client, auth):
    response = client.post("/v1/scoring/confidence", json=EXAMPLE | {"npwp_score": "tinggi"}, headers=auth)
    assert response.status_code == 422
    assert response.json()["errors"] == "VALIDATION_ERROR"


def test_http_confidence_requires_api_key(client):
    assert client.post("/v1/scoring/confidence", json=EXAMPLE).status_code == 401


def test_health_names_the_model(client):
    backends = client.get("/health").json()["backends"]
    assert backends["scoring"] == "trust_model"
    assert backends["legacy_score"] == "heuristic"
