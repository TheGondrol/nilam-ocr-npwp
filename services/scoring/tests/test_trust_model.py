import math
from typing import Any

import pytest

from app.dependencies import get_trust_model
from app.ml.trust_model import FEATURES, PAYLOAD_KEYS, TrustModel
from app.services.confidence_service import ConfidenceService

# The example payload of the ML team's scoring README (nilamnpwp scoring/README.md, 23 Sep 2026).
EXAMPLE: dict[str, Any] = {
    "npwp": "12.345.678.9-012.000",
    "npwp_score": 0.98,
    "npwp_candidate_count": 1,
    "name_base": "PT CONTOH INDONESIA",
    "name_score": 0.95,
    "avg_doc_score": 0.91,
    "min_doc_score": 0.62,
    "flag": False,
    "guardrail_probability": 0.98,
}


@pytest.fixture(scope="module")
def model() -> TrustModel:
    return get_trust_model()


def test_bundle_is_the_21sep_retrain(model):
    assert model.metadata["steps"] == ["SimpleImputer", "StandardScaler", "LogisticRegression"]
    assert tuple(model.metadata["features"]) == FEATURES
    report = model.metadata["train_report"]
    assert (report["n"], report["n_correct"], report["n_incorrect"]) == (1860, 1738, 122)
    assert report["auc"] == pytest.approx(0.9303, abs=1e-4)


def test_payload_keys_are_the_ml_contract():
    assert set(EXAMPLE) == set(PAYLOAD_KEYS)


def test_payload_maps_to_one_feature_row_per_field(model):
    rows = model.feature_rows(EXAMPLE)
    npwp, name = rows["npwp"], rows["name"]
    # field_is_npwp, field_score, shape_confidence, field_multiple_candidates, name_max_char_len
    assert npwp[:4] == [1.0, 0.98, 1.0, 0.0] and math.isnan(npwp[4])
    assert name[:3] == [0.0, 0.95, 2.0] and math.isnan(name[3]) and name[4] == 9.0
    # avg_doc_score, min_doc_score, flag, guardrail_probability
    assert npwp[5:] == name[5:] == [0.91, 0.62, 0.0, 0.98]


def test_shape_confidence_tiers(model):
    assert model.feature_rows(EXAMPLE | {"npwp": "3201234567890001"})["npwp"][2] == 2.0
    assert model.feature_rows(EXAMPLE | {"npwp": "6348341551000"})["npwp"][2] == 0.0
    assert model.feature_rows(EXAMPLE | {"name_base": "SUKIRMAN"})["name"][2] == 1.0
    assert model.feature_rows(EXAMPLE | {"name_base": None})["name"][2] == 0.0


def test_example_payload_gives_two_confidences(model):
    # The README of the ML team shows 0.93 / 0.88 for this payload; their own trust_scoring.py gives
    # these values with the 21 Sep model, so the README numbers are an illustration of the shape only.
    result = model.predict(EXAMPLE)
    assert set(result) == {"npwp_confidence", "name_confidence"}
    assert result["npwp_confidence"] == pytest.approx(0.3283, abs=1e-4)
    assert result["name_confidence"] == pytest.approx(0.0406, abs=1e-4)


def test_a_16_digit_number_is_trusted_more_than_a_15_digit_one(model):
    assert model.predict(EXAMPLE | {"npwp": "3201234567890001"})["npwp_confidence"] == pytest.approx(0.9038, abs=1e-4)


def test_a_number_that_lost_a_digit_is_not_trusted(model):
    assert model.predict(EXAMPLE | {"npwp": "6348341551000"})["npwp_confidence"] < 0.05


def test_multiple_candidates_lower_the_confidence(model):
    one = model.predict(EXAMPLE)["npwp_confidence"]
    two = model.predict(EXAMPLE | {"npwp_candidate_count": 2})["npwp_confidence"]
    assert two == pytest.approx(0.1328, abs=1e-4)
    assert two < one


def test_a_single_word_name_and_a_merged_name_are_not_trusted(model):
    assert model.predict(EXAMPLE | {"name_base": "SUKIRMAN"})["name_confidence"] < 0.01
    assert model.predict(EXAMPLE | {"name_base": "MUHAMADRINOSUKIRMA"})["name_confidence"] < 0.001


def test_the_flag_is_a_feature_not_a_veto(model):
    flagged = model.predict(EXAMPLE | {"flag": True})
    assert flagged["npwp_confidence"] == pytest.approx(0.4447, abs=1e-4)
    assert 0 < flagged["name_confidence"] < 1


def test_missing_field_has_no_confidence(model):
    result = model.predict(EXAMPLE | {"name_base": None, "name_score": None})
    assert result["name_confidence"] is None
    assert result["npwp_confidence"] == pytest.approx(0.3283, abs=1e-4)
    assert model.predict(EXAMPLE | {"npwp": "  "})["npwp_confidence"] is None


def test_missing_inputs_are_imputed_not_an_error(model):
    sparse = {"npwp": "123456789012000", "npwp_score": 0.98, "name_base": "BUDI", "name_score": 0.97}
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
            "value": "4318085607040052",
            "confidence": 0.9762,
            "source": "NPWP16:4318 0856 07040052",
            "signals": {"candidate_count": 2, "has_homoglyph": False, "invalid_province_prefix": True},
        },
        "nama": {
            "value": "RATNA SUSANTI / SYAMARIS",
            "confidence": 0.9931,
            "source": "RATNA SUSANTI/SYAMARIS",
            "signals": {"name_base": "RATNA SUSANTI/SYAMARIS", "corrected": True},
        },
        "nama_badan": {"value": None, "confidence": 0.0, "source": None, "signals": None},
    },
    "flag": True,
    "flag_reason": "Kode provinsi pada NPWP tidak valid, mohon dicek kembali",
}


def test_payload_is_built_from_the_chained_results():
    payload = ConfidenceService.payload_from_chain(GUARDRAILS, OCR, STRUCTURING)
    assert payload == {
        "npwp": "4318085607040052",
        "npwp_score": 0.9762,
        "npwp_candidate_count": 2,
        "name_base": "RATNA SUSANTI/SYAMARIS",
        "name_score": 0.9931,
        "avg_doc_score": pytest.approx(0.9885),
        "min_doc_score": 0.9762,
        "flag": True,
        "guardrail_probability": 0.7595,
    }
    assert set(payload) == set(PAYLOAD_KEYS)


def test_company_name_is_the_name_of_the_payload():
    structuring = {
        "fields": {
            "nomor_npwp": {"value": "01.234.567.8-091.000", "confidence": 0.99},
            "nama": {"value": None, "confidence": 0.0},
            "nama_badan": {"value": "PT CONTOH INDONESIA", "confidence": 0.96, "signals": {"corrected": False}},
        }
    }
    payload = ConfidenceService.payload_from_chain(None, None, structuring)
    assert (payload["name_base"], payload["name_score"]) == ("PT CONTOH INDONESIA", 0.96)
    assert payload["npwp"] == "012345678091000"
    assert payload["flag"] is False
    assert payload["avg_doc_score"] is None and payload["guardrail_probability"] is None


def test_a_rejected_guardrails_report_gives_no_guardrail_probability():
    rejected = {"document": {"verdict": "reject", "confidence": 0.88}}
    assert ConfidenceService.payload_from_chain(rejected, None, STRUCTURING)["guardrail_probability"] is None


def test_structurer_without_signals_leaves_the_candidate_count_empty():
    structuring = {"fields": {"nomor_npwp": {"value": "01.234.567.8-091.000", "confidence": 0.99}}}
    payload = ConfidenceService.payload_from_chain(None, None, structuring)
    assert payload["npwp_candidate_count"] is None and payload["name_base"] is None


def test_http_confidence_follows_the_ml_contract(client, auth):
    response = client.post("/v1/scoring/confidence", json=EXAMPLE, headers=auth)
    assert response.status_code == 200, response.text
    data = response.json()["data"]
    assert set(data) == {"npwp_confidence", "name_confidence"}
    assert data["npwp_confidence"] == pytest.approx(0.3283, abs=1e-4)


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
