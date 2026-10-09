"""`POST /v1/scoring-direct`: the trust model on the structuring service's output (plus the OCR and guardrails
results it depends on), synchronous, for testing this stage on its own. Nothing is recorded."""

import pytest

GUARDRAILS = {
    "passed": True,
    "reason": None,
    "document": {"verdict": "accepted", "confidence": 0.98, "n_pages": 1, "n_approve": 1, "n_reject": 0},
    "pages": [],
}
OCR = {
    "blocks": [
        {"text": "NPWP : 12.345.678.9-012.345", "confidence": 0.96, "page": 0},
        {"text": "NAMA : BUDI SANTOSO", "confidence": 0.9, "page": 0},
    ]
}
STRUCTURING = {
    "fields": {
        "nomor_npwp": {
            "value": "12.345.678.9-012.345",
            "confidence": 0.96,
            "source": "NPWP : 12.345.678.9-012.345",
            "signals": {"candidate_count": 1, "has_homoglyph": False},
        },
        "nama": {
            "value": "BUDI SANTOSO",
            "confidence": 0.9,
            "source": "NAMA : BUDI SANTOSO",
            "signals": {"name_base": "BUDI SANTOSO", "corrected": False},
        },
        "nama_badan": {"value": None, "confidence": 0.0, "source": None, "signals": None},
    },
    "flag": False,
    "flag_reason": None,
    "reject_reason": None,
}


def _post(client, auth, **body):
    return client.post("/v1/scoring-direct", headers=auth, json=body)


def test_the_structuring_output_is_scored_now_and_nothing_is_recorded(client, auth):
    response = _post(client, auth, request_id="QC_1", guardrails=GUARDRAILS, ocr=OCR, structuring=STRUCTURING)

    assert response.status_code == 200
    body = response.json()
    assert body["request_id"] == "QC_1"
    result = body["data"]
    assert set(result) == {"npwp_confidence", "name_confidence", "fields", "payload"}
    assert 0 <= result["npwp_confidence"] <= 1 and 0 <= result["name_confidence"] <= 1
    # No column_confidence_threshold: each field's confidence is the trust probability, with no threshold.
    assert result["fields"] == {
        "nomor_npwp": {
            "value": "12.345.678.9-012.345",
            "confidence": round(result["npwp_confidence"], 4),
            "threshold": None,
        },
        "nama": {"value": "BUDI SANTOSO", "confidence": round(result["name_confidence"], 4), "threshold": None},
    }
    payload = result["payload"]
    assert payload["npwp"] == "123456789012345"
    assert payload["name_base"] == "BUDI SANTOSO"
    assert payload["npwp_candidate_count"] == 1
    assert payload["avg_doc_score"] == 0.93 and payload["min_doc_score"] == 0.9
    assert payload["flag"] is False
    assert payload["guardrail_probability"] == 0.98

    assert client.get("/v1/scoring/jobs/QC_1", headers=auth).status_code == 404


def test_the_structuring_output_alone_is_enough(client, auth):
    """`ocr` and `guardrails` left out: the signals they feed are null, the model's imputer fills them."""
    response = _post(client, auth, structuring=STRUCTURING)

    assert response.status_code == 200
    payload = response.json()["data"]["payload"]
    assert (payload["avg_doc_score"], payload["min_doc_score"], payload["guardrail_probability"]) == (None, None, None)
    assert response.json()["data"]["npwp_confidence"] is not None


def test_column_confidence_threshold_decides_the_0_1_confidences(client, auth):
    response = _post(client, auth, structuring=STRUCTURING, column_confidence_threshold={"all_field": 1})

    fields = response.json()["data"]["fields"]
    assert (fields["nomor_npwp"]["confidence"], fields["nama"]["confidence"]) == (0, 0)
    assert (fields["nomor_npwp"]["threshold"], fields["nama"]["threshold"]) == (1.0, 1.0)


def test_a_null_column_threshold_keeps_that_fields_probability(client, auth):
    response = _post(client, auth, structuring=STRUCTURING, column_confidence_threshold={"all_field": 1, "nama": None})

    assert response.status_code == 200
    result = response.json()["data"]
    fields = result["fields"]
    assert (fields["nomor_npwp"]["confidence"], fields["nomor_npwp"]["threshold"]) == (0, 1.0)
    assert (fields["nama"]["confidence"], fields["nama"]["threshold"]) == (round(result["name_confidence"], 4), None)


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"structuring": {"flag": False}},
        {"structuring": STRUCTURING, "column_confidence_threshold": {"npwp": 0.9}},
    ],
)
def test_a_body_without_structuring_fields_or_with_an_unknown_threshold_key_is_422(client, auth, body):
    response = client.post("/v1/scoring-direct", headers=auth, json=body)

    assert response.status_code == 422
    assert response.json()["errors"] == "VALIDATION_ERROR"


def test_requires_the_api_key(client):
    assert client.post("/v1/scoring-direct", json={"structuring": STRUCTURING}).status_code == 401
