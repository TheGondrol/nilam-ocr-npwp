"""The thresholds the central orchestrator sends with a request: the guardrails threshold goes to the guardrails
service, the per-field ones travel down to scoring and decide the `confidence` of the answer; without them,
the services' own defaults hold."""

import json
from dataclasses import replace

import pytest

from tests.conftest import DONE, JPEG

RID = "OCR_361701a7-ad0f-46f7-9922-8eae7c99015e"
# The central orchestrator's example request (trust probabilities of the stub: npwp 0.7296, name 0.9471).
CENTRAL = {
    "pipeline_name_sequence": json.dumps(["guardrails", "extraction", "structuring", "scoring"]),
    "guardrails_confidence_threshold": "0.3",
    "column_confidence_threshold": json.dumps({"nomor_npwp": 0.9, "nama": 0.5}),
}


def _submit(client, auth, **form):
    return client.post(
        "/v1/extract-ocr",
        headers=auth,
        data={"request_id": RID, **form},
        files={"file": ("npwp.jpg", JPEG, "image/jpeg")},
    )


def _confidences(body):
    return {name: field["confidence"] for name, field in body["data"].items()}


def test_the_central_orchestrators_thresholds_reach_guardrails_and_the_pipeline(
    client, auth, stub_guardrails, stub_extraction
):
    response = _submit(client, auth, **CENTRAL)

    assert response.status_code == 200
    assert stub_guardrails.checked[0]["threshold"] == 0.3
    assert stub_extraction.submitted[0]["column_thresholds"] == {"nomor_npwp": 0.9, "nama": 0.5}
    # nomor_npwp 0.7296 < 0.9, nama 0.9471 >= 0.5
    assert _confidences(response.json()) == {"nomor_npwp": 0, "nama": 1}


def test_without_thresholds_the_services_defaults_hold(client, auth, stub_guardrails, stub_extraction):
    response = _submit(client, auth)

    assert response.status_code == 200
    assert "threshold" not in stub_guardrails.checked[0], "guardrails keeps its own threshold"
    assert stub_extraction.submitted[0]["column_thresholds"] is None
    # FIELD_CONFIDENCE_THRESHOLD 0.5 for both
    assert _confidences(response.json()) == {"nomor_npwp": 1, "nama": 1}


def test_a_field_left_out_of_column_confidence_threshold_uses_the_default(client, auth):
    response = _submit(client, auth, column_confidence_threshold=json.dumps({"nama": 0.95}))

    assert _confidences(response.json()) == {"nomor_npwp": 1, "nama": 0}


def test_the_guardrails_threshold_alone_is_enough(client, auth, stub_guardrails):
    response = _submit(client, auth, guardrails_confidence_threshold="0.6")

    assert response.status_code == 200
    assert stub_guardrails.checked[0]["threshold"] == 0.6


@pytest.mark.parametrize(
    "form",
    [
        {"guardrails_confidence_threshold": "tinggi"},
        {"guardrails_confidence_threshold": "0"},
        {"guardrails_confidence_threshold": "1.5"},
        {"column_confidence_threshold": "{not json"},
        {"column_confidence_threshold": "[0.9, 0.5]"},
        {"column_confidence_threshold": json.dumps({"npwp": 0.9})},
        {"column_confidence_threshold": json.dumps({"nama": 2})},
        {"column_confidence_threshold": json.dumps({"nama": "tinggi"})},
    ],
)
def test_a_threshold_that_cannot_be_read_is_422_and_nothing_runs(client, auth, stub_guardrails, stub_extraction, form):
    response = _submit(client, auth, **form)

    assert response.status_code == 422
    body = response.json()
    assert (body["errors"], body["request_id"], body["pipeline_last_stage"]) == (
        "INVALID_THRESHOLD",
        RID,
        "orchestrator",
    )
    assert stub_guardrails.checked == [] and stub_extraction.submitted == []


def test_the_get_answers_with_the_thresholds_stored_with_the_job(client, auth, stub_waiter):
    stub_waiter.snapshot_outcome = replace(DONE, column_thresholds={"nomor_npwp": 0.9, "nama": 0.5})

    response = client.get(f"/v1/extract-ocr/{RID}", headers=auth)

    assert response.status_code == 200
    assert _confidences(response.json()) == {"nomor_npwp": 0, "nama": 1}


def test_the_get_of_a_job_without_thresholds_uses_the_default(client, auth, stub_waiter):
    stub_waiter.snapshot_outcome = DONE

    response = client.get(f"/v1/extract-ocr/{RID}", headers=auth)

    assert _confidences(response.json()) == {"nomor_npwp": 1, "nama": 1}
