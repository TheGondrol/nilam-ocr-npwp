"""
Tahap STRUCTURING di pipeline async lewat HTTP: 202 segera, kerja di
background, callback, lalu handoff ke scoring. Orkestrasi dan scoring diganti
perekam (tanpa jaringan); rantai sungguhan diuji scripts/smoke_e2e.py.
"""

import pytest

from ocr_common.jobs import STAGE_STRUCTURING, InMemoryJobRepository, StagePipeline
from ocr_common.testing import RecordingCallback, RecordingNextStage, make_client, wait_for_job
from src.api.v1.jobs import get_job_service
from src.api.v1.structuring import get_structuring_service
from src.main import app
from src.services.job_service import StructuringJobService

GUARDRAILS = {"passed": True, "reason": None}
OCR = {
    "engine": "mock",
    "model": None,
    "full_text": "NPWP : 12.345.678.9-012.345\nNAMA : BUDI SANTOSO",
    "blocks": [
        {"text": "NPWP : 12.345.678.9-012.345", "confidence": 0.96, "bbox": None, "page": 0},
        {"text": "NAMA : BUDI SANTOSO", "confidence": 0.95, "bbox": None, "page": 0},
    ],
}


@pytest.fixture
def harness():
    callback, next_stage = RecordingCallback(), RecordingNextStage()
    pipeline = StagePipeline(stage=STAGE_STRUCTURING, repository=InMemoryJobRepository(), callback=callback)
    service = StructuringJobService(pipeline, get_structuring_service(), next_stage)
    app.dependency_overrides[get_job_service] = lambda: service
    # Context manager: satu event loop selama test, supaya task background hidup.
    with make_client(app) as client:
        yield client, callback, next_stage
    app.dependency_overrides.pop(get_job_service, None)


def _payload(request_id, ocr=OCR):
    return {"request_id": request_id, "document_type": "npwp", "guardrails": GUARDRAILS, "ocr": ocr}


def test_submit_returns_202_then_structures_callback_and_handoff(harness, auth):
    client, callback, next_stage = harness

    response = client.post("/v1/structuring/jobs", headers=auth, json=_payload("REQ_1"))
    assert response.status_code == 202
    assert response.json()["request_id"] == "REQ_1"
    assert response.json()["data"] == {
        "request_id": "REQ_1",
        "stage": "STRUCTURING",
        "status": "PROCESSING",
        "duplicate": False,
    }

    job = wait_for_job(client, "/v1/structuring/jobs/REQ_1")
    assert job["status"] == "DONE"
    assert job["result"]["fields"]["nomor_npwp"]["value"] == "12.345.678.9-012.345"
    assert job["result"]["fields"]["nama"]["value"] == "BUDI SANTOSO"

    assert [(c["stage"], c["status"], c["result"]) for c in callback.calls] == [("STRUCTURING", "DONE", None)]
    # Payload ke scoring: hasil guardrails + OCR (utuh, termasuk bbox/page) + structuring.
    assert next_stage.payloads == [{**_payload("REQ_1"), "structuring": job["result"]}]


def test_same_request_id_is_not_processed_twice(harness, auth):
    client, callback, next_stage = harness
    client.post("/v1/structuring/jobs", headers=auth, json=_payload("REQ_2"))
    wait_for_job(client, "/v1/structuring/jobs/REQ_2")

    again = client.post("/v1/structuring/jobs", headers=auth, json=_payload("REQ_2"))
    assert again.status_code == 202
    assert again.json()["data"]["duplicate"] is True
    assert again.json()["data"]["status"] == "DONE"
    assert len(callback.calls) == 1
    assert len(next_stage.payloads) == 1


def test_no_text_fails_the_job_and_skips_handoff(harness, auth):
    client, callback, next_stage = harness
    response = client.post("/v1/structuring/jobs", headers=auth, json=_payload("REQ_3", {"blocks": []}))
    assert response.status_code == 202

    job = wait_for_job(client, "/v1/structuring/jobs/REQ_3")
    assert (job["status"], job["error_message"]) == ("FAILED", "No text lines to structure")
    assert [(c["stage"], c["status"], c["error_message"]) for c in callback.calls] == [
        ("STRUCTURING", "FAILED", "No text lines to structure")
    ]
    assert next_stage.payloads == []


def test_missing_ocr_is_422(harness, auth):
    client, _, _ = harness
    response = client.post("/v1/structuring/jobs", headers=auth, json={"request_id": "REQ_4"})
    assert response.status_code == 422
    assert response.json()["errors"] == "VALIDATION_ERROR"


def test_get_unknown_job_is_404(harness, auth):
    client, _, _ = harness
    assert client.get("/v1/structuring/jobs/REQ_missing", headers=auth).status_code == 404


def test_requires_api_key(harness):
    client, _, _ = harness
    assert client.post("/v1/structuring/jobs", json=_payload("REQ_5")).status_code == 401
