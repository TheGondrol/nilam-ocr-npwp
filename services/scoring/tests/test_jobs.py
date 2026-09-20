"""
Tahap SCORING di pipeline async lewat HTTP: 202 segera, kerja di background,
lalu callback yang membawa hasil akhir. Orkestrasi diganti perekam (tanpa
jaringan); rantai sungguhan diuji scripts/smoke_e2e.py.
"""

import pytest

from ocr_common.jobs import STAGE_SCORING, InMemoryJobRepository, StagePipeline
from ocr_common.testing import RecordingCallback, make_client, wait_for_job
from src.api.v1.jobs import get_job_service
from src.api.v1.scoring import get_confidence_service
from src.main import app
from src.services.job_service import ScoringJobService

GUARDRAILS = {"passed": True, "reason": None}
STRUCTURING = {
    "document_type": "npwp",
    "fields": {
        "nomor_npwp": {
            "value": "12.345.678.9-012.345",
            "confidence": 0.96,
            "source": "NPWP : 12.345.678.9-012.345",
        },
        "nama": {"value": "BUDI SANTOSO", "confidence": 0.95, "source": "NAMA : BUDI SANTOSO"},
        "nama_badan": {"value": None, "confidence": 0.0, "source": None},
    },
}


@pytest.fixture
def harness():
    callback = RecordingCallback()
    pipeline = StagePipeline(stage=STAGE_SCORING, repository=InMemoryJobRepository(), callback=callback)
    service = ScoringJobService(pipeline, get_confidence_service())
    app.dependency_overrides[get_job_service] = lambda: service
    # Context manager: satu event loop selama test, supaya task background hidup.
    with make_client(app) as client:
        yield client, callback
    app.dependency_overrides.pop(get_job_service, None)


def _payload(request_id, document_type="npwp"):
    return {
        "request_id": request_id,
        "document_type": document_type,
        "guardrails": GUARDRAILS,
        "ocr": {"blocks": [{"text": "NPWP : 12.345.678.9-012.345", "confidence": 0.96, "page": 0}]},
        "structuring": STRUCTURING,
    }


def test_submit_returns_202_then_scores_and_sends_final_result(harness, auth):
    client, callback = harness

    response = client.post("/v1/scoring/jobs", headers=auth, json=_payload("REQ_1"))
    assert response.status_code == 202
    assert response.json()["data"] == {
        "request_id": "REQ_1",
        "stage": "SCORING",
        "status": "PROCESSING",
        "duplicate": False,
    }

    job = wait_for_job(client, "/v1/scoring/jobs/REQ_1")
    assert job["status"] == "DONE"
    result = job["result"]
    # Keluaran ML engineer: dua confidence per field. Tidak ada skor dokumen / keputusan.
    assert set(result) == {"npwp_confidence", "name_confidence", "payload"}
    assert 0 <= result["npwp_confidence"] <= 1 and 0 <= result["name_confidence"] <= 1
    # Payload yang dinilai model ikut tersimpan, disusun dari hasil berantai tahap sebelumnya.
    assert result["payload"]["npwp"] == "123456789012345"
    assert result["payload"]["name"] == "BUDI SANTOSO"
    assert result["payload"]["n_boxes"] == 1
    assert result["payload"]["guardrail_probability"] is None  # GUARDRAILS di test ini tanpa `document`

    # Tahap terakhir: callback membawa hasil akhir untuk requests.final_result.
    assert len(callback.calls) == 1
    call = callback.calls[0]
    assert (call["stage"], call["status"]) == ("SCORING", "DONE")
    assert call["result"] == {
        "document_type": "npwp",
        "fields": {
            "nomor_npwp": {"value": "12.345.678.9-012.345", "confidence": 0.96},
            "nama": {"value": "BUDI SANTOSO", "confidence": 0.95},
            "nama_badan": {"value": None, "confidence": 0.0},
        },
        "scoring": {"npwp_confidence": result["npwp_confidence"], "name_confidence": result["name_confidence"]},
        "guardrails": GUARDRAILS,
    }


def test_same_request_id_is_not_processed_twice(harness, auth):
    client, callback = harness
    client.post("/v1/scoring/jobs", headers=auth, json=_payload("REQ_2"))
    wait_for_job(client, "/v1/scoring/jobs/REQ_2")

    again = client.post("/v1/scoring/jobs", headers=auth, json=_payload("REQ_2"))
    assert again.status_code == 202
    assert again.json()["data"]["duplicate"] is True
    assert len(callback.calls) == 1


def test_unsupported_document_type_fails_the_job(harness, auth):
    client, callback = harness
    response = client.post("/v1/scoring/jobs", headers=auth, json=_payload("REQ_3", document_type="ktp"))
    assert response.status_code == 202

    job = wait_for_job(client, "/v1/scoring/jobs/REQ_3")
    assert job["status"] == "FAILED"
    assert job["error_message"].startswith("Unsupported document_type: ktp")
    assert [(c["stage"], c["status"], c["result"]) for c in callback.calls] == [("SCORING", "FAILED", None)]


def test_missing_structuring_is_422(harness, auth):
    client, _ = harness
    response = client.post("/v1/scoring/jobs", headers=auth, json={"request_id": "REQ_4"})
    assert response.status_code == 422
    assert response.json()["errors"] == "VALIDATION_ERROR"


def test_get_unknown_job_is_404(harness, auth):
    client, _ = harness
    assert client.get("/v1/scoring/jobs/REQ_missing", headers=auth).status_code == 404


def test_requires_api_key(harness):
    client, _ = harness
    assert client.post("/v1/scoring/jobs", json=_payload("REQ_5")).status_code == 401
