import pytest

from ocr_common.pipeline import STAGE_SCORING, InMemoryJobRepository, StagePipeline
from ocr_common.testing import RecordingCallback, make_client, wait_for_job

from app.dependencies import get_confidence_service, get_job_service
from app.main import app
from app.services.job_service import ScoringJobService

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
    "flag": True,
    "flag_reason": "Nama hanya terdiri dari 1 kata, mohon dicek kembali",
}


@pytest.fixture
def harness():
    callback = RecordingCallback()
    pipeline = StagePipeline(stage=STAGE_SCORING, repository=InMemoryJobRepository(), callback=callback)
    service = ScoringJobService(pipeline, get_confidence_service())
    app.dependency_overrides[get_job_service] = lambda: service
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
    assert set(result) == {"npwp_confidence", "name_confidence", "payload"}
    assert 0 <= result["npwp_confidence"] <= 1 and 0 <= result["name_confidence"] <= 1
    assert result["payload"]["npwp"] == "123456789012345"
    assert result["payload"]["name_base"] == "BUDI SANTOSO"
    assert result["payload"]["avg_doc_score"] == 0.96
    assert result["payload"]["flag"] is True
    assert result["payload"]["guardrail_probability"] is None

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
        "flag": True,
        "flag_reason": "Nama hanya terdiri dari 1 kata, mohon dicek kembali",
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


def test_missing_structuring_without_a_database_is_422(harness, auth):
    client, _ = harness
    response = client.post("/v1/scoring/jobs", headers=auth, json={"request_id": "REQ_4"})
    assert response.status_code == 422
    assert "structuring is missing" in response.json()["message"]


def test_get_unknown_job_is_404(harness, auth):
    client, _ = harness
    assert client.get("/v1/scoring/jobs/REQ_missing", headers=auth).status_code == 404


def test_requires_api_key(harness):
    client, _ = harness
    assert client.post("/v1/scoring/jobs", json=_payload("REQ_5")).status_code == 401


class FakeResults:
    """Stands in for the shared database: {stage_prefix: {request_id: result}}."""

    def __init__(self, **stored):
        self.stored = stored

    async def get(self, stage_prefix, request_id):
        return self.stored.get(stage_prefix, {}).get(request_id)


def test_by_reference_reads_structuring_and_ocr_from_the_database(auth):
    callback = RecordingCallback()
    pipeline = StagePipeline(stage=STAGE_SCORING, repository=InMemoryJobRepository(), callback=callback)
    ocr = {"blocks": [{"text": "NPWP : 12.345.678.9-012.345", "confidence": 0.96, "page": 0}]}
    results = FakeResults(structuring={"REQ_ref": STRUCTURING}, ocr={"REQ_ref": ocr})
    service = ScoringJobService(pipeline, get_confidence_service(), results=results)
    app.dependency_overrides[get_job_service] = lambda: service
    try:
        with make_client(app) as client:
            body = {"request_id": "REQ_ref", "document_type": "npwp", "guardrails": GUARDRAILS}
            assert client.post("/v1/scoring/jobs", headers=auth, json=body).status_code == 202
            job = wait_for_job(client, "/v1/scoring/jobs/REQ_ref")
    finally:
        app.dependency_overrides.pop(get_job_service, None)

    assert job["status"] == "DONE"
    assert job["result"]["payload"]["npwp"] == "123456789012345"
    assert job["result"]["payload"]["avg_doc_score"] == 0.96, "the OCR blocks were read from the database too"
    [call] = callback.calls
    assert call["result"]["fields"]["nomor_npwp"]["value"] == "12.345.678.9-012.345"
    assert call["result"]["guardrails"] == GUARDRAILS


async def test_a_stale_job_is_run_again_from_what_the_database_holds():
    callback = RecordingCallback()
    pipeline = StagePipeline(stage=STAGE_SCORING, repository=InMemoryJobRepository(), callback=callback)
    service = ScoringJobService(
        pipeline, get_confidence_service(), results=FakeResults(structuring={"REQ_stale": STRUCTURING})
    )
    await pipeline.repository.claim("REQ_stale", input={"document_type": "npwp", "guardrails": GUARDRAILS})

    await service.resume("REQ_stale", {"document_type": "npwp", "guardrails": GUARDRAILS})
    await pipeline.runner.drain(5)

    assert (await pipeline.get("REQ_stale"))["status"] == "DONE"
    [call] = callback.calls
    assert (call["stage"], call["status"]) == ("SCORING", "DONE")
    assert call["result"]["guardrails"] == GUARDRAILS
    assert call["result"]["fields"]["nomor_npwp"]["value"] == "12.345.678.9-012.345"


def test_scoring_ends_every_request_it_runs_for(harness, auth):
    client, callback = harness
    sequence = ["extraction", "structuring", "scoring"]

    client.post("/v1/scoring/jobs", headers=auth, json={**_payload("REQ_seq"), "pipeline_name_sequence": sequence})

    job = wait_for_job(client, "/v1/scoring/jobs/REQ_seq")
    assert (job["status"], job["pipeline_name_sequence"]) == ("DONE", sequence)
    [done] = callback.calls
    assert (done["stage"], done["status"], done["final"]) == ("SCORING", "DONE", True)


@pytest.mark.parametrize("sequence", [["guardrails", "extraction", "structuring"], ["extraction", "scoring"]])
def test_a_sequence_without_scoring_or_out_of_order_is_422(harness, auth, sequence):
    client, _ = harness

    response = client.post(
        "/v1/scoring/jobs", headers=auth, json={**_payload("REQ_seq_bad"), "pipeline_name_sequence": sequence}
    )

    assert response.status_code == 422
