import pytest

from ocr_common.pipeline import STAGE_STRUCTURING, InMemoryJobRepository, StagePipeline
from ocr_common.testing import RecordingCallback, RecordingNextStage, make_client, wait_for_job
from ocr_common.types import OcrBlock, StructuredDocument

from app.dependencies import get_job_service, get_structuring_service
from app.main import app
from app.services.job_service import StructuringJobService
from app.services.structuring_service import StructuringService

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
    pipeline = StagePipeline(
        stage=STAGE_STRUCTURING, repository=InMemoryJobRepository(), callback=callback, next_stage_client=next_stage
    )
    service = StructuringJobService(pipeline, get_structuring_service())
    app.dependency_overrides[get_job_service] = lambda: service
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


def test_missing_ocr_without_a_database_is_422(harness, auth):
    client, _, _ = harness
    response = client.post("/v1/structuring/jobs", headers=auth, json={"request_id": "REQ_4"})
    assert response.status_code == 422
    assert "ocr is missing" in response.json()["message"]


def test_get_unknown_job_is_404(harness, auth):
    client, _, _ = harness
    assert client.get("/v1/structuring/jobs/REQ_missing", headers=auth).status_code == 404


def test_requires_api_key(harness):
    client, _, _ = harness
    assert client.post("/v1/structuring/jobs", json=_payload("REQ_5")).status_code == 401


class FakeResults:
    """Stands in for the shared database: {stage_prefix: {request_id: result}}."""

    def __init__(self, **stored):
        self.stored = stored

    async def get(self, stage_prefix, request_id):
        return self.stored.get(stage_prefix, {}).get(request_id)


@pytest.fixture
def reference_harness():
    callback, next_stage = RecordingCallback(), RecordingNextStage()
    pipeline = StagePipeline(
        stage=STAGE_STRUCTURING, repository=InMemoryJobRepository(), callback=callback, next_stage_client=next_stage
    )
    results = FakeResults(ocr={"REQ_ref": OCR})
    service = StructuringJobService(pipeline, get_structuring_service(), results=results, handoff_by_reference=True)
    app.dependency_overrides[get_job_service] = lambda: service
    with make_client(app) as client:
        yield client, next_stage
    app.dependency_overrides.pop(get_job_service, None)


def test_by_reference_reads_the_ocr_result_from_the_database_and_hands_off_a_reference(reference_harness, auth):
    client, next_stage = reference_harness
    body = {"request_id": "REQ_ref", "document_type": "npwp", "guardrails": GUARDRAILS}

    assert client.post("/v1/structuring/jobs", headers=auth, json=body).status_code == 202
    job = wait_for_job(client, "/v1/structuring/jobs/REQ_ref")

    assert job["status"] == "DONE"
    assert job["result"]["fields"]["nomor_npwp"]["value"] == "12.345.678.9-012.345"
    assert next_stage.payloads == [body], "no ocr and no structuring in the hand-off: scoring reads them itself"


def test_by_reference_fails_the_job_when_the_ocr_result_is_not_stored(reference_harness, auth):
    client, next_stage = reference_harness
    client.post("/v1/structuring/jobs", headers=auth, json={"request_id": "REQ_unknown", "document_type": "npwp"})
    job = wait_for_job(client, "/v1/structuring/jobs/REQ_unknown")

    assert job["status"] == "FAILED"
    assert "no ocr result stored for REQ_unknown" in job["error_message"]
    assert next_stage.payloads == []


async def test_a_stale_job_is_run_again_from_what_the_database_holds():
    callback, next_stage = RecordingCallback(), RecordingNextStage()
    pipeline = StagePipeline(
        stage=STAGE_STRUCTURING, repository=InMemoryJobRepository(), callback=callback, next_stage_client=next_stage
    )
    service = StructuringJobService(pipeline, get_structuring_service(), results=FakeResults(ocr={"REQ_stale": OCR}))
    stored_input = {"document_type": "npwp", "guardrails": GUARDRAILS}
    await pipeline.repository.claim("REQ_stale", input=stored_input)  # the process that claimed it died here

    await service.resume("REQ_stale", stored_input)
    await pipeline.runner.drain(5)

    assert (await pipeline.get("REQ_stale"))["status"] == "DONE"
    [payload] = next_stage.payloads
    assert (payload["guardrails"], payload["ocr"]) == (GUARDRAILS, OCR)
    assert [(c["stage"], c["status"]) for c in callback.calls] == [("STRUCTURING", "DONE")]


class _RejectingStructurer:
    """Stands in for the ML team's rules when a rejecting check fires (backend-independent)."""

    name = "rejecting"

    def structure(self, lines: list[OcrBlock]) -> StructuredDocument:
        return {
            "fields": {
                name: {"value": None, "confidence": 0.0, "source": None, "signals": None}
                for name in ("nomor_npwp", "nama", "nama_badan")
            },
            "flag": True,
            "flag_reason": "Nama hanya terdiri dari 1 kata, mohon dicek kembali",
            "reject_reason": "Kode provinsi pada NPWP tidak valid, mohon dicek kembali",
        }


async def test_a_rejected_document_stops_at_structuring_with_a_failed_callback():
    callback, next_stage = RecordingCallback(), RecordingNextStage()
    pipeline = StagePipeline(
        stage=STAGE_STRUCTURING, repository=InMemoryJobRepository(), callback=callback, next_stage_client=next_stage
    )
    service = StructuringJobService(pipeline, StructuringService(_RejectingStructurer()))

    await service.submit("REQ_rejected", "npwp", GUARDRAILS, OCR)
    await pipeline.runner.drain(5)

    job = await pipeline.get("REQ_rejected")
    assert job["status"] == "DONE", "the result stays readable for the waiter and for debugging"
    assert job["result"]["reject_reason"] == "Kode provinsi pada NPWP tidak valid, mohon dicek kembali"
    assert next_stage.payloads == [], "a rejected document never reaches scoring"
    assert [(c["stage"], c["status"], c["error_message"], c["error_code"]) for c in callback.calls] == [
        (
            "STRUCTURING",
            "FAILED",
            "Kode provinsi pada NPWP tidak valid, mohon dicek kembali",
            "DOWNSTREAM_VALIDATION_ERROR",
        )
    ]
