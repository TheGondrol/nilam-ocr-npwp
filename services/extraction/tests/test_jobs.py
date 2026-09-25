import json

import pytest

from ocr_common.errors import ServiceError
from ocr_common.pipeline import STAGE_OCR, InMemoryJobRepository, StagePipeline
from ocr_common.testing import RecordingCallback, RecordingNextStage, image_upload, make_client, wait_for_job

from app.dependencies import get_extraction_service, get_job_service
from app.main import app
from app.services.job_service import ExtractionJobService

GUARDRAILS = {"passed": True, "reason": None}


@pytest.fixture
def harness():
    callback, next_stage = RecordingCallback(), RecordingNextStage()
    pipeline = StagePipeline(
        stage=STAGE_OCR, repository=InMemoryJobRepository(), callback=callback, next_stage_client=next_stage
    )
    service = ExtractionJobService(pipeline, get_extraction_service(), 5 * 1024 * 1024, simulate_delay=True)
    app.dependency_overrides[get_job_service] = lambda: service
    with make_client(app) as client:
        yield client, callback, next_stage
    app.dependency_overrides.pop(get_job_service, None)


def _submit(client, auth, request_id, **kwargs):
    data = {"request_id": request_id, "document_type": "npwp", "guardrails": json.dumps(GUARDRAILS)}
    return client.post("/v1/extraction/jobs", headers=auth, data=data, files=image_upload("npwp.jpg"), **kwargs)


def test_submit_returns_202_then_runs_ocr_callback_and_handoff(harness, auth):
    client, callback, next_stage = harness

    response = _submit(client, auth, "REQ_1")
    assert response.status_code == 202
    body = response.json()
    assert body["status_code"] == 202
    assert body["request_id"] == "REQ_1"
    assert body["data"] == {"request_id": "REQ_1", "stage": "OCR", "status": "PROCESSING", "duplicate": False}

    job = wait_for_job(client, "/v1/extraction/jobs/REQ_1")
    assert job["status"] == "DONE"
    assert job["result"]["blocks"]

    assert [(c["stage"], c["status"], c["result"]) for c in callback.calls] == [("OCR", "DONE", None)]
    assert next_stage.payloads == [
        {"request_id": "REQ_1", "document_type": "npwp", "guardrails": GUARDRAILS, "ocr": job["result"]}
    ]


def test_same_request_id_is_not_processed_twice(harness, auth):
    client, callback, next_stage = harness
    _submit(client, auth, "REQ_2")
    wait_for_job(client, "/v1/extraction/jobs/REQ_2")

    again = _submit(client, auth, "REQ_2")
    assert again.status_code == 202
    assert again.json()["data"] == {"request_id": "REQ_2", "stage": "OCR", "status": "DONE", "duplicate": True}
    assert len(callback.calls) == 1
    assert len(next_stage.payloads) == 1


def test_bad_file_fails_the_job_not_the_request(harness, auth):
    client, callback, next_stage = harness
    response = client.post(
        "/v1/extraction/jobs",
        headers=auth,
        data={"request_id": "REQ_3"},
        files={"file": ("npwp.txt", b"hello", "text/plain")},
    )
    assert response.status_code == 202

    job = wait_for_job(client, "/v1/extraction/jobs/REQ_3")
    assert job["status"] == "FAILED"
    assert job["error_message"]
    assert [(c["stage"], c["status"]) for c in callback.calls] == [("OCR", "FAILED")]
    assert callback.calls[0]["error_message"] == job["error_message"]
    assert next_stage.payloads == []


def test_handoff_failure_is_reported_as_structuring_failed(harness, auth):
    client, callback, next_stage = harness
    next_stage.error = ServiceError(503, "structuring service is unavailable")
    _submit(client, auth, "REQ_4")

    assert wait_for_job(client, "/v1/extraction/jobs/REQ_4")["status"] == "DONE"
    for _ in range(100):
        if len(callback.calls) == 2:
            break
        wait_for_job(client, "/v1/extraction/jobs/REQ_4")
    assert [(c["stage"], c["status"]) for c in callback.calls] == [("OCR", "DONE"), ("STRUCTURING", "FAILED")]


def test_file_url_is_downloaded_in_background(harness, auth, monkeypatch):
    client, callback, next_stage = harness

    async def fake_fetch(url, *, limit, timeout=10.0, policy):
        assert url == "http://minio:9000/bucket/npwp.jpg?sig=x"
        return b"\xff\xd8fake-jpeg-bytes", "npwp.jpg", "image/jpeg"

    monkeypatch.setattr("app.services.job_service.fetch", fake_fetch)
    response = client.post(
        "/v1/extraction/jobs",
        headers=auth,
        data={"request_id": "REQ_5", "file_url": "http://minio:9000/bucket/npwp.jpg?sig=x"},
    )
    assert response.status_code == 202
    assert wait_for_job(client, "/v1/extraction/jobs/REQ_5")["status"] == "DONE"
    assert next_stage.payloads[0]["guardrails"] is None


def test_requires_exactly_one_of_file_or_file_url(harness, auth):
    client, _, _ = harness
    response = client.post("/v1/extraction/jobs", headers=auth, data={"request_id": "REQ_6"})
    assert response.status_code == 400
    assert response.json()["message"] == "Send exactly one of file or file_url"


def test_malformed_guardrails_json_is_400(harness, auth):
    client, _, _ = harness
    response = client.post(
        "/v1/extraction/jobs",
        headers=auth,
        data={"request_id": "REQ_7", "guardrails": "not-json"},
        files=image_upload("npwp.jpg"),
    )
    assert response.status_code == 400
    assert response.json()["message"] == "guardrails must be a JSON object"


def test_get_unknown_job_is_404(harness, auth):
    client, _, _ = harness
    response = client.get("/v1/extraction/jobs/REQ_missing", headers=auth)
    assert response.status_code == 404
    assert response.json()["request_id"] == "REQ_missing"


def test_requires_api_key(harness):
    client, _, _ = harness
    assert client.post("/v1/extraction/jobs", data={"request_id": "REQ_8"}).status_code == 401


def test_a_delay_token_in_the_file_name_holds_the_job_back_in_local_mode(harness, auth):
    client, _, _ = harness
    data = {"request_id": "REQ_slow", "document_type": "npwp", "guardrails": json.dumps(GUARDRAILS)}
    client.post("/v1/extraction/jobs", headers=auth, data=data, files=image_upload("delay1s-npwp.jpg"))

    assert wait_for_job(client, "/v1/extraction/jobs/REQ_slow", timeout=0.3)["status"] == "PROCESSING"
    assert wait_for_job(client, "/v1/extraction/jobs/REQ_slow", timeout=3)["status"] == "DONE"


def test_handoff_by_reference_leaves_the_ocr_result_out_of_the_payload(auth):
    callback, next_stage = RecordingCallback(), RecordingNextStage()
    pipeline = StagePipeline(
        stage=STAGE_OCR, repository=InMemoryJobRepository(), callback=callback, next_stage_client=next_stage
    )
    service = ExtractionJobService(
        pipeline, get_extraction_service(), 5 * 1024 * 1024, simulate_delay=True, handoff_by_reference=True
    )
    app.dependency_overrides[get_job_service] = lambda: service
    try:
        with make_client(app) as client:
            _submit(client, auth, "REQ_ref")
            job = wait_for_job(client, "/v1/extraction/jobs/REQ_ref")
    finally:
        app.dependency_overrides.pop(get_job_service, None)

    assert job["status"] == "DONE"
    assert next_stage.payloads == [{"request_id": "REQ_ref", "document_type": "npwp", "guardrails": GUARDRAILS}]


def _stale_service():
    callback, next_stage = RecordingCallback(), RecordingNextStage()
    pipeline = StagePipeline(
        stage=STAGE_OCR, repository=InMemoryJobRepository(), callback=callback, next_stage_client=next_stage
    )
    return ExtractionJobService(pipeline, get_extraction_service(), 5 * 1024 * 1024), pipeline, callback, next_stage


async def test_a_stale_job_sent_as_file_url_is_fetched_and_run_again(monkeypatch):
    service, pipeline, callback, next_stage = _stale_service()
    stored_input = {"document_type": "npwp", "guardrails": GUARDRAILS, "file_url": "http://minio:9000/b/npwp.jpg"}
    await pipeline.repository.claim("REQ_stale", input=stored_input)

    async def fake_fetch(url, *, limit, timeout=10.0, policy):
        assert url == stored_input["file_url"]
        return b"\xff\xd8fake-jpeg-bytes", "npwp.jpg", "image/jpeg"

    monkeypatch.setattr("app.services.job_service.fetch", fake_fetch)
    await service.resume("REQ_stale", stored_input)
    await pipeline.runner.drain(5)

    assert (await pipeline.get("REQ_stale"))["status"] == "DONE"
    assert next_stage.payloads[0]["guardrails"] == GUARDRAILS
    assert [(c["stage"], c["status"]) for c in callback.calls] == [("OCR", "DONE")]


async def test_a_stale_job_of_an_inline_upload_fails_with_a_reason():
    service, pipeline, callback, next_stage = _stale_service()
    await pipeline.repository.claim("REQ_gone", input={"document_type": "npwp", "guardrails": None, "file_url": None})

    await service.resume("REQ_gone", {"document_type": "npwp", "guardrails": None, "file_url": None})
    await pipeline.runner.drain(5)

    job = await pipeline.get("REQ_gone")
    assert job["status"] == "FAILED"
    assert "uploaded inline" in job["error_message"]
    assert [(c["stage"], c["status"]) for c in callback.calls] == [("OCR", "FAILED")]
    assert next_stage.payloads == []
