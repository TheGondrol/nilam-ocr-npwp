import json

from ocr_common.pipeline import STAGE_OCR, InMemoryJobRepository, StagePipeline
from ocr_common.testing import RecordingCallback, RecordingNextStage, image_upload, make_client, wait_for_job
from ocr_common.web.app import create_app

from app.api import testing
from app.config import get_settings
from app.dependencies import (
    get_ekstraksi_service,
    get_testing_job_service,
    get_testing_next_stage,
    get_testing_pipeline,
)
from app.services.job_service import EkstraksiJobService

GUARDRAILS = {"passed": True, "reason": None}


def _submit(client, auth, request_id):
    data = {"request_id": request_id, "document_type": "npwp", "guardrails": json.dumps(GUARDRAILS)}
    return client.post("/v1/ekstraksi/jobs-test", headers=auth, data=data, files=image_upload("npwp.jpg"))


def test_testing_endpoint_is_off_by_default(client, auth):
    assert _submit(client, auth, "REQ_T0").status_code == 404


def test_testing_pipeline_hands_off_to_the_structuring_test_endpoint_without_callbacks():
    pipeline = get_testing_pipeline()
    assert pipeline.callbacks is False
    assert pipeline.metrics_stage == "TESTING_OCR"
    assert pipeline.next_stage_client is get_testing_next_stage()
    assert get_testing_next_stage()._path == "/v1/structuring/jobs-test"


def test_testing_endpoint_runs_the_same_job(auth):
    callback, next_stage = RecordingCallback(), RecordingNextStage()
    pipeline = StagePipeline(
        stage=STAGE_OCR,
        repository=InMemoryJobRepository(),
        callback=callback,
        next_stage_client=next_stage,
        callbacks=False,
    )
    app = create_app(settings=get_settings(), title="Ekstraksi", description="testing", routers=[testing.router])
    app.dependency_overrides[get_testing_job_service] = lambda: EkstraksiJobService(
        pipeline, get_ekstraksi_service(), 5 * 1024 * 1024
    )

    with make_client(app) as client:
        response = _submit(client, auth, "REQ_T1")
        assert response.status_code == 202
        assert response.json()["data"]["stage"] == "OCR"
        job = wait_for_job(client, "/v1/ekstraksi/jobs-test/REQ_T1")

    assert job["status"] == "DONE"
    assert [payload["request_id"] for payload in next_stage.payloads] == ["REQ_T1"]
    assert callback.calls == []
