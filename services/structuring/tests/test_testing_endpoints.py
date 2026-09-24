from ocr_common.pipeline import STAGE_STRUCTURING, InMemoryJobRepository, StagePipeline
from ocr_common.testing import RecordingCallback, RecordingNextStage, make_client, wait_for_job
from ocr_common.web.app import create_app

from app.api import testing
from app.config import get_settings
from app.dependencies import (
    get_structuring_service,
    get_testing_job_service,
    get_testing_next_stage,
    get_testing_pipeline,
)
from app.services.job_service import StructuringJobService
from tests.test_jobs import _payload


def test_testing_endpoint_is_off_by_default(client, auth):
    assert client.post("/v1/structuring/jobs-test", headers=auth, json=_payload("REQ_T0")).status_code == 404


def test_testing_pipeline_hands_off_to_the_scoring_test_endpoint_without_callbacks():
    pipeline = get_testing_pipeline()
    assert pipeline.callbacks is False
    assert pipeline.metrics_stage == "TESTING_STRUCTURING"
    assert pipeline.next_stage_client is get_testing_next_stage()
    assert get_testing_next_stage()._path == "/v1/scoring/jobs-test"


def test_testing_endpoint_runs_the_same_job(auth):
    callback, next_stage = RecordingCallback(), RecordingNextStage()
    pipeline = StagePipeline(
        stage=STAGE_STRUCTURING,
        repository=InMemoryJobRepository(),
        callback=callback,
        next_stage_client=next_stage,
        callbacks=False,
    )
    app = create_app(settings=get_settings(), title="Structuring", description="testing", routers=[testing.router])
    app.dependency_overrides[get_testing_job_service] = lambda: StructuringJobService(
        pipeline, get_structuring_service()
    )

    with make_client(app) as client:
        response = client.post("/v1/structuring/jobs-test", headers=auth, json=_payload("REQ_T1"))
        assert response.status_code == 202
        assert response.json()["data"]["stage"] == "STRUCTURING"
        job = wait_for_job(client, "/v1/structuring/jobs-test/REQ_T1")

    assert job["status"] == "DONE"
    assert job["result"]["fields"]["nomor_npwp"]["value"] == "12.345.678.9-012.345"
    assert next_stage.payloads == [{**_payload("REQ_T1"), "structuring": job["result"]}]
    assert callback.calls == []
