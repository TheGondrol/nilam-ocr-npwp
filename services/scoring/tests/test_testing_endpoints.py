from ocr_common.pipeline import STAGE_SCORING, InMemoryJobRepository, StagePipeline
from ocr_common.testing import RecordingCallback, make_client, wait_for_job
from ocr_common.web.app import create_app

from app.api import testing
from app.config import get_settings
from app.dependencies import get_confidence_service, get_testing_job_service, get_testing_pipeline
from app.services.job_service import ScoringJobService
from tests.test_jobs import _payload


def test_testing_endpoint_is_off_by_default(client, auth):
    assert client.post("/v1/scoring/jobs-test", headers=auth, json=_payload("REQ_T0")).status_code == 404


def test_testing_pipeline_sends_no_callback():
    pipeline = get_testing_pipeline()
    assert pipeline.callbacks is False
    assert pipeline.metrics_stage == "TESTING_SCORING"


def test_testing_endpoint_runs_the_same_job(auth):
    callback = RecordingCallback()
    pipeline = StagePipeline(
        stage=STAGE_SCORING, repository=InMemoryJobRepository(), callback=callback, callbacks=False
    )
    app = create_app(settings=get_settings(), title="Scoring", description="testing", routers=[testing.router])
    app.dependency_overrides[get_testing_job_service] = lambda: ScoringJobService(pipeline, get_confidence_service())

    with make_client(app) as client:
        response = client.post("/v1/scoring/jobs-test", headers=auth, json=_payload("REQ_T1"))
        assert response.status_code == 202
        job = wait_for_job(client, "/v1/scoring/jobs-test/REQ_T1")

    assert job["status"] == "DONE"
    assert job["result"]["payload"]["npwp"] == "123456789012345"
    assert callback.calls == []
