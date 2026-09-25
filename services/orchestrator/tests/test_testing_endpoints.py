import re

import pytest

from ocr_common.testing import make_client
from ocr_common.web.app import create_app

from app.api import testing
from app.clients.extraction import build_extraction_client
from app.clients.stages import build_stage_status_clients
from app.config import get_settings
from app.dependencies import get_guardrails_client, get_testing_extraction_client, get_testing_pipeline_waiter
from tests.conftest import JPEG, StubExtraction, StubGuardrails, StubWaiter

RID = "REQ_testing"


def _submit(client, auth, path, **data):
    return client.post(
        path, headers=auth, data={"request_id": RID, **data}, files={"file": ("npwp.jpg", JPEG, "image/jpeg")}
    )


@pytest.fixture
def testing_client():
    """The service with TESTING_ENDPOINTS on: the testing routers next to nothing else."""
    guardrails, extraction, waiter = StubGuardrails(), StubExtraction(), StubWaiter()
    app = create_app(settings=get_settings(), title="Orchestrator", description="testing", routers=testing.routers)
    app.dependency_overrides[get_guardrails_client] = lambda: guardrails
    app.dependency_overrides[get_testing_extraction_client] = lambda: extraction
    app.dependency_overrides[get_testing_pipeline_waiter] = lambda: waiter
    return make_client(app), extraction, waiter


def test_testing_endpoints_are_off_by_default(client, auth):
    assert _submit(client, auth, "/v1/extract-ocr-test").status_code == 404
    assert client.get("/v1/extract-ocr-test/TEST_x", headers=auth).status_code == 404


def test_testing_endpoint_answers_like_extract_ocr_through_the_testing_clients(
    testing_client, auth, stub_extraction, stub_waiter
):
    client, extraction, waiter = testing_client

    response = _submit(client, auth, "/v1/extract-ocr-test")

    assert response.status_code == 200
    body = response.json()
    assert (body["job_status"], body["guardrails"]) == ("completed", 0)
    # The id is minted here; the one in the form is ignored, and the pipeline runs under the minted one.
    request_id = body["request_id"]
    assert re.fullmatch(r"TEST_[0-9a-f]{32}", request_id)
    assert [job["request_id"] for job in extraction.submitted] == [request_id]
    assert [waited for waited, _ in waiter.calls] == [request_id]
    assert stub_extraction.submitted == [] and stub_waiter.calls == []  # the live clients were not used


def test_testing_status_reads_the_testing_jobs_under_the_path_request_id(testing_client, auth, stub_waiter):
    client, _, waiter = testing_client

    response = client.get("/v1/extract-ocr-test/TEST_run1_abc", headers=auth)

    assert response.status_code == 200
    body = response.json()
    assert (body["request_id"], body["job_status"]) == ("TEST_run1_abc", "completed")
    assert waiter.snapshots == ["TEST_run1_abc"]
    assert stub_waiter.snapshots == []


def test_testing_endpoint_mints_a_new_request_id_per_request_with_the_run_id(testing_client, auth):
    client, _, _ = testing_client
    first = _submit(client, auth, "/v1/extract-ocr-test", run_id="burst5rps").json()["request_id"]
    second = _submit(client, auth, "/v1/extract-ocr-test", run_id="burst5rps").json()["request_id"]
    assert re.fullmatch(r"TEST_burst5rps_[0-9a-f]{32}", first)
    assert first != second


def test_testing_endpoint_refuses_a_run_id_that_would_garble_the_request_id(testing_client, auth):
    client, _, _ = testing_client
    response = _submit(client, auth, "/v1/extract-ocr-test", run_id="a b/c")
    assert response.status_code == 422
    assert response.json()["errors"] == "VALIDATION_ERROR"


def test_testing_endpoint_takes_the_sequence_like_the_live_one(auth):
    guardrails, extraction, waiter = StubGuardrails(), StubExtraction(), StubWaiter()
    app = create_app(settings=get_settings(), title="Orchestrator", description="testing", routers=testing.routers)
    app.dependency_overrides[get_guardrails_client] = lambda: guardrails
    app.dependency_overrides[get_testing_extraction_client] = lambda: extraction
    app.dependency_overrides[get_testing_pipeline_waiter] = lambda: waiter
    client = make_client(app)

    without_guardrails = ["extraction", "structuring", "scoring"]
    skipped = _submit(client, auth, "/v1/extract-ocr-test", pipeline_name_sequence=without_guardrails)

    assert skipped.status_code == 200
    assert guardrails.checked == []
    assert [(job["guardrails"], job["sequence"]) for job in extraction.submitted] == [(None, without_guardrails)]


def test_live_endpoint_keeps_the_callers_request_id(client, auth):
    assert _submit(client, auth, "/v1/extract-ocr").json()["request_id"] == RID


def test_testing_endpoints_need_the_api_key(testing_client):
    client, _, _ = testing_client
    assert _submit(client, {}, "/v1/extract-ocr-test").status_code == 401
    assert client.get("/v1/extract-ocr-test/TEST_x").status_code == 401


def test_testing_clients_call_the_stages_test_endpoints():
    settings = get_settings()
    assert build_extraction_client(settings)._jobs_path == "/v1/extraction/jobs"
    assert build_extraction_client(settings, testing=True)._jobs_path == "/v1/extraction/jobs-test"
    assert [client._jobs_path for client in build_stage_status_clients(settings, testing=True)] == [
        "/v1/extraction/jobs-test",
        "/v1/structuring/jobs-test",
        "/v1/scoring/jobs-test",
    ]
    assert [client.stage for client in build_stage_status_clients(settings, testing=True)] == [
        "OCR",
        "STRUCTURING",
        "SCORING",
    ]
