import io
import re

import pytest
from PIL import Image

from ocr_common.testing import make_client
from ocr_common.web.app import create_app

from app.api import testing
from app.clients.ekstraksi import build_ekstraksi_client
from app.clients.stages import build_stage_status_clients
from app.config import get_settings
from app.dependencies import get_testing_ekstraksi_client, get_testing_pipeline_waiter
from tests.conftest import StubEkstraksi, StubWaiter

RID = "REQ_testing"


def _jpeg() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (200, 100), "white").save(buffer, format="JPEG")
    return buffer.getvalue()


def _submit(client, auth, path, **data):
    return client.post(
        path, headers=auth, data={"request_id": RID, **data}, files={"file": ("npwp.jpg", _jpeg(), "image/jpeg")}
    )


@pytest.fixture
def testing_client():
    """The service with TESTING_ENDPOINTS on: the testing router next to nothing else."""
    ekstraksi, waiter = StubEkstraksi(), StubWaiter()
    app = create_app(settings=get_settings(), title="Guardrails", description="testing", routers=[testing.router])
    app.dependency_overrides[get_testing_ekstraksi_client] = lambda: ekstraksi
    app.dependency_overrides[get_testing_pipeline_waiter] = lambda: waiter
    return make_client(app), ekstraksi, waiter


def test_testing_endpoint_is_off_by_default(client, auth):
    assert _submit(client, auth, "/v1/extract-ocr-test").status_code == 404


def test_testing_endpoint_answers_like_extract_ocr_through_the_testing_clients(
    testing_client, auth, stub_ekstraksi, stub_waiter
):
    client, ekstraksi, waiter = testing_client

    response = _submit(client, auth, "/v1/extract-ocr-test")

    assert response.status_code == 200
    body = response.json()
    assert (body["job_status"], body["guardrails"]) == ("completed", 1)
    # The id is minted here; the one in the form is ignored, and the pipeline runs under the minted one.
    request_id = body["request_id"]
    assert re.fullmatch(r"TEST_[0-9a-f]{32}", request_id)
    assert [job["request_id"] for job in ekstraksi.submitted] == [request_id]
    assert [waited for waited, _ in waiter.calls] == [request_id]
    assert stub_ekstraksi.submitted == [] and stub_waiter.calls == []  # the live clients were not used


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


def test_live_endpoint_keeps_the_callers_request_id(client, auth):
    assert _submit(client, auth, "/v1/extract-ocr").json()["request_id"] == RID


def test_testing_endpoint_needs_the_api_key(testing_client):
    client, _, _ = testing_client
    assert _submit(client, {}, "/v1/extract-ocr-test").status_code == 401


def test_testing_clients_call_the_stages_test_endpoints():
    settings = get_settings()
    assert build_ekstraksi_client(settings)._jobs_path == "/v1/ekstraksi/jobs"
    assert build_ekstraksi_client(settings, testing=True)._jobs_path == "/v1/ekstraksi/jobs-test"
    assert [client._jobs_path for client in build_stage_status_clients(settings, testing=True)] == [
        "/v1/ekstraksi/jobs-test",
        "/v1/structuring/jobs-test",
        "/v1/scoring/jobs-test",
    ]
    assert [client.stage for client in build_stage_status_clients(settings, testing=True)] == [
        "OCR",
        "STRUCTURING",
        "SCORING",
    ]
