"""Every failed answer of the entry point carries a stable `errors` code (the API spec's Error Codes), so the
central orchestrator branches on it and never on the wording of `message`."""

import pytest

from ocr_common.errors import BadRequest, InternalError, UpstreamTimeout, UpstreamUnavailable

from app.config import get_settings
from app.main import app
from tests.conftest import JPEG

RID = "OCR_codes"


def _submit(client, auth, files=None, **form):
    files = files if files is not None else {"file": ("npwp.jpg", JPEG, "image/jpeg")}
    return client.post("/v1/extract-ocr", headers=auth, data={"request_id": RID, **form}, files=files)


@pytest.mark.parametrize(
    ("files", "form", "status", "code"),
    [
        ({"file": ("npwp.jpg", b"", "image/jpeg")}, {}, 400, "EMPTY_FILE"),
        ({"file": ("npwp.txt", b"hello", "text/plain")}, {}, 400, "UNSUPPORTED_FILE_TYPE"),
        ({"file": ("npwp.pdf", b"not a pdf", "application/pdf")}, {}, 400, "UNREADABLE_FILE"),
        ({}, {}, 400, "INVALID_FILE_SOURCE"),
        (None, {"file_url": "https://minio.example/a.jpg"}, 400, "INVALID_FILE_SOURCE"),
        (None, {"document_type": "ktp"}, 400, "UNSUPPORTED_DOCUMENT_TYPE"),
        (None, {"params": "{bad"}, 422, "INVALID_PARAMS"),
    ],
)
def test_refusals_before_the_pipeline_carry_their_code(client, auth, files, form, status, code):
    response = _submit(client, auth, files=files, **form)

    assert (response.status_code, response.json()["errors"]) == (status, code)


def test_a_file_above_the_limit_is_413_file_too_large(client, auth):
    app.dependency_overrides[get_settings] = lambda: get_settings().model_copy(update={"max_upload_bytes": 4})
    try:
        response = _submit(client, auth)
    finally:
        app.dependency_overrides.pop(get_settings, None)

    assert (response.status_code, response.json()["errors"]) == (413, "FILE_TOO_LARGE")


def test_a_missing_api_key_is_401_unauthorized(client):
    response = _submit(client, {})

    assert (response.status_code, response.json()["errors"]) == (401, "UNAUTHORIZED")


@pytest.mark.parametrize(
    ("error", "status", "code"),
    [
        (BadRequest("Uploaded file is not a readable image"), 400, "UNREADABLE_FILE"),
        (InternalError("guardrails service returned an unexpected response"), 500, "DOWNSTREAM_SERVER_ERROR"),
        (UpstreamUnavailable("guardrails service is unavailable"), 503, "DOWNSTREAM_UNAVAILABLE"),
        (UpstreamTimeout("guardrails service timed out after 20.0s"), 504, "DOWNSTREAM_TIMEOUT"),
    ],
)
def test_a_pipeline_service_failing_carries_its_code(client, auth, stub_guardrails, error, status, code):
    stub_guardrails.error = error

    response = _submit(client, auth)

    body = response.json()
    assert (response.status_code, body["errors"], body["pipeline_last_stage"]) == (status, code, "guardrails")
    assert body["message"] == error.message


def test_an_unknown_request_id_is_404_request_id_not_found(client, auth, stub_waiter):
    stub_waiter.snapshot_outcome = None

    response = client.get(f"/v1/extract-ocr/{RID}", headers=auth)

    assert (response.status_code, response.json()["errors"]) == (404, "REQUEST_ID_NOT_FOUND")


@pytest.mark.parametrize(
    ("files", "form", "headers_ok"),
    [
        ({"file": ("npwp.jpg", b"", "image/jpeg")}, {}, True),  # a file check (plain envelope)
        (None, {"document_type": "ktp"}, True),  # a refusal in the extract-ocr shape
        (None, {}, False),  # no API key
    ],
)
def test_every_refusal_of_the_entry_point_names_the_orchestrator(client, auth, files, form, headers_ok):
    response = _submit(client, auth if headers_ok else {}, files=files, **form)

    assert response.status_code >= 400
    assert response.json()["pipeline_last_stage"] == "orchestrator"


def test_a_missing_required_field_names_the_orchestrator(client, auth):
    response = client.post("/v1/extract-ocr", headers=auth, files={"file": ("npwp.jpg", JPEG, "image/jpeg")})

    body = response.json()
    assert (response.status_code, body["errors"], body["pipeline_last_stage"]) == (
        422,
        "VALIDATION_ERROR",
        "orchestrator",
    )


def test_an_unknown_request_id_names_the_orchestrator(client, auth, stub_waiter):
    stub_waiter.snapshot_outcome = None

    assert client.get(f"/v1/extract-ocr/{RID}", headers=auth).json()["pipeline_last_stage"] == "orchestrator"
