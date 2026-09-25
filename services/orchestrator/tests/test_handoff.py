import json
from urllib.parse import parse_qs

import httpx
import pytest

from ocr_common.clients.remote import RemoteModelClient

from app.clients.ekstraksi import EkstraksiJobClient
from app.config import get_settings
from app.dependencies import get_ekstraksi_client
from app.main import app
from tests.conftest import JPEG

RID = "REQ_orchestrator_jobs"


def _accepted(request: httpx.Request) -> httpx.Response:
    return httpx.Response(
        202,
        json={
            "status_code": 202,
            "status_desc": "Accepted",
            "message": "Accepted",
            "data": {"request_id": RID, "stage": "OCR", "status": "PROCESSING", "duplicate": False},
            "errors": None,
            "request_id": RID,
        },
    )


class Ekstraksi:
    def __init__(self, *responses):
        self._responses = list(responses)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        response = self._responses.pop(0) if len(self._responses) > 1 else self._responses[0]
        if isinstance(response, Exception):
            raise response
        return response(request) if callable(response) else response


@pytest.fixture
def ekstraksi():
    def install(*responses) -> Ekstraksi:
        handler = Ekstraksi(*responses)
        remote = RemoteModelClient(
            "http://ekstraksi:8030",
            5.0,
            name="ekstraksi service",
            headers={"X-API-Key": "k"},
            passthrough_client_errors=True,
            transport=httpx.MockTransport(handler),
        )
        app.dependency_overrides[get_ekstraksi_client] = lambda: EkstraksiJobClient(remote, attempts=3, delay=0)
        return handler

    yield install
    app.dependency_overrides.pop(get_ekstraksi_client, None)


def _submit(client, auth, filename="npwp.jpg"):
    return client.post(
        "/v1/extract-ocr",
        headers=auth,
        data={"request_id": RID, "document_type": "npwp"},
        files={"file": (filename, JPEG, "image/jpeg")},
    )


def _form(request: httpx.Request) -> tuple[dict[str, str], bytes]:
    content_type = request.headers["content-type"]
    boundary = content_type.split("boundary=")[1].encode()
    fields: dict[str, str] = {}
    file_bytes = b""
    for part in request.content.split(b"--" + boundary):
        if b"\r\n\r\n" not in part:
            continue
        head, body = part.split(b"\r\n\r\n", 1)
        body = body.removesuffix(b"\r\n")
        name = head.split(b'name="')[1].split(b'"')[0].decode()
        if b"filename=" in head:
            file_bytes = body
        else:
            fields[name] = body.decode()
    return fields, file_bytes


def test_accepted_document_is_handed_to_the_ocr_stage(client, auth, ekstraksi):
    handler = ekstraksi(_accepted)

    response = _submit(client, auth)

    assert response.status_code == 200
    body = response.json()
    assert (body["status_code"], body["job_status"], body["guardrails"]) == (200, "completed", 0)

    [sent] = handler.requests
    assert sent.url.path == "/v1/ekstraksi/jobs"
    assert sent.headers["X-API-Key"] == "k"
    fields, file_bytes = _form(sent)
    assert fields["request_id"] == RID
    assert fields["document_type"] == "npwp"
    guardrails = json.loads(fields["guardrails"])
    assert guardrails["passed"] is True
    assert guardrails["document"]["verdict"] == "accepted"
    assert "job" not in guardrails
    assert file_bytes == JPEG


def test_a_document_that_skipped_guardrails_is_handed_over_without_a_guardrails_field(client, auth, ekstraksi):
    """ekstraksi refuses a `guardrails` that is not a JSON object, so no report means no field, not `null`."""
    handler = ekstraksi(_accepted)
    app.dependency_overrides[get_settings] = lambda: get_settings().model_copy(update={"guardrails_skip_allowed": True})
    try:
        response = client.post(
            "/v1/extract-ocr",
            headers=auth,
            data={"request_id": RID, "document_type": "npwp", "skip_guardrails": "true"},
            files={"file": ("npwp.jpg", JPEG, "image/jpeg")},
        )
    finally:
        app.dependency_overrides.pop(get_settings, None)

    assert response.status_code == 200
    [sent] = handler.requests
    fields, file_bytes = _form(sent)
    assert "guardrails" not in fields
    assert (fields["request_id"], file_bytes) == (RID, JPEG)


def test_rejected_document_stops_here(client, auth, ekstraksi):
    handler = ekstraksi(_accepted)

    response = _submit(client, auth, filename="notnpwp.jpg")

    assert response.status_code == 400
    body = response.json()
    assert (body["errors"], body["job_status"], body["guardrails"], body["data"]) == (
        "DOWNSTREAM_VALIDATION_ERROR",
        "failed",
        1,
        None,
    )
    assert body["message"].startswith("Document rejected by guardrails")
    assert handler.requests == []


def test_ekstraksi_unreachable_is_503(client, auth, ekstraksi):
    ekstraksi(httpx.ConnectError("refused"))

    response = _submit(client, auth)

    assert response.status_code == 503
    assert response.json()["message"] == "ekstraksi service is unavailable"


def test_ekstraksi_client_error_is_passed_through(client, auth, ekstraksi):
    ekstraksi(httpx.Response(400, json={"detail": "Uploaded file is empty"}))

    response = _submit(client, auth)

    assert response.status_code == 400
    assert response.json()["message"] == "Uploaded file is empty"


def test_ekstraksi_server_error_is_retried(client, auth, ekstraksi):
    handler = ekstraksi(httpx.Response(502, text="bad gateway"), _accepted)

    response = _submit(client, auth)

    assert response.status_code == 200
    assert len(handler.requests) == 2


def test_a_document_sent_as_file_url_is_judged_here_and_handed_over_as_the_same_url(
    client, auth, ekstraksi, stub_guardrails, monkeypatch
):
    handler = ekstraksi(_accepted)
    url = "http://minio.local/bucket/npwp.jpg?X-Amz-Signature=abc"

    async def fake_fetch(fetched, *, limit, timeout=10.0, policy):
        assert fetched == url
        return JPEG, "npwp.jpg", "image/jpeg"

    monkeypatch.setattr("ocr_common.web.intake.fetch", fake_fetch)
    response = client.post(
        "/v1/extract-ocr", headers=auth, data={"request_id": RID, "document_type": "npwp", "file_url": url}
    )

    assert response.status_code == 200
    [sent] = handler.requests
    assert sent.url.path == "/v1/ekstraksi/jobs"
    assert sent.headers["content-type"] == "application/x-www-form-urlencoded", "no bytes: the OCR stage downloads it"
    form = {key: value[0] for key, value in parse_qs(sent.content.decode()).items()}
    assert form["request_id"] == RID
    assert "guardrails" in form
    assert form["file_url"] == url
    assert stub_guardrails.checked == [{"request_id": RID, "filename": "npwp.jpg", "content_type": "image/jpeg"}]
