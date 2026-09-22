import io
import json

import httpx
import pytest
from PIL import Image

from ocr_common.remote import RemoteModelClient
from src.clients.ekstraksi import EkstraksiJobClient, get_ekstraksi_client
from src.main import app

RID = "REQ_guardrails_jobs"


def _jpeg() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (200, 100), "white").save(buffer, format="JPEG")
    return buffer.getvalue()


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
        files={"file": (filename, _jpeg(), "image/jpeg")},
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
    assert (body["status_code"], body["job_status"], body["guardrails"]) == (200, "completed", 1)

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
    assert file_bytes == _jpeg()


def test_rejected_document_stops_here(client, auth, ekstraksi):
    handler = ekstraksi(_accepted)

    response = _submit(client, auth, filename="notnpwp.jpg")

    assert response.status_code == 400
    body = response.json()
    assert (body["errors"], body["job_status"], body["guardrails"], body["data"]) == (
        "DOWNSTREAM_VALIDATION_ERROR",
        "failed",
        0,
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


def test_check_endpoint_judges_only(client, auth, stub_ekstraksi, stub_waiter):
    response = client.post(
        "/v1/guardrails/check",
        headers=auth,
        data={"request_id": RID},
        files={"file": ("npwp.jpg", _jpeg(), "image/jpeg")},
    )

    assert response.status_code == 200
    assert response.json()["data"]["passed"] is True
    assert stub_ekstraksi.submitted == []
    assert stub_waiter.calls == []
