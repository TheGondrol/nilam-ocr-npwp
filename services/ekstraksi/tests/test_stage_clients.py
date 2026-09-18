"""Klien ke service lain: payload yang dikirim dan pemetaan jawabannya, tanpa jaringan."""

import json

import httpx
import pytest

from ocr_common.errors import ServiceError
from ocr_common.remote import RemoteModelClient
from src.clients.stages import GuardrailsClient, ScoringClient, StructuringClient, build_stage_clients
from src.core.config import Settings


def _remote(handler) -> RemoteModelClient:
    return RemoteModelClient(
        "http://stage.test",
        5.0,
        name="stage",
        headers={"X-API-Key": "k"},
        passthrough_client_errors=True,
        transport=httpx.MockTransport(handler),
    )


def _envelope(data, status=200):
    return httpx.Response(status, json={"status_code": status, "message": "Success", "data": data, "errors": None})


async def test_guardrails_client_posts_multipart_and_returns_data():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["auth"] = request.headers["x-api-key"]
        seen["body"] = request.read()
        return _envelope({"document": {"verdict": "accepted"}, "pages": []})

    report = await GuardrailsClient(_remote(handler)).check("OCR_1", "npwp.jpg", "image/jpeg", b"\xff\xd8")
    assert report == {"document": {"verdict": "accepted"}, "pages": []}
    assert seen["path"] == "/v1/guardrails/check"
    assert seen["auth"] == "k"
    assert b'filename="npwp.jpg"' in seen["body"]
    assert b'name="request_id"' in seen["body"] and b"OCR_1" in seen["body"]


async def test_structuring_client_posts_lines():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["json"] = json.loads(request.read())
        return _envelope({"document_type": "npwp", "fields": {}})

    lines = [{"text": "NPWP : 1", "confidence": 0.9}]
    await StructuringClient(_remote(handler)).structure(lines)
    assert seen["path"] == "/v1/structuring/structure"
    assert seen["json"] == {"lines": lines}


async def test_scoring_client_strips_source_from_fields():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["json"] = json.loads(request.read())
        return _envelope({"score": 0.9, "decision": "approve", "field_scores": [], "reasons": []})

    fields = {"nomor_npwp": {"value": "1", "confidence": 0.9, "source": "NPWP : 1"}}
    result = await ScoringClient(_remote(handler)).score("npwp", fields)
    assert result["score"] == 0.9
    assert seen["json"] == {"document_type": "npwp", "fields": {"nomor_npwp": {"value": "1", "confidence": 0.9}}}


async def test_client_error_from_stage_is_passed_through():
    body = {
        "status_code": 400,
        "message": "No text lines to structure",
        "data": None,
        "errors": "No text lines to structure",
    }
    handler = lambda request: httpx.Response(400, json=body)  # noqa: E731
    with pytest.raises(ServiceError) as exc:
        await StructuringClient(_remote(handler)).structure([])
    assert exc.value.status_code == 400
    assert exc.value.message == "No text lines to structure"


async def test_unexpected_response_shape_is_500():
    handler = lambda request: httpx.Response(200, json={"unexpected": True})  # noqa: E731
    with pytest.raises(ServiceError) as exc:
        await ScoringClient(_remote(handler)).score("npwp", {})
    assert exc.value.status_code == 500
    assert exc.value.message == "stage returned an unexpected response"


def test_build_stage_clients_falls_back_to_own_api_key(monkeypatch):
    # Env API_KEY (dari conftest) diprioritaskan pydantic-settings di atas kwarg
    # bernama field; hapus dulu supaya nilai di bawah yang dipakai.
    monkeypatch.delenv("API_KEY", raising=False)
    settings = Settings(api_key="own", structuring_api_key="other", _env_file=None)
    stages = build_stage_clients(settings)
    assert stages.guardrails._client._client.headers["x-api-key"] == "own"
    assert stages.structuring._client._client.headers["x-api-key"] == "other"
    assert stages.scoring._client.base_url == "http://127.0.0.1:8033"
