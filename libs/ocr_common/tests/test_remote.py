import json

import httpx
import pytest

from ocr_common.clients.remote import RemoteModelClient
from ocr_common.errors import ServiceError


def _client(handler, timeout: float = 7.0, **kwargs) -> RemoteModelClient:
    return RemoteModelClient(
        "http://model.test/", timeout, name="demo model", transport=httpx.MockTransport(handler), **kwargs
    )


async def test_get_json_returns_parsed_body_and_strips_trailing_slash():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        return httpx.Response(200, json={"status": "ok"})

    assert await _client(handler).get_json("/health") == {"status": "ok"}
    assert seen["url"] == "http://model.test/health"


async def test_post_json_sends_payload_and_headers():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["json"] = json.loads(request.read())
        seen["content_type"] = request.headers.get("content-type")
        seen["auth"] = request.headers.get("x-api-key")
        return httpx.Response(200, json={"ok": True})

    await _client(handler).post_json("/v1/structuring/structure", {"lines": []}, headers={"X-API-Key": "k"})
    assert seen["json"] == {"lines": []}
    assert seen["content_type"] == "application/json"
    assert seen["auth"] == "k"


async def test_default_headers_are_sent_on_every_request():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers.get("x-api-key")
        return httpx.Response(200, json={})

    await _client(handler, headers={"X-API-Key": "svc"}).get_json("/health")
    assert seen["auth"] == "svc"


async def test_post_multipart_sends_file_field():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = request.read()
        return httpx.Response(200, json={})

    await _client(handler).post_multipart(
        "/ocr", filename="a.jpg", content=b"\xff\xd8", content_type="image/jpeg", data={"request_id": "OCR_1"}
    )
    assert b'name="file"; filename="a.jpg"' in seen["body"]
    assert b"Content-Type: image/jpeg" in seen["body"]
    assert b'name="request_id"' in seen["body"]


async def test_connect_error_maps_to_503():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(ServiceError) as exc:
        await _client(handler).get_json("/health")
    assert exc.value.status_code == 503
    assert exc.value.message == "demo model is unavailable"


async def test_connect_timeout_maps_to_503():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("slow", request=request)

    with pytest.raises(ServiceError) as exc:
        await _client(handler).get_json("/health")
    assert exc.value.status_code == 503


async def test_read_timeout_maps_to_504_with_timeout_in_message():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    with pytest.raises(ServiceError) as exc:
        await _client(handler, timeout=12.5).get_json("/health")
    assert exc.value.status_code == 504
    assert exc.value.message == "demo model timed out after 12.5s"


async def test_http_500_with_string_detail_maps_to_500_with_detail():
    handler = lambda request: httpx.Response(500, json={"detail": "error: OpenCV imdecode failed"})  # noqa: E731
    with pytest.raises(ServiceError) as exc:
        await _client(handler).post_multipart("/ocr", filename="x.jpg", content=b"", content_type="image/jpeg")
    assert exc.value.status_code == 500
    assert exc.value.message == "demo model error (500): error: OpenCV imdecode failed"


async def test_http_422_with_validation_list_is_flattened():
    body = {"detail": [{"type": "missing", "loc": ["body", "file"], "msg": "Field required", "input": None}]}
    handler = lambda request: httpx.Response(422, json=body)  # noqa: E731
    with pytest.raises(ServiceError) as exc:
        await _client(handler).post_json("/ocr", {})
    assert exc.value.message == "demo model error (422): body.file: Field required"


async def test_envelope_message_is_used_when_no_detail():
    body = {"status_code": 400, "message": "No text lines to structure", "errors": "No text lines to structure"}
    handler = lambda request: httpx.Response(400, json=body)  # noqa: E731
    with pytest.raises(ServiceError) as exc:
        await _client(handler).post_json("/v1/structuring/structure", {})
    assert exc.value.status_code == 500
    assert exc.value.message == "demo model error (400): No text lines to structure"


async def test_passthrough_keeps_4xx_status_and_message():
    body = {"status_code": 400, "message": "No text lines to structure", "errors": "No text lines to structure"}
    handler = lambda request: httpx.Response(400, json=body)  # noqa: E731
    with pytest.raises(ServiceError) as exc:
        await _client(handler, passthrough_client_errors=True).post_json("/v1/structuring/structure", {})
    assert exc.value.status_code == 400
    assert exc.value.message == "No text lines to structure"


async def test_passthrough_still_maps_5xx_to_500():
    handler = lambda request: httpx.Response(503, json={"message": "structuring model is unavailable"})  # noqa: E731
    with pytest.raises(ServiceError) as exc:
        await _client(handler, passthrough_client_errors=True).get_json("/x")
    assert exc.value.status_code == 500
    assert exc.value.message == "demo model error (503): structuring model is unavailable"


async def test_http_error_without_json_uses_text():
    handler = lambda request: httpx.Response(502, text="Bad Gateway from nginx")  # noqa: E731
    with pytest.raises(ServiceError) as exc:
        await _client(handler).get_json("/health")
    assert exc.value.message == "demo model error (502): Bad Gateway from nginx"


async def test_non_json_success_body_maps_to_500():
    handler = lambda request: httpx.Response(200, text="<html>not json</html>")  # noqa: E731
    with pytest.raises(ServiceError) as exc:
        await _client(handler).get_json("/health")
    assert exc.value.status_code == 500
    assert exc.value.message == "demo model returned an invalid response"


async def test_post_form_sends_fields_without_a_file():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(202, json={"ok": True})

    client = RemoteModelClient("http://svc", 5.0, name="svc", transport=httpx.MockTransport(handler))
    try:
        assert await client.post_form("/jobs", data={"request_id": "REQ_1", "file_url": "http://minio/a.jpg"}) == {
            "ok": True
        }
    finally:
        await client.aclose()
    [request] = seen
    assert request.headers["content-type"] == "application/x-www-form-urlencoded"
    assert b"file_url=http%3A%2F%2Fminio%2Fa.jpg" in request.content
