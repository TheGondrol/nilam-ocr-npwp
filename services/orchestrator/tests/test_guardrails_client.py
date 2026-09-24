import httpx
import pytest

from ocr_common.clients.remote import RemoteModelClient
from ocr_common.errors import ServiceError

from app.clients.guardrails import PASSTHROUGH_STATUSES, GuardrailsClient, build_guardrails_client
from app.config import get_settings
from tests.conftest import ACCEPTED_REPORT, JPEG
from tests.test_handoff import _form

RID = "REQ_guardrails_client"
TOO_MANY_PAGES = "Jumlah halaman melebihi batas, pastikan hanya mengunggah dokumen NPWP"


class Guardrails:
    def __init__(self, response):
        self._response = response
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if isinstance(self._response, Exception):
            raise self._response
        return self._response


def _client(response) -> tuple[GuardrailsClient, Guardrails]:
    handler = Guardrails(response)
    remote = RemoteModelClient(
        "http://guardrails:8031",
        5.0,
        name="guardrails service",
        headers={"X-API-Key": "k"},
        passthrough_statuses=PASSTHROUGH_STATUSES,
        transport=httpx.MockTransport(handler),
    )
    return GuardrailsClient(remote), handler


async def test_check_posts_the_document_and_returns_the_report():
    client, handler = _client(httpx.Response(200, json={"status_code": 200, "data": ACCEPTED_REPORT}))

    assert await client.check(RID, "npwp.jpg", "image/jpeg", JPEG) == ACCEPTED_REPORT

    [sent] = handler.requests
    assert sent.url.path == "/v1/guardrails/check"
    assert sent.headers["X-API-Key"] == "k"
    fields, file_bytes = _form(sent)
    assert fields == {"request_id": RID}
    assert file_bytes == JPEG
    assert b'filename="npwp.jpg"' in sent.content


async def test_a_file_without_a_name_is_sent_as_upload():
    """An empty file name would make guardrails see no file at all (400)."""
    client, handler = _client(httpx.Response(200, json={"data": ACCEPTED_REPORT}))

    await client.check(RID, "", None, JPEG)

    assert b'filename="upload"' in handler.requests[0].content


@pytest.mark.parametrize(
    ("status", "message"),
    [
        (400, TOO_MANY_PAGES),
        (413, "Ukuran dokumen melebihi batas 2,5 MB, pastikan hanya mengunggah dokumen NPWP"),
        (503, "guardrails model is unavailable"),
        (504, "guardrails model timed out after 30.0s"),
    ],
)
async def test_refusals_and_model_outages_are_passed_on_as_they_are(status, message):
    client, _ = _client(httpx.Response(status, json={"status_code": status, "message": message}))

    with pytest.raises(ServiceError) as exc:
        await client.check(RID, "npwp.jpg", "image/jpeg", JPEG)
    assert (exc.value.status_code, exc.value.message) == (status, message)


async def test_a_wrong_key_is_our_fault_and_becomes_500():
    client, _ = _client(httpx.Response(401, json={"message": "Invalid or missing API key"}))

    with pytest.raises(ServiceError) as exc:
        await client.check(RID, "npwp.jpg", "image/jpeg", JPEG)
    assert (exc.value.status_code, exc.value.message) == (
        500,
        "guardrails service error (401): Invalid or missing API key",
    )


async def test_unreachable_guardrails_is_503_and_is_not_retried():
    client, handler = _client(httpx.ConnectError("refused"))

    with pytest.raises(ServiceError) as exc:
        await client.check(RID, "npwp.jpg", "image/jpeg", JPEG)
    assert (exc.value.status_code, exc.value.message) == (503, "guardrails service is unavailable")
    assert len(handler.requests) == 1


async def test_an_answer_without_a_report_is_500():
    client, _ = _client(httpx.Response(200, json={"data": {"document": {}}}))

    with pytest.raises(ServiceError) as exc:
        await client.check(RID, "npwp.jpg", "image/jpeg", JPEG)
    assert (exc.value.status_code, exc.value.message) == (500, "guardrails service returned an unexpected response")


async def test_the_client_is_built_from_the_settings():
    settings = get_settings().model_copy(
        update={"guardrails_service_url": "http://guardrails:8031/", "guardrails_timeout_seconds": 7.0}
    )
    client = build_guardrails_client(settings)
    try:
        remote = client._client
        assert (remote.base_url, remote.timeout) == ("http://guardrails:8031", 7.0)
        assert remote._client.headers["X-API-Key"] == settings.api_key
        assert remote._passthrough_statuses == frozenset(PASSTHROUGH_STATUSES)
    finally:
        await client.aclose()
