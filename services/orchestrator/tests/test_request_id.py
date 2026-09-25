"""The central orchestrator's request_id is the X-Request-ID of every call the orchestrator makes, so one
id follows a request through the logs of guardrails and every stage."""

import httpx

from ocr_common.clients.remote import RemoteModelClient
from ocr_common.web.request_id import REQUEST_ID_HEADER

from app.clients.guardrails import GuardrailsClient
from app.dependencies import get_guardrails_client
from app.main import app
from tests.conftest import ACCEPTED_REPORT, JPEG

RID = "OCR_request_id"


def test_calls_carry_the_request_id_of_the_form_not_of_the_header(client, auth):
    seen: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers.get(REQUEST_ID_HEADER))
        return httpx.Response(200, json={"data": ACCEPTED_REPORT})

    remote = RemoteModelClient("http://guardrails:8031", 5.0, name="guardrails", transport=httpx.MockTransport(handler))
    app.dependency_overrides[get_guardrails_client] = lambda: GuardrailsClient(remote)
    try:
        response = client.post(
            "/v1/extract-ocr",
            data={"request_id": RID},
            files={"file": ("npwp.jpg", JPEG, "image/jpeg")},
            headers={**auth, REQUEST_ID_HEADER: "REQ_gateway_hop"},
        )
    finally:
        app.dependency_overrides.pop(get_guardrails_client, None)

    assert response.status_code == 200
    assert seen == [RID]
