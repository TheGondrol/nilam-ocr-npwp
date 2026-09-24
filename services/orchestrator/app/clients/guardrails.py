from typing import Any

from ocr_common.clients.remote import RemoteModelClient
from ocr_common.errors import InternalError

from app.config import Settings

GUARDRAILS_CHECK_PATH = "/v1/guardrails/check"

# Relayed as they are: the guardrails service's refusals of the document (400 page limit or unreadable
# file, 413 size, both with a message the client can show) and the outages of its model (503, 504).
# Anything else (401 wrong key, 422, 500) is a fault on our side and becomes 500.
PASSTHROUGH_STATUSES = (400, 413, 503, 504)


class GuardrailsClient:
    def __init__(self, client: RemoteModelClient):
        self._client = client

    async def check(self, request_id: str, filename: str, content_type: str | None, content: bytes) -> dict[str, Any]:
        """The guardrails report of the document: `{passed, reason, document, pages}`.

        Not retried: the check runs inside the caller's time budget, and the caller may send the same
        request_id again (the pipeline is idempotent per request_id)."""
        body = await self._client.post_multipart(
            GUARDRAILS_CHECK_PATH,
            filename=filename or "upload",
            content=content,
            content_type=content_type or "application/octet-stream",
            data={"request_id": request_id},
        )
        data = body.get("data") if isinstance(body, dict) else None
        if not isinstance(data, dict) or not isinstance(data.get("passed"), bool):
            raise InternalError(f"{self._client.name} returned an unexpected response")
        return data

    async def aclose(self) -> None:
        await self._client.aclose()


def build_guardrails_client(settings: Settings) -> GuardrailsClient:
    return GuardrailsClient(
        RemoteModelClient(
            settings.guardrails_service_url,
            settings.guardrails_timeout_seconds,
            name="guardrails service",
            headers={"X-API-Key": settings.guardrails_api_key or settings.api_key},
            passthrough_statuses=PASSTHROUGH_STATUSES,
        )
    )
