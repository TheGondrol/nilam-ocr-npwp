from dataclasses import dataclass
from typing import Any, Literal

from ocr_common.clients.remote import RemoteModelClient
from ocr_common.errors import InternalError

from app.config import Settings

GUARDRAILS_CHECK_PATH = "/v1/guardrails/check"

# Relayed as they are: the guardrails service's refusal of a file its model cannot read (400; type, size
# and page count are checked here before it is called) and the outages of its model (503, 504).
# Anything else (401 wrong key, 422, 500) is a fault on our side and becomes 500.
PASSTHROUGH_STATUSES = (400, 503, 504)


@dataclass(frozen=True)
class GuardrailsThreshold:
    """The guardrails threshold the central orchestrator sent with one request: `target` accept = a page
    passes when its accept probability reaches `value`; reject = it is rejected when its reject probability
    does."""

    value: float
    target: Literal["accept", "reject"]


class GuardrailsClient:
    def __init__(self, client: RemoteModelClient):
        self._client = client

    async def check(
        self,
        request_id: str,
        filename: str,
        content_type: str | None,
        content: bytes,
        threshold: GuardrailsThreshold | None = None,
    ) -> dict[str, Any]:
        """The guardrails report of the document: `{passed, reason, document, pages}`. `threshold`: the one
        the central orchestrator sent with this request; None leaves the guardrails service's own in force.

        Not retried: the check runs inside the caller's time budget, and the caller may send the same
        request_id again (the pipeline is idempotent per request_id)."""
        data = {"request_id": request_id}
        if threshold is not None:
            data.update(threshold=str(threshold.value), threshold_target=threshold.target)
        body = await self._client.post_multipart(
            GUARDRAILS_CHECK_PATH,
            filename=filename or "upload",
            content=content,
            content_type=content_type or "application/octet-stream",
            data=data,
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
