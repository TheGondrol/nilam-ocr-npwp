from dataclasses import dataclass
from typing import Any

from ocr_common.clients.remote import RemoteModelClient
from ocr_common.errors import ServiceError

from app.config import Settings


def _data(body: Any, name: str) -> dict[str, Any]:
    if not isinstance(body, dict) or not isinstance(body.get("data"), dict):
        raise ServiceError(500, f"{name} returned an unexpected response")
    return body["data"]


class GuardrailsClient:
    def __init__(self, client: RemoteModelClient):
        self._client = client

    async def check(self, request_id: str, filename: str, content_type: str | None, content: bytes) -> dict[str, Any]:
        body = await self._client.post_multipart(
            "/v1/guardrails/check",
            filename=filename or "upload",
            content=content,
            content_type=content_type or "image/jpeg",
            data={"request_id": request_id},
        )
        return _data(body, self._client.name)


class StructuringClient:
    def __init__(self, client: RemoteModelClient):
        self._client = client

    async def structure(self, lines: list[dict[str, Any]]) -> dict[str, Any]:
        body = await self._client.post_json("/v1/structuring/structure", {"lines": lines})
        return _data(body, self._client.name)


class ScoringClient:
    def __init__(self, client: RemoteModelClient):
        self._client = client

    async def score(self, document_type: str, fields: dict[str, Any]) -> dict[str, Any]:
        payload = {
            "document_type": document_type,
            "fields": {
                name: {"value": f.get("value"), "confidence": f.get("confidence", 1.0)} for name, f in fields.items()
            },
        }
        body = await self._client.post_json("/v1/scoring/score", payload)
        return _data(body, self._client.name)


@dataclass(frozen=True)
class StageClients:
    guardrails: GuardrailsClient
    structuring: StructuringClient
    scoring: ScoringClient

    async def aclose(self) -> None:
        for stage in (self.guardrails, self.structuring, self.scoring):
            await stage._client.aclose()


def _remote(base_url: str, api_key: str, timeout: float, name: str) -> RemoteModelClient:
    return RemoteModelClient(
        base_url,
        timeout,
        name=name,
        headers={"X-API-Key": api_key},
        passthrough_client_errors=True,
    )


def build_stage_clients(settings: Settings) -> StageClients:
    own_key = settings.api_key
    return StageClients(
        guardrails=GuardrailsClient(
            _remote(
                settings.guardrails_service_url,
                settings.guardrails_api_key or own_key,
                settings.guardrails_timeout_seconds,
                "guardrails service",
            )
        ),
        structuring=StructuringClient(
            _remote(
                settings.structuring_service_url,
                settings.structuring_api_key or own_key,
                settings.structuring_timeout_seconds,
                "structuring service",
            )
        ),
        scoring=ScoringClient(
            _remote(
                settings.scoring_service_url,
                settings.scoring_api_key or own_key,
                settings.scoring_timeout_seconds,
                "scoring service",
            )
        ),
    )
