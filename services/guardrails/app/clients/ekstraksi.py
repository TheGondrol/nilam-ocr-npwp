import json
from typing import Any

from ocr_common.clients.remote import RemoteModelClient
from ocr_common.errors import ServiceError
from ocr_common.pipeline import with_retry

from app.config import Settings

EKSTRAKSI_JOBS_PATH = "/v1/ekstraksi/jobs"


class EkstraksiJobClient:
    def __init__(self, client: RemoteModelClient, *, attempts: int = 3, delay: float = 0.5):
        self._client = client
        self._attempts = attempts
        self._delay = delay

    async def submit(
        self,
        request_id: str,
        document_type: str,
        guardrails: dict[str, Any],
        filename: str,
        content_type: str | None,
        content: bytes,
    ) -> dict[str, Any]:
        body = await with_retry(
            lambda: self._client.post_multipart(
                EKSTRAKSI_JOBS_PATH,
                filename=filename or "upload",
                content=content,
                content_type=content_type or "application/octet-stream",
                data={"request_id": request_id, "document_type": document_type, "guardrails": json.dumps(guardrails)},
            ),
            self._attempts,
            self._delay,
        )
        if not isinstance(body, dict) or not isinstance(body.get("data"), dict):
            raise ServiceError(500, f"{self._client.name} returned an unexpected response")
        return body["data"]

    async def aclose(self) -> None:
        await self._client.aclose()


def build_ekstraksi_client(settings: Settings) -> EkstraksiJobClient:
    client = RemoteModelClient(
        settings.ekstraksi_service_url,
        settings.ekstraksi_timeout_seconds,
        name="ekstraksi service",
        headers={"X-API-Key": settings.ekstraksi_api_key or settings.api_key},
        passthrough_client_errors=True,
    )
    return EkstraksiJobClient(
        client, attempts=settings.pipeline_retry_attempts, delay=settings.pipeline_retry_delay_seconds
    )
