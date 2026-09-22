from typing import Any
from urllib.parse import quote

from ocr_common.clients.remote import RemoteModelClient
from ocr_common.errors import ServiceError
from ocr_common.pipeline import STAGE_OCR, STAGE_SCORING, STAGE_STRUCTURING

from app.config import Settings


class StageStatusClient:
    def __init__(self, stage: str, client: RemoteModelClient, jobs_path: str):
        self.stage = stage
        self._client = client
        self._jobs_path = jobs_path

    async def get(self, request_id: str) -> dict[str, Any] | None:
        try:
            body = await self._client.get_json(f"{self._jobs_path}/{quote(request_id, safe='')}")
        except ServiceError as exc:
            if exc.status_code == 404:
                return None
            raise
        data = body.get("data") if isinstance(body, dict) else None
        if not isinstance(data, dict):
            raise ServiceError(500, f"{self._client.name} returned an unexpected response")
        return data

    async def aclose(self) -> None:
        await self._client.aclose()


def build_stage_status_clients(settings: Settings) -> tuple[StageStatusClient, ...]:
    def remote(base_url: str, api_key: str | None, timeout: float, name: str) -> RemoteModelClient:
        return RemoteModelClient(
            base_url,
            timeout,
            name=name,
            headers={"X-API-Key": api_key or settings.api_key},
            passthrough_client_errors=True,
        )

    return (
        StageStatusClient(
            STAGE_OCR,
            remote(
                settings.ekstraksi_service_url,
                settings.ekstraksi_api_key,
                settings.ekstraksi_timeout_seconds,
                "ekstraksi service",
            ),
            "/v1/ekstraksi/jobs",
        ),
        StageStatusClient(
            STAGE_STRUCTURING,
            remote(
                settings.structuring_service_url,
                settings.structuring_api_key,
                settings.structuring_timeout_seconds,
                "structuring service",
            ),
            "/v1/structuring/jobs",
        ),
        StageStatusClient(
            STAGE_SCORING,
            remote(
                settings.scoring_service_url,
                settings.scoring_api_key,
                settings.scoring_timeout_seconds,
                "scoring service",
            ),
            "/v1/scoring/jobs",
        ),
    )
