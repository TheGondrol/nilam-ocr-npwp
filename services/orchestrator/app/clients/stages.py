from typing import Any
from urllib.parse import quote

from ocr_common.clients.remote import RemoteModelClient
from ocr_common.errors import InternalError, ServiceError
from ocr_common.pipeline import STAGE_OCR, STAGE_SCORING, STAGE_STRUCTURING
from ocr_common.testing_endpoints import testing_path

from app.config import Settings


class StageStatusClient:
    def __init__(self, stage: str, client: RemoteModelClient, jobs_path: str):
        self.stage = stage
        self._client = client
        self._jobs_path = jobs_path

    async def get(self, request_id: str) -> dict[str, Any] | None:
        """The stage's job (`status`, `result`, `error_message`, ...), or None when the stage has no job
        for this request_id yet (404)."""
        try:
            body = await self._client.get_json(f"{self._jobs_path}/{quote(request_id, safe='')}")
        except ServiceError as exc:
            if exc.status_code == 404:
                return None
            raise
        data = body.get("data") if isinstance(body, dict) else None
        if not isinstance(data, dict):
            raise InternalError(f"{self._client.name} returned an unexpected response")
        return data

    async def aclose(self) -> None:
        await self._client.aclose()


def build_stage_status_clients(settings: Settings, *, testing: bool = False) -> tuple[StageStatusClient, ...]:
    """The clients the pipeline waiter polls, one per stage; `testing=True` reads their `-test` jobs
    (TESTING_ENDPOINTS)."""

    def jobs_path(path: str) -> str:
        return testing_path(path) if testing else path

    def remote(base_url: str, api_key: str | None, timeout: float, name: str) -> RemoteModelClient:
        # Only 404 ("no job yet") is an answer; any other 4xx (a wrong key, say) is our own fault and must
        # not reach the central orchestrator as if it were theirs, so it becomes 500.
        return RemoteModelClient(
            base_url,
            timeout,
            name=name,
            headers={"X-API-Key": api_key or settings.api_key},
            passthrough_statuses=(404,),
        )

    return (
        StageStatusClient(
            STAGE_OCR,
            remote(
                settings.extraction_service_url,
                settings.extraction_api_key,
                settings.extraction_timeout_seconds,
                "extraction service",
            ),
            jobs_path("/v1/extraction/jobs"),
        ),
        StageStatusClient(
            STAGE_STRUCTURING,
            remote(
                settings.structuring_service_url,
                settings.structuring_api_key,
                settings.structuring_timeout_seconds,
                "structuring service",
            ),
            jobs_path("/v1/structuring/jobs"),
        ),
        StageStatusClient(
            STAGE_SCORING,
            remote(
                settings.scoring_service_url,
                settings.scoring_api_key,
                settings.scoring_timeout_seconds,
                "scoring service",
            ),
            jobs_path("/v1/scoring/jobs"),
        ),
    )
