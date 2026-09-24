import json
from typing import Any

from ocr_common.clients.remote import RemoteModelClient
from ocr_common.errors import InternalError
from ocr_common.pipeline import with_retry
from ocr_common.testing_endpoints import testing_path

from app.config import Settings

EKSTRAKSI_JOBS_PATH = "/v1/ekstraksi/jobs"


class EkstraksiJobClient:
    def __init__(
        self,
        client: RemoteModelClient,
        *,
        attempts: int = 3,
        delay: float = 0.5,
        jobs_path: str = EKSTRAKSI_JOBS_PATH,
    ):
        self._client = client
        self._jobs_path = jobs_path
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
        *,
        file_url: str | None = None,
    ) -> dict[str, Any]:
        """Hand the document to the OCR stage. When the request came as `file_url`, that URL is forwarded
        instead of the bytes: the OCR service downloads it itself, and a job left behind by a dead process
        can be run again from the URL stored with the job."""
        fields = {"request_id": request_id, "document_type": document_type, "guardrails": json.dumps(guardrails)}
        if file_url:
            call = lambda: self._client.post_form(self._jobs_path, data={**fields, "file_url": file_url})  # noqa: E731
        else:
            call = lambda: self._client.post_multipart(  # noqa: E731
                self._jobs_path,
                filename=filename or "upload",
                content=content,
                content_type=content_type or "application/octet-stream",
                data=fields,
            )
        body = await with_retry(call, self._attempts, self._delay)
        if not isinstance(body, dict) or not isinstance(body.get("data"), dict):
            raise InternalError(f"{self._client.name} returned an unexpected response")
        return body["data"]

    async def aclose(self) -> None:
        await self._client.aclose()


def build_ekstraksi_client(settings: Settings, *, testing: bool = False) -> EkstraksiJobClient:
    """The client to the OCR stage; `testing=True` submits to its `-test` endpoint (TESTING_ENDPOINTS)."""
    client = RemoteModelClient(
        settings.ekstraksi_service_url,
        settings.ekstraksi_timeout_seconds,
        name="ekstraksi service (testing)" if testing else "ekstraksi service",
        headers={"X-API-Key": settings.ekstraksi_api_key or settings.api_key},
        passthrough_client_errors=True,
    )
    return EkstraksiJobClient(
        client,
        attempts=settings.pipeline_retry_attempts,
        delay=settings.pipeline_retry_delay_seconds,
        jobs_path=testing_path(EKSTRAKSI_JOBS_PATH) if testing else EKSTRAKSI_JOBS_PATH,
    )
