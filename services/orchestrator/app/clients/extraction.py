import json
from collections.abc import Sequence
from typing import Any

from ocr_common.clients.remote import RemoteModelClient
from ocr_common.errors import InternalError
from ocr_common.pipeline import with_retry
from ocr_common.testing_endpoints import testing_path

from app.config import Settings

EXTRACTION_JOBS_PATH = "/v1/extraction/jobs"


class ExtractionJobClient:
    def __init__(
        self,
        client: RemoteModelClient,
        *,
        attempts: int = 3,
        delay: float = 0.5,
        jobs_path: str = EXTRACTION_JOBS_PATH,
    ):
        self._client = client
        self._jobs_path = jobs_path
        self._attempts = attempts
        self._delay = delay

    async def submit(
        self,
        request_id: str,
        document_type: str,
        guardrails: dict[str, Any] | None,
        filename: str,
        content_type: str | None,
        content: bytes,
        *,
        file_url: str | None = None,
        sequence: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        """Hand the document to the OCR stage. When the request came as `file_url`, that URL is forwarded
        instead of the bytes: the OCR service downloads it itself, and a job left behind by a dead process
        can be run again from the URL stored with the job. `guardrails` is None when the check was skipped:
        the field is then left out (extraction refuses a `guardrails` that is not a JSON object).
        `sequence` (pipeline_name_sequence) goes along as a JSON array, so each stage knows whether to hand
        the job on."""
        fields = {"request_id": request_id, "document_type": document_type}
        if guardrails is not None:
            fields["guardrails"] = json.dumps(guardrails)
        if sequence:
            fields["pipeline_name_sequence"] = json.dumps(list(sequence))
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


def build_extraction_client(settings: Settings, *, testing: bool = False) -> ExtractionJobClient:
    """The client to the OCR stage; `testing=True` submits to its `-test` endpoint (TESTING_ENDPOINTS)."""
    client = RemoteModelClient(
        settings.extraction_service_url,
        settings.extraction_timeout_seconds,
        name="extraction service (testing)" if testing else "extraction service",
        headers={"X-API-Key": settings.extraction_api_key or settings.api_key},
        passthrough_client_errors=True,
    )
    return ExtractionJobClient(
        client,
        attempts=settings.pipeline_retry_attempts,
        delay=settings.pipeline_retry_delay_seconds,
        jobs_path=testing_path(EXTRACTION_JOBS_PATH) if testing else EXTRACTION_JOBS_PATH,
    )
