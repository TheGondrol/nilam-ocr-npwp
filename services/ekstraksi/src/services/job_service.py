from typing import Any

from ocr_common.errors import ServiceError
from ocr_common.fetch_url import STRICT_URL_POLICY, FetchUrlError, UrlPolicy, fetch
from ocr_common.jobs import STAGE_STRUCTURING, StagePipeline
from src.services.ekstraksi_service import EkstraksiService

UploadedFile = tuple[bytes, str, str | None]
Source = UploadedFile | str


class EkstraksiJobService:
    def __init__(
        self,
        pipeline: StagePipeline,
        ekstraksi: EkstraksiService,
        max_upload_bytes: int,
        url_policy: UrlPolicy = STRICT_URL_POLICY,
    ):
        self._pipeline = pipeline
        self._ekstraksi = ekstraksi
        self._max_upload_bytes = max_upload_bytes
        self._url_policy = url_policy

    async def submit(
        self, request_id: str, document_type: str, guardrails: dict[str, Any] | None, source: Source
    ) -> dict[str, Any]:
        async def work() -> dict[str, Any]:
            content, filename, content_type = await self._load(source)
            return await self._ekstraksi.extract(filename, content_type, content)

        def handoff(ocr: dict[str, Any]) -> dict[str, Any]:
            return {"request_id": request_id, "document_type": document_type, "guardrails": guardrails, "ocr": ocr}

        return await self._pipeline.submit(request_id, work, handoff_payload=handoff, next_stage=STAGE_STRUCTURING)

    async def get(self, request_id: str) -> dict[str, Any]:
        return await self._pipeline.get(request_id)

    async def _load(self, source: Source) -> UploadedFile:
        if not isinstance(source, str):
            return source
        try:
            return await fetch(source, limit=self._max_upload_bytes, policy=self._url_policy)
        except FetchUrlError as exc:
            raise ServiceError(400, str(exc)) from exc
