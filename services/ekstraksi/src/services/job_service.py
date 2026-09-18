"""
Tahap OCR di pipeline async ("ServiceOCR" di sequence diagram): terima job
dari Orkestrasi, jawab 202, lalu di background ambil file (payload atau
unduh dari MinIO), jalankan OCR, simpan hasil, callback, dan serahkan ke
structuring. Mekanismenya ada di ocr_common/jobs.py; di sini hanya kerja
tahap ini dan bentuk payload untuk tahap berikutnya.
"""

from typing import Any

from ocr_common.errors import ServiceError
from ocr_common.fetch_url import FetchUrlError, fetch
from ocr_common.jobs import STAGE_STRUCTURING, NextStage, StagePipeline
from src.services.ekstraksi_service import EkstraksiService

# File yang sudah dibaca dari payload, atau URL yang diunduh saat job jalan.
UploadedFile = tuple[bytes, str, str | None]
Source = UploadedFile | str


class EkstraksiJobService:
    def __init__(
        self,
        pipeline: StagePipeline,
        ekstraksi: EkstraksiService,
        next_stage: NextStage,
        max_upload_bytes: int,
    ):
        self._pipeline = pipeline
        self._ekstraksi = ekstraksi
        self._next_stage = next_stage
        self._max_upload_bytes = max_upload_bytes

    async def submit(
        self, request_id: str, document_type: str, guardrails: dict[str, Any] | None, source: Source
    ) -> dict[str, Any]:
        async def work() -> dict[str, Any]:
            content, filename, content_type = await self._load(source)
            return await self._ekstraksi.extract(filename, content_type, content)

        async def handoff(ocr: dict[str, Any]) -> None:
            await self._next_stage.submit(
                {"request_id": request_id, "document_type": document_type, "guardrails": guardrails, "ocr": ocr}
            )

        return await self._pipeline.submit(request_id, work, handoff=handoff, next_stage=STAGE_STRUCTURING)

    async def get(self, request_id: str) -> dict[str, Any]:
        return await self._pipeline.get(request_id)

    async def _load(self, source: Source) -> UploadedFile:
        if not isinstance(source, str):
            return source
        try:
            return await fetch(source, limit=self._max_upload_bytes)
        except FetchUrlError as exc:
            raise ServiceError(400, str(exc)) from exc
