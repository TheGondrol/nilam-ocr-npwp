from typing import Any

from starlette.concurrency import run_in_threadpool

from ocr_common.jobs import STAGE_SCORING, NextStage, StagePipeline
from src.services.structuring_service import StructuringService


class StructuringJobService:
    def __init__(self, pipeline: StagePipeline, structuring: StructuringService, next_stage: NextStage):
        self._pipeline = pipeline
        self._structuring = structuring
        self._next_stage = next_stage

    async def submit(
        self, request_id: str, document_type: str, guardrails: dict[str, Any] | None, ocr: dict[str, Any]
    ) -> dict[str, Any]:
        async def work() -> dict[str, Any]:
            lines = [
                {
                    "text": block.get("text") or "",
                    "confidence": block.get("confidence", 1.0),
                    "bbox": block.get("bbox"),
                    "page": block.get("page", 0),
                }
                for block in ocr.get("blocks") or []
            ]
            return await run_in_threadpool(self._structuring.structure, lines)

        async def handoff(structuring: dict[str, Any]) -> None:
            await self._next_stage.submit(
                {
                    "request_id": request_id,
                    "document_type": document_type,
                    "guardrails": guardrails,
                    "ocr": ocr,
                    "structuring": structuring,
                }
            )

        return await self._pipeline.submit(request_id, work, handoff=handoff, next_stage=STAGE_SCORING)

    async def get(self, request_id: str) -> dict[str, Any]:
        return await self._pipeline.get(request_id)
