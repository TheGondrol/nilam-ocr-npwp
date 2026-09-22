from typing import Any

from starlette.concurrency import run_in_threadpool

from ocr_common.errors import ServiceError
from ocr_common.jobs import StagePipeline
from ocr_common.npwp import DOCUMENT_TYPE, contract_fields, final_result
from src.services.confidence_service import ConfidenceService


class ScoringJobService:
    def __init__(self, pipeline: StagePipeline, confidence: ConfidenceService, confidence_threshold: float = 0.5):
        self._pipeline = pipeline
        self._confidence = confidence
        self._confidence_threshold = confidence_threshold

    async def submit(
        self,
        request_id: str,
        document_type: str,
        guardrails: dict[str, Any] | None,
        ocr: dict[str, Any] | None,
        structuring: dict[str, Any],
    ) -> dict[str, Any]:
        async def work() -> dict[str, Any]:
            if document_type != DOCUMENT_TYPE:
                raise ServiceError(400, f"Unsupported document_type: {document_type}. Supported: ['{DOCUMENT_TYPE}']")
            payload = self._confidence.payload_from_chain(guardrails, ocr, structuring)
            result = await run_in_threadpool(self._confidence.predict, payload)
            return {**result, "payload": payload}

        def final(scoring: dict[str, Any]) -> dict[str, Any]:
            return final_result(document_type, guardrails, structuring, scoring)

        return await self._pipeline.submit(
            request_id,
            work,
            callback_result=final,
            outcome_data=lambda scoring: contract_fields(final(scoring), self._confidence_threshold),
        )

    async def get(self, request_id: str) -> dict[str, Any]:
        return await self._pipeline.get(request_id)
