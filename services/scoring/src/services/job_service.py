from typing import Any

from starlette.concurrency import run_in_threadpool

from ocr_common.errors import ServiceError
from ocr_common.jobs import StagePipeline
from ocr_common.npwp import DOCUMENT_TYPE
from src.services.confidence_service import ConfidenceService


class ScoringJobService:
    def __init__(self, pipeline: StagePipeline, confidence: ConfidenceService):
        self._pipeline = pipeline
        self._confidence = confidence

    async def submit(
        self,
        request_id: str,
        document_type: str,
        guardrails: dict[str, Any] | None,
        ocr: dict[str, Any] | None,
        structuring: dict[str, Any],
    ) -> dict[str, Any]:
        fields = {
            name: {"value": field.get("value"), "confidence": field.get("confidence", 1.0)}
            for name, field in (structuring.get("fields") or {}).items()
        }

        async def work() -> dict[str, Any]:
            if document_type != DOCUMENT_TYPE:
                raise ServiceError(400, f"Unsupported document_type: {document_type}. Supported: ['{DOCUMENT_TYPE}']")
            payload = self._confidence.payload_from_chain(guardrails, ocr, structuring)
            result = await run_in_threadpool(self._confidence.predict, payload)
            return {**result, "payload": payload}

        def final_result(scoring: dict[str, Any]) -> dict[str, Any]:
            return {
                "document_type": document_type,
                "fields": fields,
                "scoring": {key: scoring[key] for key in ("npwp_confidence", "name_confidence")},
                "guardrails": guardrails,
            }

        return await self._pipeline.submit(request_id, work, callback_result=final_result)

    async def get(self, request_id: str) -> dict[str, Any]:
        return await self._pipeline.get(request_id)
