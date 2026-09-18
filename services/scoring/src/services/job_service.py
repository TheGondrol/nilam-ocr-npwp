"""
Tahap SCORING di pipeline async, tahap terakhir: terima job dari
ServiceStructuring, jawab 202, lalu di background nilai dokumen, simpan
hasil, dan kirim callback yang membawa HASIL AKHIR ke Orkestrasi (yang
menyimpannya sebagai requests.final_result). Tidak ada handoff.
Mekanismenya ada di ocr_common/jobs.py.
"""

from typing import Any

from starlette.concurrency import run_in_threadpool

from ocr_common.jobs import StagePipeline
from src.services.scoring_service import ScoringService


class ScoringJobService:
    def __init__(self, pipeline: StagePipeline, scoring: ScoringService):
        self._pipeline = pipeline
        self._scoring = scoring

    async def submit(
        self,
        request_id: str,
        document_type: str,
        guardrails: dict[str, Any] | None,
        structuring: dict[str, Any],
    ) -> dict[str, Any]:
        fields = {
            name: {"value": field.get("value"), "confidence": field.get("confidence", 1.0)}
            for name, field in (structuring.get("fields") or {}).items()
        }

        async def work() -> dict[str, Any]:
            # Sinkron dan CPU-bound: di threadpool supaya event loop tetap menerima job lain.
            return await run_in_threadpool(self._scoring.score, document_type, fields)

        def final_result(scoring: dict[str, Any]) -> dict[str, Any]:
            return {"document_type": document_type, "fields": fields, "scoring": scoring, "guardrails": guardrails}

        return await self._pipeline.submit(request_id, work, callback_result=final_result)

    async def get(self, request_id: str) -> dict[str, Any]:
        return await self._pipeline.get(request_id)
