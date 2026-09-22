import time
from typing import Any

from ocr_common.jobs import STAGE_SCORING, STAGE_STRUCTURING, STATUS_DONE
from ocr_common.npwp import final_result
from src.clients.ekstraksi import EkstraksiJobClient
from src.services.guardrails_service import GuardrailsService
from src.services.pipeline_waiter import PipelineWait


class GuardrailsJobService:
    def __init__(
        self,
        guardrails: GuardrailsService,
        ekstraksi: EkstraksiJobClient,
        waiter: PipelineWait | None = None,
        *,
        wait_seconds: float = 0.0,
    ):
        self._guardrails = guardrails
        self._ekstraksi = ekstraksi
        self._waiter = waiter
        self._wait_seconds = wait_seconds

    async def submit(
        self,
        request_id: str,
        document_type: str,
        filename: str,
        content_type: str | None,
        content: bytes,
        *,
        received_at: float | None = None,
    ) -> dict[str, Any]:
        started = time.monotonic() if received_at is None else received_at
        report = await self._guardrails.check(filename, content_type, content)
        if not report["passed"]:
            return {**report, "job": None, "pipeline": None, "result": None}

        job = await self._ekstraksi.submit(request_id, document_type, report, filename, content_type, content)
        if self._waiter is None or self._wait_seconds <= 0:
            return {**report, "job": job, "pipeline": None, "result": None}

        remaining = self._wait_seconds - (time.monotonic() - started)
        outcome = await self._waiter.wait(request_id, remaining)
        result = None
        if outcome.status == STATUS_DONE:
            result = final_result(
                document_type, report, outcome.results[STAGE_STRUCTURING], outcome.results[STAGE_SCORING]
            )
        pipeline = {"stage": outcome.stage, "status": outcome.status, "error_message": outcome.error_message}
        return {**report, "job": job, "pipeline": pipeline, "result": result}
