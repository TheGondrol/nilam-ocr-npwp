import logging
import time
from typing import Any

from prometheus_client import Counter

from ocr_common.errors import NotFound
from ocr_common.npwp import DOCUMENT_TYPE, final_result
from ocr_common.pipeline import STAGE_SCORING, STAGE_STRUCTURING, STATUS_DONE

from app.clients.extraction import ExtractionJobClient
from app.clients.guardrails import GuardrailsClient
from app.config import Settings
from app.services.document_checks import check_document
from app.services.pipeline_waiter import PipelineWait, WaitOutcome

logger = logging.getLogger(__name__)

GUARDRAILS_SKIPPED = Counter(
    "guardrails_skipped_total", "Documents sent into the pipeline without the guardrails model (skip_guardrails)"
)


class ExtractOcrService:
    """The pipeline behind `extract-ocr`: the guardrails check, the hand-off to the OCR stage, and the
    wait for OCR -> structuring -> scoring."""

    def __init__(
        self, guardrails: GuardrailsClient, extraction: ExtractionJobClient, waiter: PipelineWait, settings: Settings
    ):
        self._guardrails = guardrails
        self._extraction = extraction
        self._waiter = waiter
        self._settings = settings

    async def submit(
        self,
        request_id: str,
        document_type: str,
        filename: str,
        content_type: str | None,
        content: bytes,
        *,
        received_at: float | None = None,
        file_url: str | None = None,
        skip_guardrails: bool = False,
    ) -> dict[str, Any]:
        """Check the file (type, empty, `MAX_UPLOAD_BYTES`, `MAX_DOCUMENT_PAGES`) before anyone else sees it,
        judge it with the guardrails model, hand it to the OCR stage when it passes, and wait for the
        pipeline for what is left of `PIPELINE_WAIT_SECONDS` since `received_at`.

        `skip_guardrails` leaves the guardrails model out (the file checks still run): the stages get no
        guardrails report, so scoring imputes its guardrail probability, and the structuring rules still
        reject."""
        started = time.monotonic() if received_at is None else received_at
        check_document(content_type, content, self._settings)
        if skip_guardrails:
            logger.warning("guardrails skipped for request_id %s (skip_guardrails=true)", request_id)
            GUARDRAILS_SKIPPED.inc()
            report = None
            verdict: dict[str, Any] = {"passed": True, "reason": None}
        else:
            report = await self._guardrails.check(request_id, filename, content_type, content)
            if not report["passed"]:
                return {**report, "job": None, "pipeline": None, "result": None}
            verdict = report

        job = await self._extraction.submit(
            request_id, document_type, report, filename, content_type, content, file_url=file_url
        )
        wait_seconds = self._settings.pipeline_wait_seconds
        if wait_seconds <= 0:
            return {**verdict, "job": job, "pipeline": None, "result": None}

        remaining = wait_seconds - (time.monotonic() - started)
        outcome = await self._waiter.wait(request_id, remaining)
        return {**verdict, "job": job, **_pipeline(document_type, report, outcome)}

    async def status(self, request_id: str) -> dict[str, Any]:
        """Where the request is now, read from the stages without waiting; 404 when it never entered the
        pipeline. The guardrails report is not stored, so it is not part of the result; only a document
        that passed guardrails has stage jobs at all."""
        outcome = await self._waiter.snapshot(request_id)
        if outcome is None:
            raise NotFound(f"No request found for request_id {request_id}")
        return {"passed": True, "reason": None, **_pipeline(DOCUMENT_TYPE, None, outcome)}


def _pipeline(document_type: str, report: dict[str, Any] | None, outcome: WaitOutcome) -> dict[str, Any]:
    result = None
    if outcome.status == STATUS_DONE:
        result = final_result(document_type, report, outcome.results[STAGE_STRUCTURING], outcome.results[STAGE_SCORING])
    pipeline = {"stage": outcome.stage, "status": outcome.status, "error_message": outcome.error_message}
    return {"pipeline": pipeline, "result": result}
