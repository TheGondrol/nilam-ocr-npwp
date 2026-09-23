"""`StagePipeline`: how one stage runs a job. Claim it, answer 202, do the work in the background,
store the result, notify the orchestrator and hand the job to the next stage (directly, or through
the outbox), and report every failure as a `FAILED` job.
"""

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from ocr_common.errors import InternalError, NotFound, ServiceError
from ocr_common.pipeline import metrics
from ocr_common.pipeline.callbacks import NextStage, StageCallback
from ocr_common.pipeline.outbox import Outbox, OutboxMessage, OutboxRelay, callback_message, handoff_message
from ocr_common.pipeline.repository import STATUS_DONE, STATUS_FAILED, STATUS_PROCESSING, JobRepository
from ocr_common.pipeline.runner import BackgroundRunner
from ocr_common.web.request_id import bind_request_id, reset_request_id

logger = logging.getLogger(__name__)

STAGE_OCR = "OCR"
STAGE_STRUCTURING = "STRUCTURING"
STAGE_SCORING = "SCORING"

Work = Callable[[], Awaitable[Mapping[str, Any]]]
"""The job itself: returns the stage result (a TypedDict of `ocr_common.types`, or any mapping)."""
CallbackResult = Callable[[Mapping[str, Any]], Mapping[str, Any]]
"""Derives what the callback / the orchestrator's outcome row carries from the stage result."""
HandoffPayload = Callable[[Mapping[str, Any]], Mapping[str, Any]]
"""Derives the body of the hand-off to the next stage from the stage result."""
Rejection = Callable[[Mapping[str, Any]], str | None]
"""The reason the stage result rejects the document, or None. A rejected job is still stored `DONE`
(its result stays readable), but it is not handed on, its callback is `FAILED` with the reason, and
the outcome row / event log record a 400."""


class StagePipeline:
    """The per-service orchestration of one job; built by `pipeline.factory.build_stage_pipeline`."""

    def __init__(
        self,
        *,
        stage: str,
        repository: JobRepository,
        callback: StageCallback,
        next_stage_client: NextStage | None = None,
        outbox: Outbox | None = None,
        runner: BackgroundRunner | None = None,
        callbacks: bool = True,
    ):
        """`callbacks=False` when the orchestrator has no callback endpoint: no callback is sent or
        queued, and the outcome reaches the orchestrator through its table (ORCHESTRATION_OUTCOME_TABLE)
        and GET .../jobs/{request_id}."""
        self.stage = stage
        self.repository = repository
        self.callback = callback
        self.next_stage_client = next_stage_client
        self.outbox = outbox
        self.runner = runner or BackgroundRunner()
        self.callbacks = callbacks

    async def submit(
        self,
        request_id: str,
        work: Work,
        *,
        handoff_payload: HandoffPayload | None = None,
        next_stage: str | None = None,
        callback_result: CallbackResult | None = None,
        outcome_data: CallbackResult | None = None,
        rejection: Rejection | None = None,
        input: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """`input` is what a later run of this job needs besides the earlier stages' stored results
        (document_type, guardrails report, file_url); the stale-job reaper hands it back to `resume`."""
        claimed = await self.repository.claim(request_id, input=input)
        status = STATUS_PROCESSING
        if claimed:
            self.runner.spawn(
                self._run(request_id, work, handoff_payload, next_stage, callback_result, outcome_data, rejection)
            )
        else:
            record = await self.repository.get(request_id)
            status = record["status"] if record else STATUS_PROCESSING
        return {"request_id": request_id, "stage": self.stage, "status": status, "duplicate": not claimed}

    async def resume(
        self,
        request_id: str,
        work: Work,
        *,
        handoff_payload: HandoffPayload | None = None,
        next_stage: str | None = None,
        callback_result: CallbackResult | None = None,
        outcome_data: CallbackResult | None = None,
        rejection: Rejection | None = None,
    ) -> None:
        """Run a job that is already claimed (by the stale-job reaper) without claiming it again."""
        self.runner.spawn(
            self._run(request_id, work, handoff_payload, next_stage, callback_result, outcome_data, rejection)
        )

    async def get(self, request_id: str) -> dict[str, Any]:
        """The job's record for `GET /v1/<stage>/jobs/{request_id}`; `NotFound` when unknown."""
        record = await self.repository.get(request_id)
        if record is None:
            raise NotFound(f"No {self.stage} job found for request_id: {request_id}")
        return {"stage": self.stage, **record}

    async def aclose(self, drain_timeout: float, *, relay: OutboxRelay | None = None) -> None:
        """Shutdown order: finish the jobs, then let the relay send what those jobs queued, then close
        the clients the relay uses."""
        await self.runner.drain(drain_timeout)
        if relay is not None:
            await relay.stop()
        await self.callback.aclose()

    async def _run(
        self,
        request_id: str,
        work: Work,
        handoff_payload: HandoffPayload | None,
        next_stage: str | None,
        callback_result: CallbackResult | None,
        outcome_data: CallbackResult | None = None,
        rejection: Rejection | None = None,
    ) -> None:
        token = bind_request_id(request_id)  # log lines and downstream calls of this job carry its id
        try:
            await self._run_bound(
                request_id, work, handoff_payload, next_stage, callback_result, outcome_data, rejection
            )
        finally:
            reset_request_id(token)

    async def _run_bound(
        self,
        request_id: str,
        work: Work,
        handoff_payload: HandoffPayload | None,
        next_stage: str | None,
        callback_result: CallbackResult | None,
        outcome_data: CallbackResult | None,
        rejection: Rejection | None = None,
    ) -> None:
        payload: dict[str, Any] | None = None
        final: dict[str, Any] | None = None
        reason: str | None = None
        started = time.perf_counter()
        try:
            result = dict(await work())
            reason = rejection(result) if rejection else None
            if reason is None:
                payload = dict(handoff_payload(result)) if handoff_payload else None
                final = dict(callback_result(result)) if callback_result else None
            await self.repository.complete(
                request_id,
                result,
                outcome_data=dict(outcome_data(result)) if outcome_data and reason is None else None,
                rejection=reason,
                messages=self._messages(request_id, final, payload, next_stage, reason),
            )
            metrics.JOB_DURATION.labels(self.stage).observe(time.perf_counter() - started)
            metrics.JOBS.labels(self.stage, metrics.OUTCOME_REJECTED if reason else metrics.OUTCOME_DONE).inc()
        except asyncio.CancelledError:
            logger.warning("%s job %s interrupted by shutdown", self.stage, request_id)
            metrics.JOBS.labels(self.stage, metrics.OUTCOME_INTERRUPTED).inc()
            await self._failed(
                request_id, f"{self.stage} stage was interrupted by a service shutdown; submit the job again"
            )
            raise
        except ServiceError as exc:
            metrics.JOBS.labels(self.stage, metrics.OUTCOME_FAILED).inc()
            await self._failed(request_id, exc.message)
            return
        except Exception:
            logger.exception("%s job %s crashed", self.stage, request_id)
            metrics.JOBS.labels(self.stage, metrics.OUTCOME_CRASHED).inc()
            await self._failed(request_id, f"Internal error in {self.stage} stage")
            return

        if self.outbox is not None:
            return

        try:
            if self.callbacks and reason is not None:
                await self.callback.notify(request_id, self.stage, STATUS_FAILED, error_message=reason)
            elif self.callbacks:
                await self.callback.notify(request_id, self.stage, STATUS_DONE, result=final)
            if payload is not None:
                await self._hand_off(payload)
        except asyncio.CancelledError:
            if payload is not None:
                await self._handoff_failed(request_id, next_stage, "interrupted by a service shutdown")
            raise
        except ServiceError as exc:
            await self._handoff_failed(request_id, next_stage, exc.message)

    def _messages(
        self,
        request_id: str,
        final: dict[str, Any] | None,
        payload: dict[str, Any] | None,
        next_stage: str | None,
        reason: str | None = None,
    ) -> list[OutboxMessage]:
        if self.outbox is None:
            return []
        messages = []
        if self.callbacks and reason is not None:
            messages.append(callback_message(request_id, self.stage, STATUS_FAILED, error_message=reason))
        elif self.callbacks:
            messages.append(callback_message(request_id, self.stage, STATUS_DONE, result=final))
        if payload is not None and next_stage is not None:
            messages.append(handoff_message(next_stage, payload))
        return messages

    async def _hand_off(self, payload: dict[str, Any]) -> None:
        if self.next_stage_client is None:
            raise InternalError(f"the {self.stage} stage has no next stage to hand off to")
        await self.next_stage_client.submit(payload)

    async def _handoff_failed(self, request_id: str, next_stage: str | None, reason: str) -> None:
        logger.error("%s job %s: handoff to %s failed: %s", self.stage, request_id, next_stage, reason)
        message = f"Handoff to {next_stage} failed: {reason}"
        try:
            await self.repository.handoff_failed(request_id, next_stage or self.stage, message)
        except Exception:
            logger.exception("%s job %s: could not record the failed handoff", self.stage, request_id)
        if self.callbacks:
            await self.callback.notify(request_id, next_stage or self.stage, STATUS_FAILED, error_message=message)

    async def _failed(self, request_id: str, error_message: str) -> None:
        reported = self.outbox is not None
        try:
            await self.repository.fail(
                request_id,
                error_message,
                messages=[callback_message(request_id, self.stage, STATUS_FAILED, error_message=error_message)]
                if self.outbox is not None and self.callbacks
                else [],
            )
        except Exception:
            logger.exception("%s job %s: could not record failure", self.stage, request_id)
            reported = False
        if not reported and self.callbacks:
            await self.callback.notify(request_id, self.stage, STATUS_FAILED, error_message=error_message)
