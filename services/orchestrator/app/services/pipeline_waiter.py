import asyncio
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from ocr_common.errors import InternalError, ServiceError
from ocr_common.pipeline import (
    STAGE_OF,
    STATUS_DONE,
    STATUS_FAILED,
    STATUS_PROCESSING,
    InvalidSequence,
    validate_sequence,
)

logger = logging.getLogger(__name__)

# The pipeline stopped because a stage rejected the document (a rejecting check of the structuring
# rules: `reject_reason` in its result). Final, like FAILED, but answered as a 400.
STATUS_REJECTED = "REJECTED"


class StageStatus(Protocol):
    stage: str

    async def get(self, request_id: str) -> dict[str, Any] | None: ...


@dataclass(frozen=True)
class WaitOutcome:
    stage: str
    status: str
    error_message: str | None = None
    results: dict[str, dict[str, Any]] = field(default_factory=dict)


class PipelineWait(Protocol):
    async def wait(self, request_id: str, timeout: float, *, last_stage: str | None = None) -> WaitOutcome: ...

    async def snapshot(self, request_id: str) -> WaitOutcome | None: ...


class PipelineWaiter:
    """Reads the jobs of the stages, in pipeline order, to tell where a request is."""

    def __init__(self, stages: Sequence[StageStatus], *, poll_interval: float):
        self._stages = tuple(stages)
        self._poll_interval = poll_interval

    async def wait(self, request_id: str, timeout: float, *, last_stage: str | None = None) -> WaitOutcome:
        """Polls until the pipeline ends (DONE, FAILED or rejected) or `timeout` runs out (PROCESSING).
        `last_stage` is the stage that ends this request (the last of its pipeline_name_sequence); None: the
        last stage there is."""
        results: dict[str, dict[str, Any]] = {}
        stages = self._through(last_stage)
        current = stages[0].stage
        if timeout <= 0:
            return WaitOutcome(current, STATUS_PROCESSING, results=results)
        try:
            async with asyncio.timeout(timeout):
                for stage in stages:
                    current = stage.stage
                    record = await self._until_finished(stage, request_id)
                    ended = _ended(current, record, results)
                    if ended is not None:
                        return ended
        except TimeoutError:
            return WaitOutcome(current, STATUS_PROCESSING, results=results)
        return WaitOutcome(current, STATUS_DONE, results=results)

    async def snapshot(self, request_id: str) -> WaitOutcome | None:
        """Where the request is now, reading each stage at most once; None when the first stage has no job
        for it (the request never entered the pipeline, or is still being judged by guardrails).

        A stage that cannot be read raises (503/504/500) instead of being reported as still running. A
        hand-off that failed for good (retries exhausted, or an outbox dead letter) leaves the next stage
        without a job, so it reads as PROCESSING: the FAILED callback and the orchestrator's tables hold
        that final state.

        The stage that ends the request comes from the pipeline_name_sequence stored with the first stage's
        job (this service keeps nothing itself)."""
        results: dict[str, dict[str, Any]] = {}
        record = await self._stages[0].get(request_id)
        if record is None:
            return None
        stages = self._through(_last_stage(record.get("pipeline_name_sequence")))
        for index, stage in enumerate(stages):
            if index > 0:
                record = await stage.get(request_id)
                if record is None:
                    return WaitOutcome(stage.stage, STATUS_PROCESSING, results=results)
            status = record.get("status")
            if status not in (STATUS_PROCESSING, STATUS_DONE, STATUS_FAILED):
                raise InternalError(f"{stage.stage} job has an unexpected status: {status}")
            if status == STATUS_PROCESSING:
                return WaitOutcome(stage.stage, STATUS_PROCESSING, results=results)
            ended = _ended(stage.stage, record, results)
            if ended is not None:
                return ended
        return WaitOutcome(stages[-1].stage, STATUS_DONE, results=results)

    def _through(self, last_stage: str | None) -> tuple[StageStatus, ...]:
        """The stages up to and including `last_stage`; all of them when None or unknown."""
        names = [stage.stage for stage in self._stages]
        if last_stage not in names:
            return self._stages
        return self._stages[: names.index(last_stage) + 1]

    async def _until_finished(self, stage: StageStatus, request_id: str) -> dict[str, Any]:
        while True:
            try:
                record = await stage.get(request_id)
            except ServiceError as exc:
                logger.warning("waiting for %s job %s: %s", stage.stage, request_id, exc.message)
                record = None
            if record is not None and record.get("status") in (STATUS_DONE, STATUS_FAILED):
                return record
            await asyncio.sleep(self._poll_interval)


def _last_stage(sequence: Any) -> str | None:
    """The stage that ends a request, from the pipeline_name_sequence stored with its job; None (the whole
    pipeline) for a job from before it existed, or a sequence that is not a valid one."""
    if not isinstance(sequence, list) or not sequence:
        return None
    try:
        return STAGE_OF[validate_sequence(sequence)[-1]]
    except InvalidSequence:
        return None


def _ended(stage: str, record: dict[str, Any], results: dict[str, dict[str, Any]]) -> WaitOutcome | None:
    """The outcome when this finished job (DONE or FAILED) ends the pipeline, else None after keeping its
    result in `results`. Shared by `wait` and `snapshot`, so both read a job the same way."""
    if record["status"] == STATUS_FAILED:
        return WaitOutcome(stage, STATUS_FAILED, record.get("error_message"), results)
    results[stage] = record.get("result") or {}
    reject_reason = results[stage].get("reject_reason")
    if reject_reason:
        return WaitOutcome(stage, STATUS_REJECTED, reject_reason, results)
    return None
