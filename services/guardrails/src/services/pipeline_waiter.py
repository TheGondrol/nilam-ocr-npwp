import asyncio
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from ocr_common.errors import ServiceError
from ocr_common.jobs import STATUS_DONE, STATUS_FAILED, STATUS_PROCESSING

logger = logging.getLogger(__name__)


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
    async def wait(self, request_id: str, timeout: float) -> WaitOutcome: ...


class PipelineWaiter:
    def __init__(self, stages: Sequence[StageStatus], *, poll_interval: float):
        self._stages = tuple(stages)
        self._poll_interval = poll_interval

    async def wait(self, request_id: str, timeout: float) -> WaitOutcome:
        results: dict[str, dict[str, Any]] = {}
        current = self._stages[0].stage
        if timeout <= 0:
            return WaitOutcome(current, STATUS_PROCESSING, results=results)
        try:
            async with asyncio.timeout(timeout):
                for stage in self._stages:
                    current = stage.stage
                    record = await self._until_finished(stage, request_id)
                    if record["status"] == STATUS_FAILED:
                        return WaitOutcome(current, STATUS_FAILED, record.get("error_message"), results)
                    results[current] = record.get("result") or {}
        except TimeoutError:
            return WaitOutcome(current, STATUS_PROCESSING, results=results)
        return WaitOutcome(current, STATUS_DONE, results=results)

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
