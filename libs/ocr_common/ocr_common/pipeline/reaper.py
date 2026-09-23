"""Runs again the jobs a dead process left behind.

A job runs as an asyncio task of the process that claimed it. When that process dies without
cleaning up (OOM, SIGKILL, node lost), its `<stage>_jobs` rows stay `PROCESSING` with nobody working
on them. This reaper, one per process like the outbox relay, claims rows whose `updated_at` is older
than the job lease and hands them to the service's `resume(request_id, input)`, which rebuilds the
work from what is in the database (the stored `input` and the earlier stages' results)."""

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from ocr_common.pipeline.repository import JobRepository

logger = logging.getLogger(__name__)

STOP_TIMEOUT_SECONDS = 5.0

Resume = Callable[[str, dict[str, Any] | None], Awaitable[None]]


class StaleJobReaper:
    """Per-process task that runs again the jobs of this stage a dead process left `PROCESSING`."""

    def __init__(
        self,
        repository: JobRepository,
        resume: Resume,
        *,
        stage: str,
        interval_seconds: float = 30.0,
        batch: int = 10,
    ):
        self._repository = repository
        self._resume = resume
        self._stage = stage
        self._interval = interval_seconds
        self._batch = batch
        self._task: asyncio.Task[None] | None = None
        self._stopping = asyncio.Event()

    def start(self) -> None:
        """Start the loop as a background task (idempotent)."""
        if self._task is None:
            self._stopping.clear()
            self._task = asyncio.create_task(self.run(), name=f"stale-jobs-{self._stage}")

    async def stop(self, timeout: float = STOP_TIMEOUT_SECONDS) -> None:
        """Stop looking for more work; jobs already handed to `resume` finish with the pipeline drain."""
        self._stopping.set()
        if self._task is not None:
            try:
                await asyncio.wait_for(self._task, timeout)
            except TimeoutError:
                logger.warning("stale job reaper %s: did not stop in %.0fs, cancelled", self._stage, timeout)
            self._task = None

    async def run(self) -> None:
        """The loop: reclaim a batch, hand each job to `resume`, sleep `interval_seconds` when nothing was found."""
        while not self._stopping.is_set():
            try:
                reclaimed = await self.run_once()
            except Exception:
                logger.exception("stale job reaper %s failed", self._stage)
                reclaimed = 0
            if reclaimed:
                continue
            try:
                await asyncio.wait_for(self._stopping.wait(), self._interval)
            except TimeoutError:
                pass

    async def run_once(self) -> int:
        """Claim up to `batch` stale jobs and start them again; returns how many were started."""
        stale = await self._repository.reclaim_stale(self._batch)
        if stale:
            from ocr_common.pipeline import metrics

            metrics.STALE_JOBS_RECLAIMED.labels(self._stage).inc(len(stale))
        for job in stale:
            logger.warning(
                "%s job %s was left PROCESSING past its lease (process died?); running it again",
                self._stage,
                job.request_id,
            )
            await self._resume(job.request_id, job.input)
        return len(stale)
