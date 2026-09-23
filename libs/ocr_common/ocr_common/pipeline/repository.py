"""Job status storage of a stage: the `JobRepository` protocol, the in-memory implementation for a
single process without a database, and (in `repository_sql`) the PostgreSQL one.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol, TypedDict

from ocr_common.config import DEFAULT_JOB_LEASE_SECONDS
from ocr_common.pipeline.outbox import Outbox, OutboxMessage
from ocr_common.pipeline.outcomes import StageOutcome

STATUS_PROCESSING = "PROCESSING"
STATUS_DONE = "DONE"
STATUS_FAILED = "FAILED"


class JobRecord(TypedDict):
    """A job as `GET /v1/<stage>/jobs/{request_id}` returns it."""

    request_id: str
    status: str
    result: dict[str, Any] | None
    error_message: str | None
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class StaleJob:
    """A job left PROCESSING past its lease, reclaimed for another run, with the input stored at claim."""

    request_id: str
    input: dict[str, Any] | None


class JobRepository(Protocol):
    """What `StagePipeline` needs from job storage. Every method is safe to call from several processes."""

    name: str

    async def claim(self, request_id: str, *, input: dict[str, Any] | None = None) -> bool:
        """Atomically take `request_id` for a run: True for a new job, a `FAILED` one, or one whose lease
        expired; False when it is `DONE` or still `PROCESSING` within its lease.
        """
        ...

    async def reclaim_stale(self, limit: int) -> list[StaleJob]:
        """Take up to `limit` `PROCESSING` jobs whose lease expired, for the reaper."""
        ...

    async def complete(
        self,
        request_id: str,
        result: dict[str, Any],
        *,
        outcome_data: dict[str, Any] | None = None,
        messages: Sequence[OutboxMessage] = (),
    ) -> None:
        """Store `result`, mark `DONE`, and in the same transaction write the outcome row and the outbox messages."""
        ...

    async def fail(self, request_id: str, error_message: str, *, messages: Sequence[OutboxMessage] = ()) -> None:
        """Mark `FAILED` with `error_message`; in the same transaction write the outcome row and the outbox
        messages."""
        ...

    async def handoff_failed(self, request_id: str, next_stage: str, error_message: str) -> None:
        """Record in the outcome row that `next_stage` never received the job."""
        ...

    async def get(self, request_id: str) -> JobRecord | None:
        """The job's record, or None when the request_id is unknown to this stage."""
        ...


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


class InMemoryJobRepository:
    """Jobs in a dict: for local runs without `DATABASE_URL` and for tests. Not shared between processes."""

    name = "memory"

    def __init__(self, lease_seconds: float = DEFAULT_JOB_LEASE_SECONDS) -> None:
        self._jobs: dict[str, JobRecord] = {}
        self._inputs: dict[str, dict[str, Any] | None] = {}
        self._lease = timedelta(seconds=lease_seconds)

    async def claim(self, request_id: str, *, input: dict[str, Any] | None = None) -> bool:
        """See `JobRepository.claim`."""
        current = datetime.now(UTC)
        now = current.isoformat()
        record = self._jobs.get(request_id)
        if record is None:
            self._jobs[request_id] = {
                "request_id": request_id,
                "status": STATUS_PROCESSING,
                "result": None,
                "error_message": None,
                "created_at": now,
                "updated_at": now,
            }
            self._inputs[request_id] = input
            return True
        expired = (
            record["status"] == STATUS_PROCESSING
            and datetime.fromisoformat(record["updated_at"]) < current - self._lease
        )
        if record["status"] == STATUS_FAILED or expired:
            record.update(status=STATUS_PROCESSING, error_message=None, updated_at=now)
            self._inputs[request_id] = input
            return True
        return False

    async def reclaim_stale(self, limit: int) -> list[StaleJob]:
        """See `JobRepository.reclaim_stale`."""
        current = datetime.now(UTC)
        stale: list[StaleJob] = []
        for request_id, record in self._jobs.items():
            expired = (
                record["status"] == STATUS_PROCESSING
                and datetime.fromisoformat(record["updated_at"]) < current - self._lease
            )
            if not expired:
                continue
            record["updated_at"] = current.isoformat()
            stale.append(StaleJob(request_id, self._inputs.get(request_id)))
            if len(stale) >= limit:
                break
        return stale

    async def complete(
        self,
        request_id: str,
        result: dict[str, Any],
        *,
        outcome_data: dict[str, Any] | None = None,
        messages: Sequence[OutboxMessage] = (),
    ) -> None:
        """See `JobRepository.complete`; the outcome row and outbox do not exist in memory."""
        self._jobs[request_id].update(status=STATUS_DONE, result=result, updated_at=_now_iso())

    async def fail(self, request_id: str, error_message: str, *, messages: Sequence[OutboxMessage] = ()) -> None:
        """See `JobRepository.fail`."""
        self._jobs[request_id].update(status=STATUS_FAILED, error_message=error_message, updated_at=_now_iso())

    async def handoff_failed(self, request_id: str, next_stage: str, error_message: str) -> None:
        """No outcome row in memory: nothing to record."""
        pass

    async def get(self, request_id: str) -> JobRecord | None:
        """See `JobRepository.get`."""
        record = self._jobs.get(request_id)
        return record.copy() if record else None


def build_job_repository(
    database_url: str | None,
    table_prefix: str,
    *,
    lease_seconds: float = DEFAULT_JOB_LEASE_SECONDS,
    outcome: StageOutcome | None = None,
    outbox: Outbox | None = None,
    stage: str = "",
) -> JobRepository:
    """The SQL repository when `database_url` is set, else the in-memory one."""
    if not database_url:
        return InMemoryJobRepository(lease_seconds)
    from ocr_common.pipeline.repository_sql import SqlJobRepository

    return SqlJobRepository(
        database_url, table_prefix, lease_seconds=lease_seconds, outcome=outcome, outbox=outbox, stage=stage
    )
