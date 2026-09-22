from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol, TypedDict

from ocr_common.config import DEFAULT_JOB_LEASE_SECONDS
from ocr_common.pipeline.outbox import Outbox, OutboxMessage
from ocr_common.pipeline.outcomes import StageOutcome

STATUS_PROCESSING = "PROCESSING"
STATUS_DONE = "DONE"
STATUS_FAILED = "FAILED"


class JobRecord(TypedDict):
    request_id: str
    status: str
    result: dict[str, Any] | None
    error_message: str | None
    created_at: str
    updated_at: str


class JobRepository(Protocol):
    name: str

    async def claim(self, request_id: str) -> bool: ...

    async def complete(
        self,
        request_id: str,
        result: dict[str, Any],
        *,
        outcome_data: dict[str, Any] | None = None,
        messages: Sequence[OutboxMessage] = (),
    ) -> None: ...

    async def fail(self, request_id: str, error_message: str, *, messages: Sequence[OutboxMessage] = ()) -> None: ...

    async def handoff_failed(self, request_id: str, next_stage: str, error_message: str) -> None: ...

    async def get(self, request_id: str) -> JobRecord | None: ...


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


class InMemoryJobRepository:
    name = "memory"

    def __init__(self, lease_seconds: float = DEFAULT_JOB_LEASE_SECONDS) -> None:
        self._jobs: dict[str, JobRecord] = {}
        self._lease = timedelta(seconds=lease_seconds)

    async def claim(self, request_id: str) -> bool:
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
            return True
        expired = (
            record["status"] == STATUS_PROCESSING
            and datetime.fromisoformat(record["updated_at"]) < current - self._lease
        )
        if record["status"] == STATUS_FAILED or expired:
            record.update(status=STATUS_PROCESSING, error_message=None, updated_at=now)
            return True
        return False

    async def complete(
        self,
        request_id: str,
        result: dict[str, Any],
        *,
        outcome_data: dict[str, Any] | None = None,
        messages: Sequence[OutboxMessage] = (),
    ) -> None:
        self._jobs[request_id].update(status=STATUS_DONE, result=result, updated_at=_now_iso())

    async def fail(self, request_id: str, error_message: str, *, messages: Sequence[OutboxMessage] = ()) -> None:
        self._jobs[request_id].update(status=STATUS_FAILED, error_message=error_message, updated_at=_now_iso())

    async def handoff_failed(self, request_id: str, next_stage: str, error_message: str) -> None:
        pass

    async def get(self, request_id: str) -> JobRecord | None:
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
    if not database_url:
        return InMemoryJobRepository(lease_seconds)
    from ocr_common.pipeline.repository_sql import SqlJobRepository

    return SqlJobRepository(
        database_url, table_prefix, lease_seconds=lease_seconds, outcome=outcome, outbox=outbox, stage=stage
    )
