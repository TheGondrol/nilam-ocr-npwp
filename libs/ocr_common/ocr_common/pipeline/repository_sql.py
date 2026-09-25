"""`JobRepository` on PostgreSQL (`<prefix>_jobs`, `<prefix>_results`), with the outcome row and the
outbox written in the job's transaction.
"""

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import MetaData, Table, and_, or_, select, update
from sqlalchemy.dialects import postgresql, sqlite
from sqlalchemy.ext.asyncio import AsyncEngine

from ocr_common.config import DEFAULT_JOB_LEASE_SECONDS
from ocr_common.pipeline import STATUS_DONE, STATUS_FAILED, STATUS_PROCESSING, JobRecord, StaleJob
from ocr_common.pipeline.database import get_engine
from ocr_common.pipeline.outbox import Outbox, OutboxMessage
from ocr_common.pipeline.outcomes import StageOutcome
from ocr_common.pipeline.repository import stored_sequence
from ocr_common.pipeline.tables import pipeline_tables


def build_tables(table_prefix: str) -> tuple[MetaData, Table, Table]:
    """The `jobs` and `results` tables of `table_prefix` on a fresh `MetaData`."""
    metadata = MetaData()
    jobs, results = pipeline_tables(table_prefix, metadata)
    return metadata, jobs, results


def _iso(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.isoformat()


class SqlJobRepository:
    """See `JobRepository`. Claims are atomic across replicas (`INSERT ... ON CONFLICT DO NOTHING`)."""

    name = "postgres"

    def __init__(
        self,
        database_url: str,
        table_prefix: str,
        *,
        lease_seconds: float = DEFAULT_JOB_LEASE_SECONDS,
        outcome: StageOutcome | None = None,
        outbox: Outbox | None = None,
        stage: str = "",
    ):
        self._url = database_url
        self._table_prefix = table_prefix
        self._lease = timedelta(seconds=lease_seconds)
        self._outcome = outcome
        self._outbox = outbox
        self._stage = stage or table_prefix.upper()
        self.metadata, self._jobs, self._results = build_tables(table_prefix)

    @property
    def engine(self) -> AsyncEngine:
        """The shared engine for this repository's database URL."""
        return get_engine(self._url)

    def _insert(self, table: Table):
        dialect = postgresql if get_engine(self._url).dialect.name == "postgresql" else sqlite
        return dialect.insert(table)

    async def claim(self, request_id: str, *, input: dict[str, Any] | None = None) -> bool:
        """See `JobRepository.claim`."""
        now = datetime.now(UTC)
        jobs = self._jobs
        async with self.engine.begin() as conn:
            inserted = await conn.execute(
                self._insert(jobs)
                .values(
                    request_id=request_id,
                    status=STATUS_PROCESSING,
                    attempts=1,
                    input=input,
                    created_at=now,
                    updated_at=now,
                    ds=now.strftime("%Y%m%d"),
                )
                .on_conflict_do_nothing(index_elements=["request_id"])
            )
            claimed = inserted.rowcount == 1
            if not claimed:
                retried = await conn.execute(
                    update(jobs)
                    .where(
                        jobs.c.request_id == request_id,
                        or_(
                            jobs.c.status == STATUS_FAILED,
                            and_(jobs.c.status == STATUS_PROCESSING, jobs.c.updated_at < now - self._lease),
                        ),
                    )
                    .values(
                        status=STATUS_PROCESSING,
                        error_message=None,
                        attempts=jobs.c.attempts + 1,
                        input=input,
                        updated_at=now,
                    )
                )
                claimed = retried.rowcount == 1
            if claimed and self._outcome is not None:
                await self._outcome.claimed(conn, request_id)
            return claimed

    async def reclaim_stale(self, limit: int) -> list[StaleJob]:
        """Jobs left PROCESSING past the lease: bump their attempts and updated_at in one transaction
        (FOR UPDATE SKIP LOCKED, so two replicas never take the same one) and return them."""
        now = datetime.now(UTC)
        jobs = self._jobs
        async with self.engine.begin() as conn:
            rows = (
                await conn.execute(
                    select(jobs.c.request_id, jobs.c.input)
                    .where(jobs.c.status == STATUS_PROCESSING, jobs.c.updated_at < now - self._lease)
                    .order_by(jobs.c.updated_at)
                    .limit(limit)
                    .with_for_update(skip_locked=True)
                )
            ).all()
            if not rows:
                return []
            ids = [row.request_id for row in rows]
            await conn.execute(
                update(jobs).where(jobs.c.request_id.in_(ids)).values(attempts=jobs.c.attempts + 1, updated_at=now)
            )
            if self._outcome is not None:
                for request_id in ids:
                    await self._outcome.claimed(conn, request_id)
        return [StaleJob(row.request_id, row.input) for row in rows]

    async def complete(
        self,
        request_id: str,
        result: dict[str, Any],
        *,
        outcome_data: dict[str, Any] | None = None,
        rejection: str | None = None,
        messages: Sequence[OutboxMessage] = (),
    ) -> None:
        """See `JobRepository.complete`."""
        now = datetime.now(UTC)
        jobs, results = self._jobs, self._results
        async with self.engine.begin() as conn:
            await conn.execute(
                self._insert(results)
                .values(request_id=request_id, result=result, created_at=now, updated_at=now, ds=now.strftime("%Y%m%d"))
                .on_conflict_do_update(index_elements=["request_id"], set_={"result": result, "updated_at": now})
            )
            await conn.execute(
                update(jobs).where(jobs.c.request_id == request_id).values(status=STATUS_DONE, updated_at=now)
            )
            if self._outcome is not None:
                if rejection:
                    await self._outcome.rejected(conn, request_id, rejection)
                else:
                    await self._outcome.completed(conn, request_id, outcome_data)
            if self._outbox is not None:
                await self._outbox.add(conn, request_id, self._stage, messages)
        self._wake(messages)

    async def fail(self, request_id: str, error_message: str, *, messages: Sequence[OutboxMessage] = ()) -> None:
        """See `JobRepository.fail`."""
        jobs = self._jobs
        async with self.engine.begin() as conn:
            await conn.execute(
                update(jobs)
                .where(jobs.c.request_id == request_id)
                .values(status=STATUS_FAILED, error_message=error_message, updated_at=datetime.now(UTC))
            )
            if self._outcome is not None:
                await self._outcome.failed(conn, request_id, error_message)
            if self._outbox is not None:
                await self._outbox.add(conn, request_id, self._stage, messages)
        self._wake(messages)

    async def handoff_failed(self, request_id: str, next_stage: str, error_message: str) -> None:
        """See `JobRepository.handoff_failed`."""
        if self._outcome is None:
            return
        async with self.engine.begin() as conn:
            await self._outcome.failed(conn, request_id, error_message, stage=next_stage)

    def _wake(self, messages: Sequence[OutboxMessage]) -> None:
        # After the commit: a relay woken inside the transaction would poll before the rows are visible.
        if self._outbox is not None and messages:
            self._outbox.wake()

    async def get(self, request_id: str) -> JobRecord | None:
        """See `JobRepository.get`."""
        jobs, results = self._jobs, self._results
        async with self.engine.connect() as conn:
            row = (
                await conn.execute(
                    select(
                        jobs.c.status,
                        jobs.c.error_message,
                        jobs.c.created_at,
                        jobs.c.updated_at,
                        jobs.c.input,
                        results.c.result,
                    )
                    .select_from(jobs.outerjoin(results, results.c.request_id == jobs.c.request_id))
                    .where(jobs.c.request_id == request_id)
                )
            ).one_or_none()
        if row is None:
            return None
        return {
            "request_id": request_id,
            "status": row.status,
            "result": row.result,
            "error_message": row.error_message,
            "created_at": _iso(row.created_at),
            "updated_at": _iso(row.updated_at),
            "pipeline_name_sequence": stored_sequence(row.input),
        }
