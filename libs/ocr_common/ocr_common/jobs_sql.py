from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import MetaData, Table, and_, or_, select, update
from sqlalchemy.dialects import postgresql, sqlite
from sqlalchemy.ext.asyncio import AsyncEngine

from ocr_common.config import DEFAULT_JOB_LEASE_SECONDS
from ocr_common.database import get_engine
from ocr_common.jobs import STATUS_DONE, STATUS_FAILED, STATUS_PROCESSING, JobRecord
from ocr_common.tables import pipeline_tables


def build_tables(table_prefix: str) -> tuple[MetaData, Table, Table]:
    metadata = MetaData()
    jobs, results = pipeline_tables(table_prefix, metadata)
    return metadata, jobs, results


def _iso(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.isoformat()


class SqlJobRepository:
    name = "postgres"

    def __init__(self, database_url: str, table_prefix: str, *, lease_seconds: float = DEFAULT_JOB_LEASE_SECONDS):
        self._url = database_url
        self._table_prefix = table_prefix
        self._lease = timedelta(seconds=lease_seconds)
        self.metadata, self._jobs, self._results = build_tables(table_prefix)

    @property
    def engine(self) -> AsyncEngine:
        return get_engine(self._url)

    def _insert(self, table: Table):
        dialect = postgresql if get_engine(self._url).dialect.name == "postgresql" else sqlite
        return dialect.insert(table)

    async def claim(self, request_id: str) -> bool:
        now = datetime.now(UTC)
        jobs = self._jobs
        async with self.engine.begin() as conn:
            inserted = await conn.execute(
                self._insert(jobs)
                .values(
                    request_id=request_id,
                    status=STATUS_PROCESSING,
                    attempts=1,
                    created_at=now,
                    updated_at=now,
                    ds=now.strftime("%Y%m%d"),
                )
                .on_conflict_do_nothing(index_elements=["request_id"])
            )
            if inserted.rowcount == 1:
                return True
            retried = await conn.execute(
                update(jobs)
                .where(
                    jobs.c.request_id == request_id,
                    or_(
                        jobs.c.status == STATUS_FAILED,
                        and_(jobs.c.status == STATUS_PROCESSING, jobs.c.updated_at < now - self._lease),
                    ),
                )
                .values(status=STATUS_PROCESSING, error_message=None, attempts=jobs.c.attempts + 1, updated_at=now)
            )
            return retried.rowcount == 1

    async def complete(self, request_id: str, result: dict[str, Any]) -> None:
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

    async def fail(self, request_id: str, error_message: str) -> None:
        jobs = self._jobs
        async with self.engine.begin() as conn:
            await conn.execute(
                update(jobs)
                .where(jobs.c.request_id == request_id)
                .values(status=STATUS_FAILED, error_message=error_message, updated_at=datetime.now(UTC))
            )

    async def get(self, request_id: str) -> JobRecord | None:
        jobs, results = self._jobs, self._results
        async with self.engine.connect() as conn:
            row = (
                await conn.execute(
                    select(jobs.c.status, jobs.c.error_message, jobs.c.created_at, jobs.c.updated_at, results.c.result)
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
        }
