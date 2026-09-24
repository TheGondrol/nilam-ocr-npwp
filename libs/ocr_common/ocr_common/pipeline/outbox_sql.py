"""The outbox table behind `ocr_common.pipeline.outbox`: the only part that needs SQLAlchemy, kept apart so
that a service without a database (guardrails) can import the relay and the message helpers."""

import asyncio
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import DateTime, MetaData, Row, case, func, select, update
from sqlalchemy.ext.asyncio import AsyncConnection

from ocr_common.pipeline.database import get_engine
from ocr_common.pipeline.outbox import OutboxMessage, OutboxStats
from ocr_common.pipeline.tables import outbox_table


def _aware(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


class SqlOutbox:
    """The outbox table. `add` runs inside the caller's transaction and does not wake the relay:
    call `wake()` after that transaction committed, or the relay polls before the rows are visible."""

    def __init__(self, database_url: str, table_prefix: str = ""):
        self._url = database_url
        self.table = outbox_table(MetaData(), table_prefix)
        self.pending = asyncio.Event()

    async def add(self, conn: AsyncConnection, request_id: str, stage: str, messages: Sequence[OutboxMessage]) -> None:
        """Insert `messages` for `request_id` and `stage` using the caller's connection."""
        if not messages:
            return
        now = datetime.now(UTC)
        await conn.execute(
            self.table.insert(),
            [
                {
                    "request_id": request_id,
                    "stage": stage,
                    "kind": message.kind,
                    "payload": message.payload,
                    "attempts": 0,
                    "next_attempt_at": now,
                    "created_at": now,
                    "updated_at": now,
                    "ds": now.strftime("%Y%m%d"),
                }
                for message in messages
            ],
        )

    def wake(self) -> None:
        """Set the event the relay of this process waits on."""
        self.pending.set()

    async def claim(self, stage: str, limit: int, lease_seconds: float) -> list[Row[Any]]:
        """Lease up to `limit` due messages of `stage` for `lease_seconds` (`FOR UPDATE SKIP LOCKED`),
        counting the attempt; returns their rows.
        """
        table = self.table
        now = datetime.now(UTC)
        async with get_engine(self._url).begin() as conn:
            due = (
                await conn.execute(
                    select(table.c.id)
                    .where(table.c.stage == stage, table.c.failed_at.is_(None), table.c.next_attempt_at <= now)
                    .order_by(table.c.id)
                    .limit(limit)
                    .with_for_update(skip_locked=True)
                )
            ).scalars()
            ids = list(due)
            if not ids:
                return []
            await conn.execute(
                update(table)
                .where(table.c.id.in_(ids))
                .values(
                    attempts=table.c.attempts + 1,
                    next_attempt_at=now + timedelta(seconds=lease_seconds),
                    updated_at=now,
                )
            )
            rows = await conn.execute(select(table).where(table.c.id.in_(ids)).order_by(table.c.id))
            return list(rows.all())

    async def done(self, message_id: int) -> None:
        """Delete a delivered message."""
        async with get_engine(self._url).begin() as conn:
            await conn.execute(self.table.delete().where(self.table.c.id == message_id))

    async def retry_later(self, message_id: int, delay_seconds: float, error: str) -> None:
        """Schedule a failed message again after `delay_seconds`, recording the error."""
        now = datetime.now(UTC)
        async with get_engine(self._url).begin() as conn:
            await conn.execute(
                update(self.table)
                .where(self.table.c.id == message_id)
                .values(next_attempt_at=now + timedelta(seconds=delay_seconds), last_error=error, updated_at=now)
            )

    async def give_up(self, row: Row[Any], error: str, replacement: OutboxMessage | None = None) -> None:
        """Keep the row as a dead letter (never claimed again, visible in `stats`) and, in the same
        transaction, queue the message that takes its place, if any."""
        now = datetime.now(UTC)
        async with get_engine(self._url).begin() as conn:
            await conn.execute(
                update(self.table)
                .where(self.table.c.id == row.id)
                .values(failed_at=now, last_error=error, updated_at=now)
            )
            if replacement is not None:
                await self.add(conn, row.request_id, row.stage, [replacement])

    async def release(self, stage: str, request_id: str | None = None) -> int:
        """Puts the dead letters of `stage` (of one request, or all of them) back in the queue, due now,
        and wakes the relay. Returns how many were released."""
        now = datetime.now(UTC)
        table = self.table
        statement = (
            update(table)
            .where(table.c.stage == stage, table.c.failed_at.is_not(None))
            .values(failed_at=None, next_attempt_at=now, updated_at=now)
        )
        if request_id is not None:
            statement = statement.where(table.c.request_id == request_id)
        async with get_engine(self._url).begin() as conn:
            released = (await conn.execute(statement)).rowcount
        if released:
            self.wake()
        return released

    async def stats(self, stage: str) -> OutboxStats:
        """Count the pending, retrying and dead messages of `stage` and the age of the oldest pending one."""
        table = self.table
        alive = table.c.failed_at.is_(None)
        async with get_engine(self._url).connect() as conn:
            row = (
                await conn.execute(
                    select(
                        func.count(case((alive, 1))).label("pending"),
                        func.count(case(((alive & (table.c.attempts > 0)), 1))).label("retrying"),
                        func.min(case((alive, table.c.created_at), else_=None), type_=DateTime(timezone=True)).label(
                            "oldest"
                        ),
                        func.count(table.c.failed_at).label("dead_letters"),
                    ).where(table.c.stage == stage)
                )
            ).one()
        oldest = None
        if row.oldest is not None:
            oldest = max((datetime.now(UTC) - _aware(row.oldest)).total_seconds(), 0.0)
        return OutboxStats(
            stage=stage,
            pending=row.pending,
            retrying=row.retrying,
            oldest_pending_seconds=oldest,
            dead_letters=row.dead_letters,
        )
