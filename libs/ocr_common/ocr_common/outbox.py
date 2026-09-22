import asyncio
import contextlib
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol

from sqlalchemy import MetaData, Row, select, update
from sqlalchemy.ext.asyncio import AsyncConnection

from ocr_common.database import get_engine
from ocr_common.errors import ServiceError
from ocr_common.tables import outbox_table

logger = logging.getLogger(__name__)

KIND_CALLBACK = "callback"
KIND_HANDOFF = "handoff"


@dataclass(frozen=True)
class OutboxMessage:
    kind: str
    payload: dict[str, Any]


def callback_message(
    request_id: str,
    stage: str,
    status: str,
    *,
    result: dict[str, Any] | None = None,
    error_message: str | None = None,
) -> OutboxMessage:
    return OutboxMessage(
        KIND_CALLBACK,
        {
            "request_id": request_id,
            "stage": stage,
            "status": status,
            "result": result,
            "error_message": error_message,
        },
    )


def handoff_message(next_stage: str, body: dict[str, Any]) -> OutboxMessage:
    return OutboxMessage(KIND_HANDOFF, {"next_stage": next_stage, "body": body})


class Outbox(Protocol):
    async def add(
        self, conn: AsyncConnection, request_id: str, stage: str, messages: Sequence[OutboxMessage]
    ) -> None: ...


class Sender(Protocol):
    async def send(self, body: dict[str, Any], /) -> None: ...


class SqlOutbox:
    def __init__(self, database_url: str):
        self._url = database_url
        self.table = outbox_table(MetaData())
        self.pending = asyncio.Event()

    async def add(self, conn: AsyncConnection, request_id: str, stage: str, messages: Sequence[OutboxMessage]) -> None:
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
        self.pending.set()

    async def claim(self, stage: str, limit: int, lease_seconds: float) -> list[Row[Any]]:
        table = self.table
        now = datetime.now(UTC)
        async with get_engine(self._url).begin() as conn:
            due = (
                await conn.execute(
                    select(table.c.id)
                    .where(table.c.stage == stage, table.c.next_attempt_at <= now)
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
        async with get_engine(self._url).begin() as conn:
            await conn.execute(self.table.delete().where(self.table.c.id == message_id))

    async def retry_later(self, message_id: int, delay_seconds: float) -> None:
        now = datetime.now(UTC)
        async with get_engine(self._url).begin() as conn:
            await conn.execute(
                update(self.table)
                .where(self.table.c.id == message_id)
                .values(next_attempt_at=now + timedelta(seconds=delay_seconds), updated_at=now)
            )

    async def replace_with(self, message_id: int, request_id: str, stage: str, message: OutboxMessage) -> None:
        async with get_engine(self._url).begin() as conn:
            await self.add(conn, request_id, stage, [message])
            await conn.execute(self.table.delete().where(self.table.c.id == message_id))


class OutboxRelay:
    def __init__(
        self,
        outbox: SqlOutbox,
        *,
        stage: str,
        callback: Sender,
        next_stage: Sender | None = None,
        interval_seconds: float = 1.0,
        batch: int = 20,
        lease_seconds: float = 30.0,
        retry_delay_seconds: float = 1.0,
        max_attempts: int = 20,
    ):
        self._outbox = outbox
        self._stage = stage
        self._callback = callback
        self._next_stage = next_stage
        self._interval = interval_seconds
        self._batch = batch
        self._lease = lease_seconds
        self._retry_delay = retry_delay_seconds
        self._max_attempts = max_attempts

    async def run(self) -> None:
        while True:
            try:
                delivered = await self.deliver_due()
            except Exception:
                logger.exception("outbox relay failed to deliver")
                delivered = 0
            if delivered:
                continue
            self._outbox.pending.clear()
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._outbox.pending.wait(), self._interval)

    async def deliver_due(self) -> int:
        rows = await self._outbox.claim(self._stage, self._batch, self._lease)
        delivered = 0
        for row in sorted(rows, key=lambda row: (row.kind != KIND_HANDOFF, row.id)):
            try:
                await self._send(row)
            except ServiceError as exc:
                if exc.status_code >= 500 and row.attempts < self._max_attempts:
                    await self._outbox.retry_later(row.id, self._backoff(row.attempts))
                    continue
                await self._give_up(row, exc.message)
                continue
            await self._outbox.done(row.id)
            delivered += 1
        return delivered

    def _backoff(self, attempts: int) -> float:
        return self._retry_delay * min(2 ** max(attempts - 1, 0), 32)

    async def _send(self, row: Row[Any]) -> None:
        if row.kind == KIND_CALLBACK:
            await self._callback.send(row.payload)
            return
        if self._next_stage is None:
            raise ServiceError(500, f"{row.stage} has no next stage to hand off to")
        await self._next_stage.send(row.payload["body"])

    async def _give_up(self, row: Row[Any], reason: str) -> None:
        logger.error("outbox gave up on %s %s of %s: %s", row.kind, row.stage, row.request_id, reason)
        if row.kind != KIND_HANDOFF:
            await self._outbox.done(row.id)
            return
        next_stage = row.payload["next_stage"]
        await self._outbox.replace_with(
            row.id,
            row.request_id,
            row.stage,
            callback_message(
                row.request_id,
                next_stage,
                "FAILED",
                error_message=f"Handoff to {next_stage} failed: {reason}",
            ),
        )
