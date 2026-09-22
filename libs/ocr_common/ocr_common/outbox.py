from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Protocol

from ocr_common.errors import ServiceError

if TYPE_CHECKING:
    from sqlalchemy import Row
    from sqlalchemy.ext.asyncio import AsyncConnection

    from ocr_common.outbox_sql import SqlOutbox

logger = logging.getLogger(__name__)

KIND_CALLBACK = "callback"
KIND_HANDOFF = "handoff"

DEFAULT_MAX_BACKOFF_SECONDS = 300.0
DEFAULT_MAX_AGE_SECONDS = 24 * 3600.0
DEFAULT_STALE_AFTER_SECONDS = 300.0
STOP_DELIVERY_SECONDS = 5.0
WATCH_INTERVAL_SECONDS = 60.0


@dataclass(frozen=True)
class OutboxMessage:
    kind: str
    payload: dict[str, Any]


@dataclass(frozen=True)
class OutboxStats:
    stage: str
    pending: int
    retrying: int
    oldest_pending_seconds: float | None
    dead_letters: int


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

    def wake(self) -> None: ...


class Sender(Protocol):
    async def send(self, body: dict[str, Any], /) -> None: ...


HandoffFailed = Callable[[str, str, str], Awaitable[None]]


class OutboxRelay:
    """Delivers one stage's messages. A message is retried with exponential back-off while the receiver
    answers 5xx or cannot be reached, until it is older than `max_age_seconds`; a 4xx, or that age,
    turns it into a dead letter that stays in the table. A hand-off that becomes a dead letter is
    replaced by a FAILED callback on behalf of the next stage, like in direct mode."""

    def __init__(
        self,
        outbox: SqlOutbox,
        *,
        stage: str,
        callback: Sender,
        next_stage: Sender | None = None,
        callbacks: bool = True,
        handoff_failed: HandoffFailed | None = None,
        interval_seconds: float = 1.0,
        batch: int = 20,
        lease_seconds: float = 30.0,
        retry_delay_seconds: float = 1.0,
        max_backoff_seconds: float = DEFAULT_MAX_BACKOFF_SECONDS,
        max_age_seconds: float = DEFAULT_MAX_AGE_SECONDS,
        stale_after_seconds: float = DEFAULT_STALE_AFTER_SECONDS,
        watch_interval_seconds: float = WATCH_INTERVAL_SECONDS,
    ):
        self._outbox = outbox
        self._stage = stage
        self._callback = callback
        self._next_stage = next_stage
        self._callbacks = callbacks
        self._handoff_failed = handoff_failed
        self._interval = interval_seconds
        self._batch = batch
        self._lease = lease_seconds
        self._retry_delay = retry_delay_seconds
        self._max_backoff = max_backoff_seconds
        self._max_age = max_age_seconds
        self._stale_after = stale_after_seconds
        self._watch_interval = watch_interval_seconds
        self._next_watch = 0.0
        self._task: asyncio.Task[None] | None = None
        self._stopping = asyncio.Event()

    @property
    def stage(self) -> str:
        return self._stage

    def start(self) -> None:
        if self._task is None:
            self._stopping.clear()
            self._task = asyncio.create_task(self.run(), name=f"outbox-relay-{self._stage}")

    async def stop(self, delivery_timeout: float = STOP_DELIVERY_SECONDS) -> None:
        """Let the loop finish the delivery it is in (cancelling mid-query would leave the connection
        in an unknown state), then make one bounded attempt at what is still due, so the messages of
        jobs that finished during the drain do not wait for the next start. What does not make it stays
        leased and is picked up by another replica or after the restart."""
        self._stopping.set()
        if self._task is not None:
            try:
                await asyncio.wait_for(self._task, delivery_timeout)
            except TimeoutError:
                logger.warning("outbox relay %s: did not stop in %.0fs, cancelled", self._stage, delivery_timeout)
            self._task = None
        try:
            await asyncio.wait_for(self.deliver_due(), delivery_timeout)
        except TimeoutError:
            logger.warning("outbox relay %s: shutdown delivery did not finish in %.0fs", self._stage, delivery_timeout)
        except Exception:
            logger.exception("outbox relay %s: shutdown delivery failed", self._stage)

    async def run(self) -> None:
        while not self._stopping.is_set():
            try:
                delivered = await self.deliver_due()
            except Exception:
                logger.exception("outbox relay %s failed to deliver", self._stage)
                delivered = 0
            if delivered:
                continue
            await self.watch()
            self._outbox.pending.clear()
            await self._idle()

    async def _idle(self) -> None:
        """Sleep until a message is queued in this process, a stop is requested, or the poll interval
        (for messages queued by other replicas) elapses."""
        waiters = [asyncio.ensure_future(self._outbox.pending.wait()), asyncio.ensure_future(self._stopping.wait())]
        try:
            await asyncio.wait(waiters, timeout=self._interval, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for waiter in waiters:
                waiter.cancel()

    async def watch(self) -> OutboxStats | None:
        """Once per `watch_interval_seconds` while idle: log a warning when the backlog is stale or dead
        letters are waiting, so that an alert can be raised on the log without another service."""
        now = time.monotonic()
        if now < self._next_watch:
            return None
        self._next_watch = now + self._watch_interval
        try:
            stats = await self._outbox.stats(self._stage)
        except Exception:
            logger.exception("outbox relay %s: could not read the backlog", self._stage)
            return None
        stale = stats.oldest_pending_seconds is not None and stats.oldest_pending_seconds > self._stale_after
        if stale or stats.dead_letters:
            logger.warning(
                "outbox %s backlog: %d pending (%d retrying, oldest %.0fs), %d dead letters",
                self._stage,
                stats.pending,
                stats.retrying,
                stats.oldest_pending_seconds or 0.0,
                stats.dead_letters,
            )
        return stats

    async def deliver_due(self) -> int:
        rows = await self._outbox.claim(self._stage, self._batch, self._lease)
        delivered = 0
        for row in sorted(rows, key=lambda row: (row.kind != KIND_HANDOFF, row.id)):
            try:
                await self._send(row)
            except ServiceError as exc:
                if exc.status_code >= 500 and self._age(row) < self._max_age:
                    await self._outbox.retry_later(row.id, self._backoff(row.attempts), exc.message)
                    continue
                await self._give_up(row, exc.message)
                continue
            await self._outbox.done(row.id)
            delivered += 1
        return delivered

    def _age(self, row: Row[Any]) -> float:
        created = row.created_at if row.created_at.tzinfo is not None else row.created_at.replace(tzinfo=UTC)
        return (datetime.now(UTC) - created).total_seconds()

    def _backoff(self, attempts: int) -> float:
        return min(self._retry_delay * 2 ** max(attempts - 1, 0), self._max_backoff)

    async def _send(self, row: Row[Any]) -> None:
        if row.kind == KIND_CALLBACK:
            await self._callback.send(row.payload)
            return
        if self._next_stage is None:
            raise ServiceError(500, f"{row.stage} has no next stage to hand off to")
        await self._next_stage.send(row.payload["body"])

    async def _give_up(self, row: Row[Any], reason: str) -> None:
        logger.error(
            "outbox gave up on %s %s of %s after %d attempts: %s",
            row.kind,
            row.stage,
            row.request_id,
            row.attempts,
            reason,
        )
        replacement = None
        if row.kind == KIND_HANDOFF:
            next_stage = row.payload["next_stage"]
            message = f"Handoff to {next_stage} failed: {reason}"
            if self._handoff_failed is not None:
                await self._handoff_failed(row.request_id, next_stage, message)
            if self._callbacks:
                replacement = callback_message(row.request_id, next_stage, "FAILED", error_message=message)
        await self._outbox.give_up(row, reason, replacement)
