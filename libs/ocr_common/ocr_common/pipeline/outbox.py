"""Transactional outbox: the messages a finished job must send (callback, hand-off) are written in the
same transaction as its result and delivered afterwards by `OutboxRelay`, so a crash between the two
loses nothing. The table itself lives in `outbox_sql` (SQLAlchemy); this module has no database
dependency so that a service without one can import the message helpers.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Protocol

from ocr_common.clients.remote import RemoteClientError
from ocr_common.errors import InternalError, ServiceError, UpstreamTimeout, UpstreamUnavailable
from ocr_common.overload import full_jitter, spread
from ocr_common.pipeline.callbacks import (
    NOT_READY_RETRIES,
    RESULT_NOT_READY,
    TRACE_PARENT_KEY,
    not_ready,
    not_ready_delay,
    stage_callback_body,
)
from ocr_common.web import apm
from ocr_common.web.logging import log_event
from ocr_common.web.request_id import bind_request_id, reset_request_id

if TYPE_CHECKING:
    from sqlalchemy import Row
    from sqlalchemy.ext.asyncio import AsyncConnection

    from ocr_common.pipeline.outbox_sql import SqlOutbox

logger = logging.getLogger(__name__)

KIND_CALLBACK = "callback"
KIND_HANDOFF = "handoff"

DEFAULT_MAX_BACKOFF_SECONDS = 300.0
DEFAULT_MAX_AGE_SECONDS = 24 * 3600.0
# The central orchestrator stops waiting for a callback 300 s after its 202 (to be raised to 600 s,
# 5 Oct 2026) and answers a later one with 409 RESULT_CONFLICT, so retrying a callback longer is useless.
DEFAULT_CALLBACK_MAX_AGE_SECONDS = 600.0
DEFAULT_STALE_AFTER_SECONDS = 300.0
STOP_DELIVERY_SECONDS = 5.0
WATCH_INTERVAL_SECONDS = 60.0
# `last_error` keeps one line per failed attempt, newest first, so a timeout is not lost under the 409 that
# followed it; the oldest lines go first when it grows past this.
LAST_ERROR_MAX_CHARS = 2000
# What the central orchestrator's 4xx codes to a callback mean (its contract, 2 and 5 Oct 2026), added to
# `last_error` so a dead letter says why without the contract at hand.
CENTRAL_CODE_HINTS = {
    "RESULT_CONFLICT": "the central orchestrator already closed this request, e.g. its callback deadline after the "
    "202 passed",
    "CALLBACK_NOT_EXPECTED": "the central orchestrator has this document type in poll mode and takes no callback",
}


@dataclass(frozen=True)
class OutboxMessage:
    """A message to deliver: `kind` is `callback` or `handoff`, `payload` is what to send."""

    kind: str
    payload: dict[str, Any]


@dataclass(frozen=True)
class OutboxStats:
    """One reading of a stage's backlog, as `GET /v1/<stage>/outbox` and the metrics report it."""

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
    error_code: str | None = None,
    final: bool = False,
    answer: dict[str, Any] | None = None,
    guardrails: int | float | None = None,
) -> OutboxMessage:
    """The callback body for `(stage, status)` of `request_id`, as a message."""
    return OutboxMessage(
        KIND_CALLBACK,
        _traced(
            stage_callback_body(
                request_id,
                stage,
                status,
                result=result,
                error_message=error_message,
                error_code=error_code,
                final=final,
                answer=answer,
                guardrails=guardrails,
            )
        ),
    )


def handoff_message(next_stage: str, body: dict[str, Any]) -> OutboxMessage:
    """The hand-off to `next_stage` with `body`, as a message."""
    return OutboxMessage(KIND_HANDOFF, _traced({"next_stage": next_stage, "body": body}))


def _traced(payload: dict[str, Any]) -> dict[str, Any]:
    """`payload` with the `traceparent` of the job writing it, while APM is on; the senders leave it out."""
    parent = apm.trace_parent()
    return {**payload, TRACE_PARENT_KEY: parent} if parent else payload


class Outbox(Protocol):
    """What the repository needs from an outbox: add messages inside its transaction, wake the relay after."""

    async def add(self, conn: AsyncConnection, request_id: str, stage: str, messages: Sequence[OutboxMessage]) -> None:
        """Insert `messages` for `request_id` using the caller's connection (inside its transaction)."""
        ...

    def wake(self) -> None:
        """Tell the relay of this process that new rows are committed."""
        ...


class Sender(Protocol):
    """Anything that can send one message body: the orchestrator callback or the next-stage client."""

    async def send(self, body: dict[str, Any], /) -> bool | None:
        """Send `body` once; False when there was nothing to send. Raise `ServiceError` on failure."""
        ...


HandoffFailed = Callable[[str, str, str], Awaitable[None]]


class OutboxRelay:
    """Delivers one stage's messages. A message is retried with exponential back-off while the receiver
    answers 5xx or cannot be reached, until it is older than `max_age_seconds` (a callback:
    `callback_max_age_seconds`); a 4xx, or that age, turns it into a dead letter that stays in the
    table. The one 4xx retried is the central orchestrator's 409 RESULT_NOT_READY to a callback, every
    1-2 s (`not_ready_delay`), `NOT_READY_RETRIES` times. Retry waits are jittered (`_backoff`). A hand-off that
    becomes a dead letter is
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
        callback_max_age_seconds: float = DEFAULT_CALLBACK_MAX_AGE_SECONDS,
        stale_after_seconds: float = DEFAULT_STALE_AFTER_SECONDS,
        watch_interval_seconds: float = WATCH_INTERVAL_SECONDS,
        metrics_stage: str | None = None,
    ):
        self._outbox = outbox
        self._stage = stage
        self._metrics_stage = metrics_stage or stage
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
        self._callback_max_age = callback_max_age_seconds
        self._stale_after = stale_after_seconds
        self._watch_interval = watch_interval_seconds
        self._next_watch = 0.0
        self._task: asyncio.Task[None] | None = None
        self._stopping = asyncio.Event()

    @property
    def stage(self) -> str:
        """The stage whose messages this relay delivers."""
        return self._stage

    def start(self) -> None:
        """Start the delivery loop as a background task (idempotent)."""
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
        """The delivery loop: deliver what is due, then sleep until woken, stopped, or the poll interval passes."""
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
            # +/- 20%, so the relays of replicas started together do not poll the table in step.
            await asyncio.wait(waiters, timeout=spread(self._interval), return_when=asyncio.FIRST_COMPLETED)
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
        from ocr_common.pipeline import metrics

        metrics.observe_outbox(stats, self._metrics_stage)
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
        """Claim and send one batch of due messages; returns how many were delivered."""
        from ocr_common.pipeline import metrics

        rows = await self._outbox.claim(self._stage, self._batch, self._lease)
        delivered = 0
        for row in sorted(rows, key=lambda row: (row.kind != KIND_HANDOFF, row.id)):
            result = await self._deliver(row)
            # Nothing to send is still a delivery for the metrics: the row is done.
            outcome = apm.DELIVERED if result == apm.SKIPPED else result
            metrics.OUTBOX_DELIVERIES.labels(self._metrics_stage, row.kind, outcome).inc()
            if outcome == apm.DELIVERED:
                delivered += 1
        return delivered

    async def _deliver(self, row: Row[Any]) -> str:
        """Send one message and record what happened to it: `delivered`, `skipped`, `retry` or `dead`. One APM
        transaction per delivery, in the trace of the job that queued it."""
        token = bind_request_id(row.request_id)
        target = row.payload.get("next_stage") if row.kind == KIND_HANDOFF else None
        try:
            with apm.delivery_transaction(
                row.stage,
                row.kind,
                row.request_id,
                row.payload.get(TRACE_PARENT_KEY),
                attempt=row.attempts,
                target=target,
            ):
                try:
                    sent = await self._send(row)
                except ServiceError as exc:
                    delay = self._retry_in(row, exc)
                    detail = self._failure(row, exc, target, delay)
                    last_error = _prepend(detail, row.last_error)
                    if delay is not None:
                        await self._outbox.retry_later(row.id, delay, last_error)
                        result = apm.RETRY
                    else:
                        result = apm.DEAD
                    self._log_delivery(row, result, target, error=exc, retry_in=delay, detail=detail)
                    if result == apm.DEAD:
                        await self._give_up(row, exc.message, last_error)
                else:
                    await self._outbox.done(row.id)
                    result = apm.DELIVERED if sent else apm.SKIPPED
                    self._log_delivery(row, result, target)
                apm.delivery_result(result)
                return result
        finally:
            reset_request_id(token)

    def _log_delivery(
        self,
        row: Row[Any],
        result: str,
        target: str | None,
        *,
        error: ServiceError | None = None,
        retry_in: float | None = None,
        detail: str | None = None,
    ) -> None:
        """One log event per delivery (`event.dataset: outbox`, searchable in Elasticsearch): what was sent where,
        which attempt, and what came of it. Never the payload: it carries the document's fields."""
        to = target or "orchestration"
        message = f"outbox {row.kind} {row.stage} -> {to}: {result} (attempt {row.attempts})"
        if error is not None:
            message += f": {error.message}"
        outbox: dict[str, Any] = {
            "id": row.id,
            "kind": row.kind,
            "stage": row.stage,
            "target": to,
            "attempt": row.attempts,
            "age_seconds": round(self._age(row), 1),
        }
        if error is not None:
            outbox["error"] = error.message
            outbox["status_code"] = error.status_code
            if isinstance(error, RemoteClientError) and error.remote_code:
                outbox["remote_code"] = error.remote_code
        if detail is not None:
            outbox["detail"] = detail
        if retry_in is not None:
            outbox["retry_in_seconds"] = round(retry_in, 1)
        failed = result in (apm.RETRY, apm.DEAD)
        level = logging.ERROR if result == apm.DEAD else logging.WARNING if failed else logging.INFO
        log_event(
            logger,
            level,
            message,
            event={"dataset": "outbox", "action": result, "outcome": "failure" if failed else "success"},
            outbox=outbox,
        )

    def _age(self, row: Row[Any]) -> float:
        created = row.created_at if row.created_at.tzinfo is not None else row.created_at.replace(tzinfo=UTC)
        return (datetime.now(UTC) - created).total_seconds()

    def _retry_in(self, row: Row[Any], exc: ServiceError) -> float | None:
        """Seconds until the next try of a message that failed with `exc`, or None to give up on it.
        `row.attempts` counts the sends so far, this one included."""
        if exc.status_code >= 500:
            max_age = self._callback_max_age if row.kind == KIND_CALLBACK else self._max_age
            return self._backoff(row.attempts) if self._age(row) < max_age else None
        if row.kind == KIND_CALLBACK and not_ready(exc) and row.attempts <= NOT_READY_RETRIES:
            return not_ready_delay()
        return None

    def _failure(self, row: Row[Any], exc: ServiceError, target: str | None, retry_in: float | None) -> str:
        """The `last_error` line of this failed attempt: when, what the receiver did (its status and error code,
        a timeout, or no connection), and what the relay does about it: retry, or give up and why."""
        age = self._age(row)
        outcome = (
            f"retry in {round(retry_in, 1):g}s"
            if retry_in is not None
            else f"dead letter: {self._why_dead(row, exc, age)}"
        )
        when = f"attempt {row.attempts} at {datetime.now(UTC):%H:%M:%S}Z, {age:.0f}s after it was queued"
        return f"{when}: {_describe(exc, target or 'orchestration')} -> {outcome}"

    def _why_dead(self, row: Row[Any], exc: ServiceError, age: float) -> str:
        """Why `_retry_in` gave up on `row` (it returned None)."""
        if exc.status_code >= 500:
            max_age = self._callback_max_age if row.kind == KIND_CALLBACK else self._max_age
            return f"still failing {age:.0f}s after it was queued, past the {max_age:.0f}s limit"
        if row.kind == KIND_CALLBACK and not_ready(exc):
            return f"still {RESULT_NOT_READY} after {NOT_READY_RETRIES} retries"
        return f"a {exc.status_code} is final, it is not sent again"

    def _backoff_cap(self, attempts: int) -> float:
        """The longest wait before try `attempts + 1`: the retry delay doubled per try, up to the maximum."""
        return min(self._retry_delay * 2 ** max(attempts - 1, 0), self._max_backoff)

    def _backoff(self, attempts: int) -> float:
        """A random wait up to `_backoff_cap` ("full jitter"): after an outage of the receiver, the messages that
        failed together are spread over the window instead of all being sent again in the same second."""
        return full_jitter(self._backoff_cap(attempts))

    async def _send(self, row: Row[Any]) -> bool:
        """Send the message; False when there was nothing to send."""
        if row.kind == KIND_CALLBACK:
            return await self._callback.send(row.payload) is not False
        if self._next_stage is None:
            raise InternalError(f"{row.stage} has no next stage to hand off to")
        await self._next_stage.send(row.payload["body"])
        return True

    async def _give_up(self, row: Row[Any], reason: str, last_error: str) -> None:
        """Turn the message into a dead letter (logged as a `dead` delivery by `_deliver`), keeping `last_error`.
        A dead hand-off is replaced by a FAILED callback whose message is the receiver's `reason` only: that one
        goes to the central orchestrator."""
        replacement = None
        if row.kind == KIND_HANDOFF:
            next_stage = row.payload["next_stage"]
            message = f"Handoff to {next_stage} failed: {reason}"
            if self._handoff_failed is not None:
                await self._handoff_failed(row.request_id, next_stage, message)
            if self._callbacks:
                replacement = callback_message(row.request_id, next_stage, "FAILED", error_message=message)
        await self._outbox.give_up(row, last_error, replacement)


def _describe(exc: ServiceError, target: str) -> str:
    """What `target` did with the message, from the error its client raised: answered with a status (and its
    own error code), did not answer in time, or could not be reached."""
    if isinstance(exc, RemoteClientError):
        code = f" {exc.remote_code}" if exc.remote_code else ""
        text = f"{target} answered {exc.status_code}{code}: {exc.message}"
        hint = CENTRAL_CODE_HINTS.get(exc.remote_code or "") if target == "orchestration" else None
        return f"{text} ({hint})" if hint else text
    if isinstance(exc, UpstreamTimeout):
        return f"{exc.message}, no answer in time ({target} may still have received it)"
    if isinstance(exc, UpstreamUnavailable):
        cause = exc.__cause__
        if cause is None:
            return exc.message
        return f"{exc.message} ({type(cause).__name__}{f': {cause}' if str(cause) else ''})"
    if isinstance(exc, InternalError):
        return exc.message
    return f"{exc.message} ({exc.status_code})"


def _prepend(line: str, previous: str | None) -> str:
    """`line` above the earlier attempts' lines, the oldest dropped past `LAST_ERROR_MAX_CHARS`."""
    text = f"{line}\n{previous}" if previous else line
    if len(text) <= LAST_ERROR_MAX_CHARS:
        return text
    cut = text.rfind("\n", 0, LAST_ERROR_MAX_CHARS)
    return text[:cut] if cut > 0 else text[:LAST_ERROR_MAX_CHARS]
