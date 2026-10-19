"""Protection against overload, shared by the services: jittered retry delays, bulkheads (a bounded number of jobs
running plus a bounded queue, the rest refused at once), and a circuit breaker in front of a remote model.

Why: a stage used to start every job it accepted right away, so a burst (a stress test at 3 rps of PDFs, 15 Oct
2026) put hundreds of calls on the OCR model VM at once, every one of them timed out, and the extraction pod held
every upload in memory while waiting. With a bulkhead a burst waits in a short queue, and beyond it the caller
gets 503 `SERVICE_BUSY` with `Retry-After` before anything is started; with the breaker a model that keeps failing
is given room to recover instead of more calls; with jitter the retries after an outage do not all land in the
same second.

Every limit is off by default (0): the settings of each service turn them on (deploy/helm values)."""

import asyncio
import logging
import math
import random
import time
from collections.abc import Callable
from types import TracebackType
from typing import Self

from prometheus_client import Counter, Gauge

from ocr_common.errors import ServiceBusy, ServiceError, UpstreamUnavailable

logger = logging.getLogger(__name__)

BULKHEAD_RUNNING = Gauge("bulkhead_running", "Callers inside a bulkhead now", ["bulkhead"])
BULKHEAD_WAITING = Gauge("bulkhead_waiting", "Callers waiting for a slot of a bulkhead", ["bulkhead"])
BULKHEAD_REJECTED = Counter("bulkhead_rejected_total", "Callers refused because a bulkhead was full", ["bulkhead"])
CIRCUIT_OPEN = Gauge("circuit_open", "1 while a circuit breaker refuses calls (open or probing), else 0", ["circuit"])
CIRCUIT_OPENED = Counter("circuit_opened_total", "Times a circuit breaker opened", ["circuit"])

_rng = random.Random()


# --- jitter ---------------------------------------------------------------------------------------------------


def full_jitter(delay: float, rng: random.Random | None = None) -> float:
    """A random wait between 0 and `delay` ("full jitter"): callers that failed together retry spread over the
    whole window instead of all at once."""
    return (rng or _rng).uniform(0.0, max(delay, 0.0))


def spread(interval: float, fraction: float = 0.2, rng: random.Random | None = None) -> float:
    """`interval` +/- `fraction` of it, for the sleep of a polling loop, so replicas started together do not
    keep polling in step."""
    return interval * (rng or _rng).uniform(1.0 - fraction, 1.0 + fraction)


def retry_after_header(seconds: float | None) -> str | None:
    """The `Retry-After` header value (whole seconds, at least 1) of `seconds`; None without one."""
    if seconds is None:
        return None
    return str(max(1, math.ceil(seconds)))


# --- bulkhead -------------------------------------------------------------------------------------------------


class Bulkhead:
    """At most `limit` callers inside at once, at most `queue` more waiting for a slot; one more is refused with
    `error` (503 `SERVICE_BUSY` by default) carrying `retry_after`. `limit` 0 means no limit at all.

    A caller reserves its place with `reserve()` (synchronous, so the count cannot be overtaken between the check and
    the wait) and enters with `async with ticket:`. A stage reserves before it claims a job, so a refused request
    leaves nothing behind, and only waits for the slot in the background task that runs the job."""

    def __init__(
        self,
        name: str,
        limit: int,
        queue: int = 0,
        *,
        retry_after: float = 5.0,
        error: Callable[[str, float], ServiceError] | None = None,
    ):
        self.name = name
        self.limit = max(limit, 0)
        self.queue = max(queue, 0)
        self.retry_after = retry_after
        self._error = error or (lambda message, after: ServiceBusy(message, retry_after=after))
        self._slots = asyncio.Semaphore(self.limit) if self.limit else None
        self._reserved = 0
        self._running = 0

    @property
    def enabled(self) -> bool:
        return self.limit > 0

    @property
    def reserved(self) -> int:
        """Callers holding a place: running plus waiting."""
        return self._reserved

    @property
    def running(self) -> int:
        return self._running

    def reserve(self, *, force: bool = False) -> "Ticket":
        """A place in the bulkhead, or `error` when running and waiting are both full. `force`: never refused (a
        job that is already claimed and must run, e.g. one the stale-job reaper hands back)."""
        if self.enabled and not force and self._reserved >= self.limit + self.queue:
            BULKHEAD_REJECTED.labels(self.name).inc()
            raise self._error(
                f"{self.name} is busy: {self._running} running and {self._reserved - self._running} waiting "
                f"(limit {self.limit} + queue {self.queue}); send again later",
                self.retry_after,
            )
        self._reserved += 1
        self._observe()
        return Ticket(self)

    def slot(self) -> "Ticket":
        """`async with bulkhead.slot():` for a caller that reserves and enters in one go."""
        return self.reserve()

    def _observe(self) -> None:
        BULKHEAD_RUNNING.labels(self.name).set(self._running)
        BULKHEAD_WAITING.labels(self.name).set(self._reserved - self._running)


class Ticket:
    """A reserved place in a `Bulkhead`: `async with ticket:` waits for a slot and gives it back; `release()`
    gives the place back without entering (the request turned out not to need it)."""

    def __init__(self, bulkhead: Bulkhead):
        self._bulkhead = bulkhead
        self._entered = False
        self._done = False

    async def __aenter__(self) -> Self:
        bulkhead = self._bulkhead
        if bulkhead._slots is not None:
            try:
                await bulkhead._slots.acquire()
            except BaseException:
                self.release()
                raise
        self._entered = True
        bulkhead._running += 1
        bulkhead._observe()
        return self

    async def __aexit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        bulkhead = self._bulkhead
        bulkhead._running -= 1
        if bulkhead._slots is not None:
            bulkhead._slots.release()
        self._entered = False
        self.release()

    def release(self) -> None:
        """Give the place back (idempotent)."""
        if self._done:
            return
        self._done = True
        self._bulkhead._reserved -= 1
        self._bulkhead._observe()


# --- circuit breaker --------------------------------------------------------------------------------------------


class CircuitBreaker:
    """Stops calling a remote that keeps failing. After `failures` failures in a row the circuit opens: calls are
    refused at once with 503 (and `Retry-After`) for `reset_seconds`. Then one call is let through as a probe:
    success closes the circuit, failure opens it again. `failures` 0 turns the breaker off.

    A failure is a 5xx-class `ServiceError` from the call (unreachable, timed out, a 5xx answer); a 4xx is the
    remote answering properly and counts as success."""

    def __init__(
        self, name: str, failures: int, reset_seconds: float = 30.0, *, clock: Callable[[], float] = time.monotonic
    ):
        self.name = name
        self.failures = max(failures, 0)
        self.reset_seconds = reset_seconds
        self._clock = clock
        self._consecutive = 0
        self._opened_at: float | None = None
        self._probing = False
        CIRCUIT_OPEN.labels(name).set(0)

    @property
    def enabled(self) -> bool:
        return self.failures > 0

    @property
    def state(self) -> str:
        if self._opened_at is None:
            return "closed"
        if self._probing:
            return "half_open"
        return "open" if self._clock() - self._opened_at < self.reset_seconds else "half_open"

    def before_call(self) -> None:
        """Raises 503 while the circuit is open, or while another call is already probing it."""
        if not self.enabled or self._opened_at is None:
            return
        remaining = self.reset_seconds - (self._clock() - self._opened_at)
        if remaining > 0:
            raise UpstreamUnavailable(
                f"{self.name} is failing; not calling it for another {math.ceil(remaining)}s (circuit open)",
                retry_after=remaining,
            )
        if self._probing:
            raise UpstreamUnavailable(
                f"{self.name} is failing; waiting for the probe call (circuit half-open)", retry_after=1.0
            )
        self._probing = True

    def record_success(self) -> None:
        if self._opened_at is not None:
            logger.info("circuit %s closed: the probe call succeeded", self.name)
        self._consecutive = 0
        self._opened_at = None
        self._probing = False
        CIRCUIT_OPEN.labels(self.name).set(0)

    def record_failure(self) -> None:
        if not self.enabled:
            return
        self._consecutive += 1
        if self._probing or self._consecutive >= self.failures:
            reopening = self._probing
            self._opened_at = self._clock()
            self._probing = False
            CIRCUIT_OPEN.labels(self.name).set(1)
            CIRCUIT_OPENED.labels(self.name).inc()
            logger.warning(
                "circuit %s %s after %d failure(s) in a row; refusing calls for %.0fs",
                self.name,
                "opened again" if reopening else "opened",
                self._consecutive,
                self.reset_seconds,
            )

    def record(self, error: BaseException | None) -> None:
        """Account for the end of a call: `error` None or a 4xx is a success, a 5xx-class `ServiceError` a
        failure; anything else (a bug, a cancellation) leaves the count alone."""
        if error is None or (isinstance(error, ServiceError) and error.status_code < 500):
            self.record_success()
        elif isinstance(error, ServiceError):
            self.record_failure()
        elif self._probing:
            # The probe never finished (cancelled): let the next call probe instead.
            self._probing = False
