"""The reject threshold of the guardrails model. It is owned by the central orchestrator, so it can be
changed there without a deploy here; this service reads it from the orchestrator's endpoint and falls
back to its own default."""

import asyncio
import logging
import math
import time
from collections.abc import Callable
from typing import Any

from ocr_common.clients.remote import RemoteModelClient
from ocr_common.errors import ServiceError

logger = logging.getLogger(__name__)

DEFAULT_REJECT_THRESHOLD = 0.5


def default_threshold(configured: float | None, classifier: Any) -> float:
    """The threshold used when the orchestrator does not give one: GUARDRAILS_REJECT_THRESHOLD when set,
    else the one stored in the model's checkpoint, else 0.5."""
    if configured is not None:
        return configured
    return float(getattr(classifier, "reject_threshold", DEFAULT_REJECT_THRESHOLD))


def parse_threshold(body: Any) -> float:
    """`{"reject_threshold": 0.5}` into 0.5. Raises ValueError for anything else, including a value
    outside (0, 1): 0 would reject every page, 1 almost none."""
    value = body.get("reject_threshold") if isinstance(body, dict) else None
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise ValueError(f"no numeric reject_threshold in {body!r}")
    if not 0 < value < 1:
        raise ValueError(f"reject_threshold must be between 0 and 1, got {value}")
    return float(value)


class RejectThreshold:
    """The threshold in force. With a client (GUARDRAILS_THRESHOLD_URL set), `GET` on the orchestrator's
    endpoint, kept for `cache_seconds` so a document does not wait on an extra call. When that call fails
    or answers something that is not a threshold, the last value the orchestrator gave stays in force
    (the default if it never gave one), and the endpoint is tried again after `cache_seconds`."""

    def __init__(
        self,
        client: RemoteModelClient | None,
        path: str,
        default: float,
        *,
        cache_seconds: float,
        clock: Callable[[], float] = time.monotonic,
    ):
        self._client = client
        self._path = path
        self.default = default
        self._cache_seconds = cache_seconds
        self._clock = clock
        self._value: float | None = None
        self._checked_at: float | None = None
        self._lock = asyncio.Lock()

    async def get(self) -> float:
        if self._client is None:
            return self.default
        if not self._due():
            return self._current()
        async with self._lock:
            if self._due():  # another request may have fetched it while this one waited for the lock
                await self._fetch(self._client)
        return self._current()

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()

    def _due(self) -> bool:
        return self._checked_at is None or self._clock() - self._checked_at >= self._cache_seconds

    def _current(self) -> float:
        return self.default if self._value is None else self._value

    async def _fetch(self, client: RemoteModelClient) -> None:
        try:
            value = parse_threshold(await client.get_json(self._path))
        except (ServiceError, ValueError) as exc:
            message = exc.message if isinstance(exc, ServiceError) else str(exc)
            logger.warning(
                "reject threshold from the orchestrator unavailable (%s); using %s", message, self._current()
            )
        else:
            if value != self._value:
                logger.info("reject threshold from the orchestrator: %s", value)
            self._value = value
        self._checked_at = self._clock()
