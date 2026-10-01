"""The threshold of the guardrails model. It is owned by the central orchestrator, so it can be changed
there without a deploy here; this service reads it from the orchestrator's endpoint and falls back to
its own default.

There is one threshold, on the model's accepted probability: a page is accepted when
`proba_approve >= threshold` and rejected when it is below (a value exactly at the limit is accepted).

With the model's `proba_approve = 0.7`: threshold 0.6 -> accepted, threshold 0.8 -> rejected.
"""

import asyncio
import logging
import math
import time
from collections.abc import Callable
from typing import Any

from ocr_common.clients.remote import RemoteModelClient
from ocr_common.errors import ServiceError

logger = logging.getLogger(__name__)

DEFAULT_THRESHOLD = 0.5


def rejects(proba_approve: float, threshold: float) -> bool:
    """Whether a page with this accepted probability is rejected."""
    return proba_approve < threshold


def default_threshold(configured: float | None, classifier: Any) -> float:
    """The threshold used when the orchestrator does not give one:
    GUARDRAILS_THRESHOLD when set, else the one derived from the model's checkpoint, else 0.5."""
    if configured is not None:
        return configured
    return float(getattr(classifier, "accept_threshold", DEFAULT_THRESHOLD))


def parse_threshold(body: Any) -> float:
    """`{"threshold": 0.6}` into the threshold. Raises ValueError for anything else, including a value
    outside (0, 1)."""
    if not isinstance(body, dict):
        raise ValueError(f"not a JSON object: {body!r}")
    value = body.get("threshold")
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise ValueError(f"no numeric threshold in {body!r}")
    if not 0 < value < 1:
        raise ValueError(f"threshold must be between 0 and 1, got {value}")
    return float(value)


class GuardrailsThreshold:
    """The threshold in force. With a client (GUARDRAILS_THRESHOLD_URL set), `GET` on the orchestrator's
    endpoint, kept for `cache_seconds` so a document does not wait on an extra call. When that call fails
    or answers something that is not a threshold, the last one the orchestrator gave stays in force (the
    default if it never gave one), and the endpoint is tried again after `cache_seconds`."""

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
            logger.warning("threshold from the orchestrator unavailable (%s); using %s", message, self._current())
        else:
            if value != self._value:
                logger.info("threshold from the orchestrator: %s", value)
            self._value = value
        self._checked_at = self._clock()
