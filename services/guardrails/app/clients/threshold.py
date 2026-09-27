"""The threshold of the guardrails model. It is owned by the central orchestrator, so it can be changed
there without a deploy here; this service reads it from the orchestrator's endpoint and falls back to
its own default.

The orchestrator sets it on either side of the model's answer, and each side is compared with its own
probability (never converted with `1 - x`, so a value exactly at the limit is decided as stated):

- `accept`: a page is accepted when `proba_approve >= threshold` (it passed the minimum), else rejected.
- `reject`: a page is rejected when `proba_reject >= threshold`, else accepted (below the tolerance).

With the model's `proba_approve = 0.7` (`proba_reject = 0.3`): accept 0.6 -> accepted, reject 0.6 ->
accepted, accept 0.8 -> rejected, reject 0.3 -> rejected.
"""

import asyncio
import logging
import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from ocr_common.clients.remote import RemoteModelClient
from ocr_common.errors import ServiceError

logger = logging.getLogger(__name__)

ACCEPT = "accept"
REJECT = "reject"
Target = Literal["accept", "reject"]
TARGETS: tuple[Target, ...] = (ACCEPT, REJECT)
DEFAULT_REJECT_THRESHOLD = 0.5


@dataclass(frozen=True)
class Threshold:
    """A threshold and the side of the model's answer it applies to."""

    value: float
    target: Target = REJECT

    def rejects(self, proba_approve: float, proba_reject: float) -> bool:
        """Whether a page with these probabilities is rejected."""
        if self.target == ACCEPT:
            return proba_approve < self.value
        return proba_reject >= self.value


def default_threshold(configured: float | None, classifier: Any) -> Threshold:
    """The threshold used when the orchestrator does not give one, on the reject side:
    GUARDRAILS_REJECT_THRESHOLD when set, else the one stored in the model's checkpoint, else 0.5."""
    if configured is not None:
        return Threshold(configured, REJECT)
    return Threshold(float(getattr(classifier, "reject_threshold", DEFAULT_REJECT_THRESHOLD)), REJECT)


def parse_threshold(body: Any) -> Threshold:
    """`{"threshold": 0.6, "target": "accept"}` into a `Threshold`. Raises ValueError for anything else,
    including a value outside (0, 1) or a target that is not `accept` / `reject`."""
    if not isinstance(body, dict):
        raise ValueError(f"not a JSON object: {body!r}")
    value, target = body.get("threshold"), body.get("target")
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise ValueError(f"no numeric threshold in {body!r}")
    if not 0 < value < 1:
        raise ValueError(f"threshold must be between 0 and 1, got {value}")
    if target not in TARGETS:
        raise ValueError(f"target must be one of {', '.join(TARGETS)}, got {target!r}")
    return Threshold(float(value), target)


class GuardrailsThreshold:
    """The threshold in force. With a client (GUARDRAILS_THRESHOLD_URL set), `GET` on the orchestrator's
    endpoint, kept for `cache_seconds` so a document does not wait on an extra call. When that call fails
    or answers something that is not a threshold, the last one the orchestrator gave stays in force (the
    default if it never gave one), and the endpoint is tried again after `cache_seconds`."""

    def __init__(
        self,
        client: RemoteModelClient | None,
        path: str,
        default: Threshold,
        *,
        cache_seconds: float,
        clock: Callable[[], float] = time.monotonic,
    ):
        self._client = client
        self._path = path
        self.default = default
        self._cache_seconds = cache_seconds
        self._clock = clock
        self._value: Threshold | None = None
        self._checked_at: float | None = None
        self._lock = asyncio.Lock()

    async def get(self) -> Threshold:
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

    def _current(self) -> Threshold:
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
