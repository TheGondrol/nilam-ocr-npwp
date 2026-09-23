"""The two HTTP calls a stage makes when a job finishes: the callback to the orchestrator and the
hand-off to the next stage. `with_retry` is the retry policy of the direct (non-outbox) mode.
"""

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any, Protocol

from ocr_common.clients.remote import RemoteModelClient
from ocr_common.errors import ServiceError

logger = logging.getLogger(__name__)


async def with_retry(call: Callable[[], Awaitable[Any]], attempts: int, delay: float) -> Any:
    """Calls `call` up to `attempts` times, doubling `delay` between tries, on 5xx-class `ServiceError`s only;
    a 4xx is raised at once.
    """
    for attempt in range(1, attempts + 1):
        try:
            return await call()
        except ServiceError as exc:
            if exc.status_code < 500 or attempt >= attempts:
                raise
            await asyncio.sleep(delay * 2 ** (attempt - 1))


class StageCallback(Protocol):
    """What the pipeline needs from the callback to the orchestrator."""

    async def notify(
        self,
        request_id: str,
        stage: str,
        status: str,
        *,
        result: dict[str, Any] | None = None,
        error_message: str | None = None,
    ) -> bool:
        """Direct mode: build and send the callback with retries; returns False when it was skipped or gave up."""
        ...

    async def send(self, body: dict[str, Any]) -> None:
        """Outbox mode: send an already-built callback body once; raises `ServiceError` on failure."""
        ...

    async def aclose(self) -> None:
        """Close the HTTP client."""
        ...


class NextStage(Protocol):
    """What the pipeline needs from the client of the next stage."""

    async def submit(self, payload: dict[str, Any]) -> None:
        """Direct mode: POST the hand-off with retries."""
        ...

    async def send(self, payload: dict[str, Any]) -> None:
        """Outbox mode: POST the hand-off once; raises `ServiceError` on failure."""
        ...

    async def aclose(self) -> None:
        """Close the HTTP client."""
        ...


class OrchestrationCallback:
    """The callback to `ORCHESTRATION_URL`; with no client (URL unset) every call is skipped and logged."""

    def __init__(self, client: RemoteModelClient | None, path: str, *, attempts: int = 3, delay: float = 0.5):
        self._client = client
        self._path = path
        self._attempts = attempts
        self._delay = delay

    async def notify(
        self,
        request_id: str,
        stage: str,
        status: str,
        *,
        result: dict[str, Any] | None = None,
        error_message: str | None = None,
    ) -> bool:
        """Send `{request_id, stage, status, result, error_message}` with retries; False when skipped or failed."""
        if self._client is None:
            logger.info("callback skipped (ORCHESTRATION_URL not set): %s %s %s", request_id, stage, status)
            return False
        client = self._client
        payload = {
            "request_id": request_id,
            "stage": stage,
            "status": status,
            "result": result,
            "error_message": error_message,
        }
        try:
            await with_retry(lambda: client.post_json(self._path, payload), self._attempts, self._delay)
        except ServiceError as exc:
            logger.error("callback failed: %s %s %s: %s", request_id, stage, status, exc.message)
            return False
        return True

    async def send(self, body: dict[str, Any]) -> None:
        """Send one callback body without retries (the outbox relay retries)."""
        if self._client is None:
            logger.info("callback skipped (ORCHESTRATION_URL not set): %s", body.get("request_id"))
            return
        await self._client.post_json(self._path, body)

    async def aclose(self) -> None:
        """Close the HTTP client, if any."""
        if self._client is not None:
            await self._client.aclose()


class NextStageClient:
    """POSTs the hand-off body to the next stage's `/v1/<stage>/jobs`."""

    def __init__(self, client: RemoteModelClient, path: str, *, attempts: int = 3, delay: float = 0.5):
        self._client = client
        self._path = path
        self._attempts = attempts
        self._delay = delay

    async def submit(self, payload: dict[str, Any]) -> None:
        """POST with retries (direct mode)."""
        await with_retry(lambda: self._client.post_json(self._path, payload), self._attempts, self._delay)

    async def send(self, payload: dict[str, Any]) -> None:
        """POST once (outbox mode)."""
        await self._client.post_json(self._path, payload)

    async def aclose(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()
