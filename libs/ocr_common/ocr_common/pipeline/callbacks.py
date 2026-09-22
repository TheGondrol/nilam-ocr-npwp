import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any, Protocol

from ocr_common.clients.remote import RemoteModelClient
from ocr_common.errors import ServiceError

logger = logging.getLogger(__name__)


async def with_retry(call: Callable[[], Awaitable[Any]], attempts: int, delay: float) -> Any:
    for attempt in range(1, attempts + 1):
        try:
            return await call()
        except ServiceError as exc:
            if exc.status_code < 500 or attempt >= attempts:
                raise
            await asyncio.sleep(delay * 2 ** (attempt - 1))


class StageCallback(Protocol):
    async def notify(
        self,
        request_id: str,
        stage: str,
        status: str,
        *,
        result: dict[str, Any] | None = None,
        error_message: str | None = None,
    ) -> bool: ...

    async def send(self, body: dict[str, Any]) -> None: ...

    async def aclose(self) -> None: ...


class NextStage(Protocol):
    async def submit(self, payload: dict[str, Any]) -> None: ...

    async def send(self, payload: dict[str, Any]) -> None: ...

    async def aclose(self) -> None: ...


class OrchestrationCallback:
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
        if self._client is None:
            logger.info("callback skipped (ORCHESTRATION_URL not set): %s", body.get("request_id"))
            return
        await self._client.post_json(self._path, body)

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()


class NextStageClient:
    def __init__(self, client: RemoteModelClient, path: str, *, attempts: int = 3, delay: float = 0.5):
        self._client = client
        self._path = path
        self._attempts = attempts
        self._delay = delay

    async def submit(self, payload: dict[str, Any]) -> None:
        await with_retry(lambda: self._client.post_json(self._path, payload), self._attempts, self._delay)

    async def send(self, payload: dict[str, Any]) -> None:
        await self._client.post_json(self._path, payload)

    async def aclose(self) -> None:
        await self._client.aclose()
