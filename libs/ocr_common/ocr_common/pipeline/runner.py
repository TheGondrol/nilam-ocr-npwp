import asyncio
from collections.abc import Coroutine
from typing import Any

CANCEL_GRACE_SECONDS = 5.0


class BackgroundRunner:
    def __init__(self) -> None:
        self._tasks: set[asyncio.Task[None]] = set()

    def spawn(self, coro: Coroutine[Any, Any, None]) -> None:
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def drain(self, timeout: float, *, cancel_grace: float = CANCEL_GRACE_SECONDS) -> None:
        if not self._tasks:
            return
        _, pending = await asyncio.wait(self._tasks, timeout=timeout)
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.wait(pending, timeout=cancel_grace)
