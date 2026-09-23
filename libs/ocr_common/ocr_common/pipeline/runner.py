"""Background tasks of a process and their graceful shutdown."""

import asyncio
from collections.abc import Coroutine
from typing import Any

CANCEL_GRACE_SECONDS = 5.0


class BackgroundRunner:
    """Keeps a strong reference to every spawned task and drains them on shutdown."""

    def __init__(self) -> None:
        self._tasks: set[asyncio.Task[None]] = set()

    def spawn(self, coro: Coroutine[Any, Any, None]) -> None:
        """Run `coro` as a task now."""
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def drain(self, timeout: float, *, cancel_grace: float = CANCEL_GRACE_SECONDS) -> None:
        """Wait up to `timeout` for the running tasks, then cancel the rest and give them `cancel_grace`
        seconds to record their failure.
        """
        if not self._tasks:
            return
        _, pending = await asyncio.wait(self._tasks, timeout=timeout)
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.wait(pending, timeout=cancel_grace)
