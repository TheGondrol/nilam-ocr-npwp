import asyncio
import logging
from collections.abc import Awaitable, Callable, Coroutine
from datetime import UTC, datetime
from typing import Any, Protocol, TypedDict

from ocr_common.config import PipelineSettings
from ocr_common.errors import ServiceError
from ocr_common.remote import RemoteModelClient

logger = logging.getLogger(__name__)

STATUS_PROCESSING = "PROCESSING"
STATUS_DONE = "DONE"
STATUS_FAILED = "FAILED"

STAGE_OCR = "OCR"
STAGE_STRUCTURING = "STRUCTURING"
STAGE_SCORING = "SCORING"

Work = Callable[[], Awaitable[dict[str, Any]]]
Handoff = Callable[[dict[str, Any]], Awaitable[None]]
CallbackResult = Callable[[dict[str, Any]], dict[str, Any]]


class JobRecord(TypedDict):
    request_id: str
    status: str
    result: dict[str, Any] | None
    error_message: str | None
    created_at: str
    updated_at: str


class JobRepository(Protocol):
    name: str

    async def claim(self, request_id: str) -> bool: ...

    async def complete(self, request_id: str, result: dict[str, Any]) -> None: ...

    async def fail(self, request_id: str, error_message: str) -> None: ...

    async def get(self, request_id: str) -> JobRecord | None: ...


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


class InMemoryJobRepository:
    name = "memory"

    def __init__(self) -> None:
        self._jobs: dict[str, JobRecord] = {}

    async def claim(self, request_id: str) -> bool:
        now = _now_iso()
        record = self._jobs.get(request_id)
        if record is None:
            self._jobs[request_id] = {
                "request_id": request_id,
                "status": STATUS_PROCESSING,
                "result": None,
                "error_message": None,
                "created_at": now,
                "updated_at": now,
            }
            return True
        if record["status"] == STATUS_FAILED:
            record.update(status=STATUS_PROCESSING, error_message=None, updated_at=now)
            return True
        return False

    async def complete(self, request_id: str, result: dict[str, Any]) -> None:
        self._jobs[request_id].update(status=STATUS_DONE, result=result, updated_at=_now_iso())

    async def fail(self, request_id: str, error_message: str) -> None:
        self._jobs[request_id].update(status=STATUS_FAILED, error_message=error_message, updated_at=_now_iso())

    async def get(self, request_id: str) -> JobRecord | None:
        record = self._jobs.get(request_id)
        return record.copy() if record else None


def build_job_repository(database_url: str | None, table_prefix: str) -> JobRepository:
    if not database_url:
        return InMemoryJobRepository()
    from ocr_common.jobs_sql import SqlJobRepository

    return SqlJobRepository(database_url, table_prefix)


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

    async def aclose(self) -> None: ...


class NextStage(Protocol):
    async def submit(self, payload: dict[str, Any]) -> None: ...

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

    async def aclose(self) -> None:
        await self._client.aclose()


class BackgroundRunner:
    def __init__(self) -> None:
        self._tasks: set[asyncio.Task[None]] = set()

    def spawn(self, coro: Coroutine[Any, Any, None]) -> None:
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def drain(self, timeout: float) -> None:
        if not self._tasks:
            return
        _, pending = await asyncio.wait(self._tasks, timeout=timeout)
        for task in pending:
            task.cancel()


class StagePipeline:
    def __init__(
        self,
        *,
        stage: str,
        repository: JobRepository,
        callback: StageCallback,
        runner: BackgroundRunner | None = None,
    ):
        self.stage = stage
        self.repository = repository
        self.callback = callback
        self.runner = runner or BackgroundRunner()

    async def submit(
        self,
        request_id: str,
        work: Work,
        *,
        handoff: Handoff | None = None,
        next_stage: str | None = None,
        callback_result: CallbackResult | None = None,
    ) -> dict[str, Any]:
        claimed = await self.repository.claim(request_id)
        status = STATUS_PROCESSING
        if claimed:
            self.runner.spawn(self._run(request_id, work, handoff, next_stage, callback_result))
        else:
            record = await self.repository.get(request_id)
            status = record["status"] if record else STATUS_PROCESSING
        return {"request_id": request_id, "stage": self.stage, "status": status, "duplicate": not claimed}

    async def get(self, request_id: str) -> dict[str, Any]:
        record = await self.repository.get(request_id)
        if record is None:
            raise ServiceError(404, f"No {self.stage} job found for request_id: {request_id}")
        return {"stage": self.stage, **record}

    async def aclose(self, drain_timeout: float) -> None:
        await self.runner.drain(drain_timeout)
        await self.callback.aclose()

    async def _run(
        self,
        request_id: str,
        work: Work,
        handoff: Handoff | None,
        next_stage: str | None,
        callback_result: CallbackResult | None,
    ) -> None:
        try:
            result = await work()
            await self.repository.complete(request_id, result)
        except ServiceError as exc:
            await self._failed(request_id, exc.message)
            return
        except Exception:
            logger.exception("%s job %s crashed", self.stage, request_id)
            await self._failed(request_id, f"Internal error in {self.stage} stage")
            return

        await self.callback.notify(
            request_id, self.stage, STATUS_DONE, result=callback_result(result) if callback_result else None
        )
        if handoff is None:
            return
        try:
            await handoff(result)
        except ServiceError as exc:
            logger.error("%s job %s: handoff to %s failed: %s", self.stage, request_id, next_stage, exc.message)
            await self.callback.notify(
                request_id,
                next_stage or self.stage,
                STATUS_FAILED,
                error_message=f"Handoff to {next_stage} failed: {exc.message}",
            )

    async def _failed(self, request_id: str, error_message: str) -> None:
        try:
            await self.repository.fail(request_id, error_message)
        except Exception:
            logger.exception("%s job %s: could not record failure", self.stage, request_id)
        await self.callback.notify(request_id, self.stage, STATUS_FAILED, error_message=error_message)


def _remote(base_url: str, api_key: str, timeout: float, name: str) -> RemoteModelClient:
    return RemoteModelClient(
        base_url, timeout, name=name, headers={"X-API-Key": api_key}, passthrough_client_errors=True
    )


def build_stage_pipeline(settings: PipelineSettings, *, stage: str, table_prefix: str) -> StagePipeline:
    client = None
    if settings.orchestration_url:
        client = _remote(
            settings.orchestration_url,
            settings.orchestration_api_key or settings.api_key,
            settings.orchestration_timeout_seconds,
            "orchestration callback",
        )
    callback = OrchestrationCallback(
        client,
        settings.orchestration_callback_path,
        attempts=settings.pipeline_retry_attempts,
        delay=settings.pipeline_retry_delay_seconds,
    )
    return StagePipeline(
        stage=stage, repository=build_job_repository(settings.database_url, table_prefix), callback=callback
    )


def build_next_stage_client(
    settings: PipelineSettings, *, base_url: str, api_key: str | None, timeout: float, path: str, name: str
) -> NextStageClient:
    return NextStageClient(
        _remote(base_url, api_key or settings.api_key, timeout, name),
        path,
        attempts=settings.pipeline_retry_attempts,
        delay=settings.pipeline_retry_delay_seconds,
    )
