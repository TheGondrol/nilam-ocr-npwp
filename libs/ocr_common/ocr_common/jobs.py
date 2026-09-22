import asyncio
import logging
from collections.abc import Awaitable, Callable, Coroutine, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol, TypedDict

from ocr_common.config import DEFAULT_JOB_LEASE_SECONDS, PipelineSettings
from ocr_common.errors import ServiceError
from ocr_common.outbox import Outbox, OutboxMessage, OutboxRelay, callback_message, handoff_message
from ocr_common.outcomes import StageOutcome, build_stage_outcome
from ocr_common.remote import RemoteModelClient

logger = logging.getLogger(__name__)

STATUS_PROCESSING = "PROCESSING"
STATUS_DONE = "DONE"
STATUS_FAILED = "FAILED"

STAGE_OCR = "OCR"
STAGE_STRUCTURING = "STRUCTURING"
STAGE_SCORING = "SCORING"

CANCEL_GRACE_SECONDS = 5.0

Work = Callable[[], Awaitable[dict[str, Any]]]
CallbackResult = Callable[[dict[str, Any]], dict[str, Any]]
HandoffPayload = Callable[[dict[str, Any]], dict[str, Any]]


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

    async def complete(
        self,
        request_id: str,
        result: dict[str, Any],
        *,
        outcome_data: dict[str, Any] | None = None,
        messages: Sequence[OutboxMessage] = (),
    ) -> None: ...

    async def fail(self, request_id: str, error_message: str, *, messages: Sequence[OutboxMessage] = ()) -> None: ...

    async def handoff_failed(self, request_id: str, next_stage: str, error_message: str) -> None: ...

    async def get(self, request_id: str) -> JobRecord | None: ...


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


class InMemoryJobRepository:
    name = "memory"

    def __init__(self, lease_seconds: float = DEFAULT_JOB_LEASE_SECONDS) -> None:
        self._jobs: dict[str, JobRecord] = {}
        self._lease = timedelta(seconds=lease_seconds)

    async def claim(self, request_id: str) -> bool:
        current = datetime.now(UTC)
        now = current.isoformat()
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
        expired = (
            record["status"] == STATUS_PROCESSING
            and datetime.fromisoformat(record["updated_at"]) < current - self._lease
        )
        if record["status"] == STATUS_FAILED or expired:
            record.update(status=STATUS_PROCESSING, error_message=None, updated_at=now)
            return True
        return False

    async def complete(
        self,
        request_id: str,
        result: dict[str, Any],
        *,
        outcome_data: dict[str, Any] | None = None,
        messages: Sequence[OutboxMessage] = (),
    ) -> None:
        self._jobs[request_id].update(status=STATUS_DONE, result=result, updated_at=_now_iso())

    async def fail(self, request_id: str, error_message: str, *, messages: Sequence[OutboxMessage] = ()) -> None:
        self._jobs[request_id].update(status=STATUS_FAILED, error_message=error_message, updated_at=_now_iso())

    async def handoff_failed(self, request_id: str, next_stage: str, error_message: str) -> None:
        pass

    async def get(self, request_id: str) -> JobRecord | None:
        record = self._jobs.get(request_id)
        return record.copy() if record else None


def build_job_repository(
    database_url: str | None,
    table_prefix: str,
    *,
    lease_seconds: float = DEFAULT_JOB_LEASE_SECONDS,
    outcome: StageOutcome | None = None,
    outbox: Outbox | None = None,
    stage: str = "",
) -> JobRepository:
    if not database_url:
        return InMemoryJobRepository(lease_seconds)
    from ocr_common.jobs_sql import SqlJobRepository

    return SqlJobRepository(
        database_url, table_prefix, lease_seconds=lease_seconds, outcome=outcome, outbox=outbox, stage=stage
    )


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


class StagePipeline:
    def __init__(
        self,
        *,
        stage: str,
        repository: JobRepository,
        callback: StageCallback,
        next_stage_client: NextStage | None = None,
        outbox: Outbox | None = None,
        runner: BackgroundRunner | None = None,
        callbacks: bool = True,
    ):
        """`callbacks=False` when the orchestrator has no callback endpoint: no callback is sent or
        queued, and the outcome reaches the orchestrator through its table (ORCHESTRATION_OUTCOME_TABLE)
        and GET .../jobs/{request_id}."""
        self.stage = stage
        self.repository = repository
        self.callback = callback
        self.next_stage_client = next_stage_client
        self.outbox = outbox
        self.runner = runner or BackgroundRunner()
        self.callbacks = callbacks

    async def submit(
        self,
        request_id: str,
        work: Work,
        *,
        handoff_payload: HandoffPayload | None = None,
        next_stage: str | None = None,
        callback_result: CallbackResult | None = None,
        outcome_data: CallbackResult | None = None,
    ) -> dict[str, Any]:
        claimed = await self.repository.claim(request_id)
        status = STATUS_PROCESSING
        if claimed:
            self.runner.spawn(self._run(request_id, work, handoff_payload, next_stage, callback_result, outcome_data))
        else:
            record = await self.repository.get(request_id)
            status = record["status"] if record else STATUS_PROCESSING
        return {"request_id": request_id, "stage": self.stage, "status": status, "duplicate": not claimed}

    async def get(self, request_id: str) -> dict[str, Any]:
        record = await self.repository.get(request_id)
        if record is None:
            raise ServiceError(404, f"No {self.stage} job found for request_id: {request_id}")
        return {"stage": self.stage, **record}

    async def aclose(self, drain_timeout: float, *, relay: OutboxRelay | None = None) -> None:
        """Shutdown order: finish the jobs, then let the relay send what those jobs queued, then close
        the clients the relay uses."""
        await self.runner.drain(drain_timeout)
        if relay is not None:
            await relay.stop()
        await self.callback.aclose()

    async def _run(
        self,
        request_id: str,
        work: Work,
        handoff_payload: HandoffPayload | None,
        next_stage: str | None,
        callback_result: CallbackResult | None,
        outcome_data: CallbackResult | None = None,
    ) -> None:
        payload: dict[str, Any] | None = None
        final: dict[str, Any] | None = None
        try:
            result = await work()
            payload = handoff_payload(result) if handoff_payload else None
            final = callback_result(result) if callback_result else None
            await self.repository.complete(
                request_id,
                result,
                outcome_data=outcome_data(result) if outcome_data else None,
                messages=self._messages(request_id, final, payload, next_stage),
            )
        except asyncio.CancelledError:
            logger.warning("%s job %s interrupted by shutdown", self.stage, request_id)
            await self._failed(
                request_id, f"{self.stage} stage was interrupted by a service shutdown; submit the job again"
            )
            raise
        except ServiceError as exc:
            await self._failed(request_id, exc.message)
            return
        except Exception:
            logger.exception("%s job %s crashed", self.stage, request_id)
            await self._failed(request_id, f"Internal error in {self.stage} stage")
            return

        if self.outbox is not None:
            return

        try:
            if self.callbacks:
                await self.callback.notify(request_id, self.stage, STATUS_DONE, result=final)
            if payload is not None:
                await self._hand_off(payload)
        except asyncio.CancelledError:
            if payload is not None:
                await self._handoff_failed(request_id, next_stage, "interrupted by a service shutdown")
            raise
        except ServiceError as exc:
            await self._handoff_failed(request_id, next_stage, exc.message)

    def _messages(
        self,
        request_id: str,
        final: dict[str, Any] | None,
        payload: dict[str, Any] | None,
        next_stage: str | None,
    ) -> list[OutboxMessage]:
        if self.outbox is None:
            return []
        messages = []
        if self.callbacks:
            messages.append(callback_message(request_id, self.stage, STATUS_DONE, result=final))
        if payload is not None and next_stage is not None:
            messages.append(handoff_message(next_stage, payload))
        return messages

    async def _hand_off(self, payload: dict[str, Any]) -> None:
        if self.next_stage_client is None:
            raise ServiceError(500, f"the {self.stage} stage has no next stage to hand off to")
        await self.next_stage_client.submit(payload)

    async def _handoff_failed(self, request_id: str, next_stage: str | None, reason: str) -> None:
        logger.error("%s job %s: handoff to %s failed: %s", self.stage, request_id, next_stage, reason)
        message = f"Handoff to {next_stage} failed: {reason}"
        try:
            await self.repository.handoff_failed(request_id, next_stage or self.stage, message)
        except Exception:
            logger.exception("%s job %s: could not record the failed handoff", self.stage, request_id)
        if self.callbacks:
            await self.callback.notify(request_id, next_stage or self.stage, STATUS_FAILED, error_message=message)

    async def _failed(self, request_id: str, error_message: str) -> None:
        reported = self.outbox is not None
        try:
            await self.repository.fail(
                request_id,
                error_message,
                messages=[callback_message(request_id, self.stage, STATUS_FAILED, error_message=error_message)]
                if self.outbox is not None and self.callbacks
                else [],
            )
        except Exception:
            logger.exception("%s job %s: could not record failure", self.stage, request_id)
            reported = False
        if not reported and self.callbacks:
            await self.callback.notify(request_id, self.stage, STATUS_FAILED, error_message=error_message)


def _remote(base_url: str, api_key: str, timeout: float, name: str) -> RemoteModelClient:
    return RemoteModelClient(
        base_url, timeout, name=name, headers={"X-API-Key": api_key}, passthrough_client_errors=True
    )


def build_stage_pipeline(
    settings: PipelineSettings, *, stage: str, table_prefix: str, next_stage: NextStage | None = None
) -> StagePipeline:
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
    outbox = None
    if settings.pipeline_outbox and settings.database_url:
        from ocr_common.outbox_sql import SqlOutbox

        outbox = SqlOutbox(settings.database_url)
    repository = build_job_repository(
        settings.database_url,
        table_prefix,
        lease_seconds=settings.pipeline_job_lease_seconds,
        outcome=build_stage_outcome(settings, stage=stage),
        outbox=outbox,
        stage=stage,
    )
    return StagePipeline(
        stage=stage,
        repository=repository,
        callback=callback,
        next_stage_client=next_stage,
        outbox=outbox,
        callbacks=settings.callbacks_enabled,
    )


def build_outbox_relay(settings: PipelineSettings, pipeline: StagePipeline) -> OutboxRelay | None:
    from ocr_common.outbox_sql import SqlOutbox

    if not isinstance(pipeline.outbox, SqlOutbox):
        return None
    return OutboxRelay(
        pipeline.outbox,
        stage=pipeline.stage,
        callback=pipeline.callback,
        next_stage=pipeline.next_stage_client,
        callbacks=pipeline.callbacks,
        handoff_failed=pipeline.repository.handoff_failed,
        interval_seconds=settings.pipeline_outbox_interval_seconds,
        batch=settings.pipeline_outbox_batch,
        lease_seconds=settings.pipeline_outbox_lease_seconds,
        retry_delay_seconds=settings.pipeline_retry_delay_seconds,
        max_backoff_seconds=settings.pipeline_outbox_max_backoff_seconds,
        max_age_seconds=settings.pipeline_outbox_max_age_seconds,
        stale_after_seconds=settings.pipeline_outbox_stale_after_seconds,
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
