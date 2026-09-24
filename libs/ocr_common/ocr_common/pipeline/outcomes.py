"""How the orchestrator learns how a request ends without any callback, written by the stages inside
their job transactions (so the stage's own tables and the orchestrator's are written together or not
at all): the outcome row (`ORCHESTRATION_OUTCOME_TABLE`, one upserted row per request) and/or the API
event log (`ORCHESTRATION_API_EVENTS_TABLE`, one appended row per final state).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Protocol

from ocr_common.config import PipelineSettings
from ocr_common.npwp import DOCUMENT_TYPE, REJECTED_CODE

if TYPE_CHECKING:
    from sqlalchemy import Table
    from sqlalchemy.ext.asyncio import AsyncConnection

STATUS_PROCESSING = "processing"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"


class StageOutcome(Protocol):
    """What the repository needs from the outcome writer; every call receives the job's connection."""

    async def claimed(self, conn: AsyncConnection, request_id: str) -> None:
        """The request is `processing` at this stage."""
        ...

    async def completed(self, conn: AsyncConnection, request_id: str, data: dict[str, Any] | None) -> None:
        """The request is `completed` with `data` in the `extract-ocr` contract (scoring only)."""
        ...

    async def failed(
        self, conn: AsyncConnection, request_id: str, error_message: str, *, stage: str | None = None
    ) -> None:
        """The request `failed` at this stage, or at `stage` when a hand-off to it failed."""
        ...

    async def rejected(self, conn: AsyncConnection, request_id: str, reason: str) -> None:
        """This stage rejected the document (a 400 with `reason`); the pipeline stops here."""
        ...


class OrchestrationOutcome:
    """Upserts one row per `request_id` in the orchestrator's table with the pipeline's status."""

    def __init__(self, table: Table, *, stage: str, document_type: str = DOCUMENT_TYPE):
        self.table = table
        self._stage = stage
        self._document_type = document_type

    async def claimed(self, conn: AsyncConnection, request_id: str) -> None:
        """Upsert `processing` plus this stage."""
        await self._write(conn, request_id, 202, STATUS_PROCESSING)

    async def completed(self, conn: AsyncConnection, request_id: str, data: dict[str, Any] | None) -> None:
        """Upsert `completed` with `result_data`; nothing when `data` is None (not the last stage)."""
        if data is None:
            return
        await self._write(conn, request_id, 200, STATUS_COMPLETED, result_data=data)

    async def failed(
        self, conn: AsyncConnection, request_id: str, error_message: str, *, stage: str | None = None
    ) -> None:
        """`stage` names the stage that failed when it is not this one: a hand-off that the next stage
        never accepted is reported as that stage's failure, like the FAILED callback would be."""
        await self._write(
            conn,
            request_id,
            422,
            STATUS_FAILED,
            stage=stage,
            error_code=f"{stage or self._stage}_FAILED",
            error_message=error_message,
        )

    async def rejected(self, conn: AsyncConnection, request_id: str, reason: str) -> None:
        """Upsert `failed` with 400 and `DOWNSTREAM_VALIDATION_ERROR`, the reason as the message."""
        await self._write(conn, request_id, 400, STATUS_FAILED, error_code=REJECTED_CODE, error_message=reason)

    async def _write(
        self,
        conn: AsyncConnection,
        request_id: str,
        status_code: int,
        downstream_status: str,
        *,
        stage: str | None = None,
        result_data: dict[str, Any] | None = None,
        error_code: str | None = None,
        error_message: str | None = None,
    ) -> None:
        values: dict[str, Any] = {
            "status_code": status_code,
            "downstream_status": downstream_status,
            "downstream_stage": stage or self._stage,
            "error_code": error_code,
            "error_message": error_message,
            "result_data": result_data,
        }
        from sqlalchemy.dialects import postgresql, sqlite

        dialect = postgresql if conn.dialect.name == "postgresql" else sqlite
        statement = dialect.insert(self.table).values(
            request_id=request_id,
            document_type=self._document_type,
            ds=datetime.now(UTC).strftime("%Y%m%d"),
            **values,
        )
        await conn.execute(statement.on_conflict_do_update(index_elements=["request_id"], set_=values))


# Our stage names -> the orchestrator's `downstream_stage` enum.
API_EVENT_STAGE = {"OCR": "EXTRACTION", "STRUCTURING": "STRUCTURING", "SCORING": "SCORING", "GUARDRAILS": "GUARDRAILS"}
API_EVENT_ENDPOINT = "GET_OCR_RESULT"


class ApiEventOutcome:
    """Appends the request's final state to the orchestrator's API event log
    (`ORCHESTRATION_API_EVENTS_TABLE`), in the shape the orchestrator itself writes when a client polls a
    finished request: `endpoint = GET_OCR_RESULT`, `downstream_status` COMPLETED / FAILED, and
    `result_data = {result, status, document_type, error_code, error_message, created_at, updated_at}`.

    Only terminal states are written; the log has no key on `request_id`, so a request that is run again
    gets another row, and the newest row per `request_id` is its state."""

    def __init__(self, table: Table, *, stage: str, document_type: str = DOCUMENT_TYPE):
        self.table = table
        self._stage = stage
        self._document_type = document_type

    async def claimed(self, conn: AsyncConnection, request_id: str) -> None:
        """Nothing: the orchestrator already logged the 202 of `extract-ocr`."""

    async def completed(self, conn: AsyncConnection, request_id: str, data: dict[str, Any] | None) -> None:
        """A COMPLETED row with the `extract-ocr` data; nothing when `data` is None (not the last stage)."""
        if data is None:
            return
        result = {"document_type": self._document_type, **data, "guardrails": 1}
        await self._append(conn, request_id, 200, STATUS_COMPLETED, self._stage, result=result)

    async def failed(
        self, conn: AsyncConnection, request_id: str, error_message: str, *, stage: str | None = None
    ) -> None:
        """A FAILED row naming the stage that failed (`stage` when a hand-off to it failed)."""
        failed_stage = stage or self._stage
        await self._append(
            conn,
            request_id,
            422,
            STATUS_FAILED,
            failed_stage,
            error_code=f"{failed_stage}_FAILED",
            error_message=error_message,
        )

    async def rejected(self, conn: AsyncConnection, request_id: str, reason: str) -> None:
        """A FAILED row with 400 and `DOWNSTREAM_VALIDATION_ERROR`, the reason as the message."""
        await self._append(
            conn, request_id, 400, STATUS_FAILED, self._stage, error_code=REJECTED_CODE, error_message=reason
        )

    async def _append(
        self,
        conn: AsyncConnection,
        request_id: str,
        status_code: int,
        status: str,
        stage: str,
        *,
        result: dict[str, Any] | None = None,
        error_code: str | None = None,
        error_message: str | None = None,
    ) -> None:
        now = datetime.now(UTC)
        await conn.execute(
            self.table.insert().values(
                endpoint=API_EVENT_ENDPOINT,
                request_id=request_id,
                status_code=status_code,
                error_code=error_code,
                result_data={
                    "result": result,
                    "status": status,
                    "document_type": self._document_type,
                    "error_code": error_code,
                    "error_message": error_message,
                    "created_at": now.isoformat(),
                    "updated_at": now.isoformat(),
                },
                ds=now.strftime("%Y%m%d"),
                downstream_status=status.upper(),
                downstream_stage=API_EVENT_STAGE.get(stage),
                document_type=self._document_type,
            )
        )


class CompositeOutcome:
    """Several outcome writers on the same job transaction, called in order (the double write)."""

    def __init__(self, writers: list[StageOutcome]):
        self.writers = writers

    async def claimed(self, conn: AsyncConnection, request_id: str) -> None:
        for writer in self.writers:
            await writer.claimed(conn, request_id)

    async def completed(self, conn: AsyncConnection, request_id: str, data: dict[str, Any] | None) -> None:
        for writer in self.writers:
            await writer.completed(conn, request_id, data)

    async def failed(
        self, conn: AsyncConnection, request_id: str, error_message: str, *, stage: str | None = None
    ) -> None:
        for writer in self.writers:
            await writer.failed(conn, request_id, error_message, stage=stage)

    async def rejected(self, conn: AsyncConnection, request_id: str, reason: str) -> None:
        for writer in self.writers:
            await writer.rejected(conn, request_id, reason)


def build_stage_outcome(settings: PipelineSettings, *, stage: str) -> StageOutcome | None:
    """The outcome writer(s) for `stage` from settings: the outcome row (`ORCHESTRATION_OUTCOME_TABLE`),
    the API event log (`ORCHESTRATION_API_EVENTS_TABLE`), both, or None when neither is set."""
    from ocr_common.pipeline.tables import orchestration_api_events_table, orchestration_outcome_table

    writers: list[StageOutcome] = []
    if settings.orchestration_outcome_table:
        writers.append(
            OrchestrationOutcome(orchestration_outcome_table(settings.orchestration_outcome_table), stage=stage)
        )
    if settings.orchestration_api_events_table:
        writers.append(
            ApiEventOutcome(orchestration_api_events_table(settings.orchestration_api_events_table), stage=stage)
        )
    if not writers:
        return None
    return writers[0] if len(writers) == 1 else CompositeOutcome(writers)
