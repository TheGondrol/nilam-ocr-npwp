from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Protocol

from ocr_common.config import PipelineSettings
from ocr_common.npwp import DOCUMENT_TYPE

if TYPE_CHECKING:
    from sqlalchemy import Table
    from sqlalchemy.ext.asyncio import AsyncConnection

STATUS_PROCESSING = "processing"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"


class StageOutcome(Protocol):
    async def claimed(self, conn: AsyncConnection, request_id: str) -> None: ...

    async def completed(self, conn: AsyncConnection, request_id: str, data: dict[str, Any] | None) -> None: ...

    async def failed(
        self, conn: AsyncConnection, request_id: str, error_message: str, *, stage: str | None = None
    ) -> None: ...


class OrchestrationOutcome:
    def __init__(self, table: Table, *, stage: str, document_type: str = DOCUMENT_TYPE):
        self.table = table
        self._stage = stage
        self._document_type = document_type

    async def claimed(self, conn: AsyncConnection, request_id: str) -> None:
        await self._write(conn, request_id, 202, STATUS_PROCESSING)

    async def completed(self, conn: AsyncConnection, request_id: str, data: dict[str, Any] | None) -> None:
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


def build_stage_outcome(settings: PipelineSettings, *, stage: str) -> OrchestrationOutcome | None:
    if not settings.orchestration_outcome_table:
        return None
    from ocr_common.tables import orchestration_outcome_table

    return OrchestrationOutcome(orchestration_outcome_table(settings.orchestration_outcome_table), stage=stage)
