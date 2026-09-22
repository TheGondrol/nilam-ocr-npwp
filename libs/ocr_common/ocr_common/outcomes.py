from datetime import UTC, datetime
from typing import Any, Protocol

from sqlalchemy import Table
from sqlalchemy.dialects import postgresql, sqlite
from sqlalchemy.ext.asyncio import AsyncConnection

from ocr_common.config import PipelineSettings
from ocr_common.npwp import DOCUMENT_TYPE
from ocr_common.tables import orchestration_outcome_table

STATUS_PROCESSING = "processing"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"


class StageOutcome(Protocol):
    async def claimed(self, conn: AsyncConnection, request_id: str) -> None: ...

    async def completed(self, conn: AsyncConnection, request_id: str, data: dict[str, Any] | None) -> None: ...

    async def failed(self, conn: AsyncConnection, request_id: str, error_message: str) -> None: ...


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

    async def failed(self, conn: AsyncConnection, request_id: str, error_message: str) -> None:
        await self._write(
            conn,
            request_id,
            422,
            STATUS_FAILED,
            error_code=f"{self._stage}_FAILED",
            error_message=error_message,
        )

    async def _write(
        self,
        conn: AsyncConnection,
        request_id: str,
        status_code: int,
        downstream_status: str,
        *,
        result_data: dict[str, Any] | None = None,
        error_code: str | None = None,
        error_message: str | None = None,
    ) -> None:
        values: dict[str, Any] = {
            "status_code": status_code,
            "downstream_status": downstream_status,
            "downstream_stage": self._stage,
            "error_code": error_code,
            "error_message": error_message,
            "result_data": result_data,
        }
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
    return OrchestrationOutcome(orchestration_outcome_table(settings.orchestration_outcome_table), stage=stage)
