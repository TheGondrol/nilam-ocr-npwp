"""`nilam_ocr_results`: the request's final answer, one row per request_id in the shape of the extract-ocr answer
(`status_code`, `status_desc`, `message`, `data`, `errors`, `guardrails`, `pipeline_last_stage`). The service that
ends the request writes it in the same transaction as its own result: a stage through `OcrResultsOutcome` (a
`StageOutcome`), the orchestrator NPWP through `write_ocr_result` for guardrails.

A stage only writes when the request ends with it: the last service of the pipeline_name_sequence completed, the
structuring rules rejected the document, or a stage failed (its own work, or the hand-off to the next one). The
`processing` state is not written. The table is append-only: a request_id run again gets another row when it ends
again, and its newest row is its state now.
"""

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import Table, select
from sqlalchemy.ext.asyncio import AsyncConnection

from ocr_common.npwp import COMPLETED_MESSAGE, REJECTED_CODE
from ocr_common.pipeline.repository import stored_sequence
from ocr_common.pipeline.sequence import GUARDRAILS, SERVICE_OF_STAGE
from ocr_common.web.envelope import STATUS_DESC

# `guardrails`, as in the extract-ocr answer: 0 passed, 1 rejected (by the guardrails model or the structuring
# rules); null when the request left guardrails out.
GUARDRAILS_PASSED = 0
GUARDRAILS_REJECTED = 1


async def write_ocr_result(
    conn: AsyncConnection,
    table: Table,
    request_id: str,
    status_code: int,
    message: str | None,
    *,
    data: dict[str, Any] | None = None,
    errors: str | None = None,
    guardrails: int | None = None,
    pipeline_last_stage: str | None = None,
) -> None:
    """Append the final answer of `request_id` as a new row (the table is append-only; `update_at` =
    `created_at`). `pipeline_last_stage` is the service that ended the request."""
    now = datetime.now(UTC)
    await conn.execute(
        table.insert().values(
            request_id=request_id,
            status_code=status_code,
            status_desc=STATUS_DESC.get(status_code, "Error"),
            message=message,
            data=data,
            errors=errors,
            guardrails=guardrails,
            pipeline_last_stage=pipeline_last_stage,
            created_at=now,
            update_at=now,
        )
    )


class OcrResultsOutcome:
    """The stage's `StageOutcome` that writes `nilam_ocr_results` when the request ends at this stage. `jobs` is
    the stage's jobs table: the pipeline_name_sequence stored in the job's `input` says whether the request ran
    guardrails (no sequence: the full pipeline, so it did)."""

    def __init__(self, table: Table, jobs: Table, *, stage: str):
        self.table = table
        self._jobs = jobs
        self._stage = stage

    async def claimed(self, conn: AsyncConnection, request_id: str) -> None:
        """Nothing: only the end of a request is written."""

    async def completed(self, conn: AsyncConnection, request_id: str, data: dict[str, Any] | None) -> None:
        """200 with `data` (this stage's result as it is, scoring: the contract's fields); nothing when `data` is
        None (not the last stage of the request)."""
        if data is None:
            return
        guardrails = await self._guardrails(conn, request_id, GUARDRAILS_PASSED)
        await write_ocr_result(
            conn,
            self.table,
            request_id,
            200,
            COMPLETED_MESSAGE,
            data=data,
            guardrails=guardrails,
            pipeline_last_stage=self._service(),
        )

    async def failed(
        self, conn: AsyncConnection, request_id: str, error_message: str, *, stage: str | None = None
    ) -> None:
        """422 `<STAGE>_FAILED`, naming `stage` when the hand-off to it failed."""
        guardrails = await self._guardrails(conn, request_id, GUARDRAILS_PASSED)
        await write_ocr_result(
            conn,
            self.table,
            request_id,
            422,
            error_message,
            errors=f"{stage or self._stage}_FAILED",
            guardrails=guardrails,
            pipeline_last_stage=self._service(stage),
        )

    async def rejected(self, conn: AsyncConnection, request_id: str, reason: str) -> None:
        """400 `DOWNSTREAM_VALIDATION_ERROR` with the rules' reason as the message."""
        guardrails = await self._guardrails(conn, request_id, GUARDRAILS_REJECTED)
        await write_ocr_result(
            conn,
            self.table,
            request_id,
            400,
            reason,
            errors=REJECTED_CODE,
            guardrails=guardrails,
            pipeline_last_stage=self._service(),
        )

    def _service(self, stage: str | None = None) -> str:
        """`stage` (default: this one) as pipeline_name_sequence names it: OCR -> extraction, ..."""
        stage = stage or self._stage
        return SERVICE_OF_STAGE.get(stage, stage.lower())

    async def _guardrails(self, conn: AsyncConnection, request_id: str, value: int) -> int | None:
        """`value`, or None when the request's pipeline_name_sequence left guardrails out."""
        input = (await conn.execute(select(self._jobs.c.input).where(self._jobs.c.request_id == request_id))).scalar()
        sequence = stored_sequence(input)
        return value if sequence is None or GUARDRAILS in sequence else None
