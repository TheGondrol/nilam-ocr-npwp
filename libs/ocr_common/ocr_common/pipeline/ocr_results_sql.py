"""`nilam_ocr_results`: the log of every answer this pipeline gives the central orchestrator, one append-only row per
answer, in the shape of the extract-ocr answer (`status_code`, `status_desc`, `message`, `data`, `errors`,
`guardrails`, `pipeline_last_stage`). Two writers:

- the orchestrator NPWP, for every answer to `POST /v1/extract-ocr` (200, 202, 400, 422, 5xx), just before it
  answers (`write_ocr_result`, from its response log);
- the stage that ends the request, for its result callback, once it reached the orchestrator
  (`CallbackResultsLog`, behind `callbacks.LoggedCallback`): a callback retried by the outbox is one row, written
  when a try succeeds; one that is never sent (a dead letter, no ORCHESTRATION_URL) is none.

A request answered 202 therefore has the 202 row, then the row of its callback; one answered 200 has the 200
row and, when the stages send callbacks, the callback's. Its newest row is the last thing the orchestrator got.
"""

import asyncio
import logging
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import Table, select
from sqlalchemy.ext.asyncio import AsyncConnection

from ocr_common.npwp import COMPLETED_MESSAGE
from ocr_common.pipeline.callbacks import RESULT_COMPLETED, result_callback_body
from ocr_common.pipeline.database import get_engine
from ocr_common.pipeline.repository import stored_sequence
from ocr_common.pipeline.sequence import GUARDRAILS, SERVICE_OF_STAGE
from ocr_common.web.envelope import STATUS_DESC

logger = logging.getLogger(__name__)

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
    """Append one answer of `request_id` as a new row (the table is append-only; `update_at` = `created_at`)."""
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


def callback_row(body: dict[str, Any]) -> dict[str, Any] | None:
    """The row of a per-stage callback body that ends the request, as the extract-ocr answer of the same outcome:
    completed 200 with the result as `data`, rejected 400 `DOWNSTREAM_VALIDATION_ERROR` (`guardrails` 1), failed
    422 `<STAGE>_FAILED`. `pipeline_last_stage` is the service whose callback it is (on a failed hand-off, the
    stage that never received the job). None for a body that does not end the request."""
    sent = result_callback_body(body)
    if sent is None:
        return None
    stage = str(body["stage"])
    service = SERVICE_OF_STAGE.get(stage, stage.lower())
    if sent["status"] == RESULT_COMPLETED and sent.get("result") is not None:
        status_code, message, errors, guardrails = 200, COMPLETED_MESSAGE, None, GUARDRAILS_PASSED
    elif sent["status"] == RESULT_COMPLETED:
        status_code, message, errors, guardrails = 400, sent.get("message"), sent.get("error_code"), GUARDRAILS_REJECTED
    else:
        status_code, message, errors, guardrails = 422, sent.get("message"), sent.get("error_code"), GUARDRAILS_PASSED
    return {
        "status_code": status_code,
        "message": message,
        "data": sent.get("result") if status_code == 200 else None,
        "errors": errors,
        "guardrails": guardrails,
        "pipeline_last_stage": service,
    }


class CallbackResultsLog:
    """`delivered` of `LoggedCallback` for one stage: appends the row of a result callback the orchestrator
    received. `jobs` is the stage's jobs table: the pipeline_name_sequence in the job's `input` says whether the
    request ran guardrails (no sequence: the full pipeline, so it did), else `guardrails` is null.

    Best-effort, like the other logs: the callback is already delivered, so a write that fails or takes longer
    than `timeout` is logged and does not fail the delivery (which would send the callback again)."""

    def __init__(self, database_url: str, table: Table, jobs: Table, *, timeout: float = 2.0):
        self._url = database_url
        self.table = table
        self._jobs = jobs
        self._timeout = timeout

    async def __call__(self, body: dict[str, Any]) -> None:
        row = callback_row(body)
        if row is None:
            return
        request_id = str(body["request_id"])
        try:
            await asyncio.wait_for(self._insert(request_id, row), self._timeout)
        except Exception:  # noqa: BLE001 - best-effort, see the class docstring
            logger.exception("nilam_ocr_results: callback answer of %s not recorded", request_id)

    async def _insert(self, request_id: str, row: dict[str, Any]) -> None:
        async with get_engine(self._url).begin() as conn:
            if row["guardrails"] is not None and not await self._ran_guardrails(conn, request_id):
                row = {**row, "guardrails": None}
            await write_ocr_result(conn, self.table, request_id, row.pop("status_code"), row.pop("message"), **row)

    async def _ran_guardrails(self, conn: AsyncConnection, request_id: str) -> bool:
        jobs = self._jobs
        input = (await conn.execute(select(jobs.c.input).where(jobs.c.request_id == request_id))).scalar()
        sequence = stored_sequence(input)
        return sequence is None or GUARDRAILS in sequence
