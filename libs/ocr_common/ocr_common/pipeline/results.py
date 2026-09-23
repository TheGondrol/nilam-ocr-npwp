"""Reading what an earlier stage stored, for hand-offs by reference.

With `PIPELINE_HANDOFF_BY_REFERENCE` the sender leaves the big parts (OCR blocks, structured fields)
out of the hand-off body and the outbox row; the receiving stage reads them from `<prefix>_results`
of the shared database instead. The payload then only carries `request_id`, `document_type` and the
guardrails report."""

from typing import Any, Protocol

from sqlalchemy import MetaData, Table, select

from ocr_common.errors import InternalError
from ocr_common.pipeline.database import get_engine
from ocr_common.pipeline.tables import pipeline_tables


class StageResults(Protocol):
    """Reader of an earlier stage's stored result."""

    async def get(self, stage_prefix: str, request_id: str) -> dict[str, Any] | None:
        """The `result` an earlier stage stored for this request, or None when there is none (yet)."""
        ...


class SqlStageResults:
    """`StageResults` on the shared database."""

    def __init__(self, database_url: str):
        self._url = database_url
        self._tables: dict[str, Table] = {}

    def _results(self, stage_prefix: str) -> Table:
        if stage_prefix not in self._tables:
            _, results = pipeline_tables(stage_prefix, MetaData())
            self._tables[stage_prefix] = results
        return self._tables[stage_prefix]

    async def get(self, stage_prefix: str, request_id: str) -> dict[str, Any] | None:
        """See `StageResults.get`."""
        results = self._results(stage_prefix)
        async with get_engine(self._url).connect() as conn:
            row = (await conn.execute(select(results.c.result).where(results.c.request_id == request_id))).one_or_none()
        return None if row is None else row.result


async def load_upstream(results: StageResults | None, stage_prefix: str, request_id: str) -> dict[str, Any]:
    """The stored result of an earlier stage, when the hand-off referred to it instead of carrying it.
    Fails the job with a clear message when it cannot be read: a hand-off by reference only works when
    both services share the database."""
    if results is None:
        raise InternalError(
            f"the hand-off referred to the {stage_prefix} result of {request_id} but this service has no "
            "DATABASE_URL to read it from (PIPELINE_HANDOFF_BY_REFERENCE needs a shared database)",
        )
    stored = await results.get(stage_prefix, request_id)
    if stored is None:
        raise InternalError(f"no {stage_prefix} result stored for {request_id}; the hand-off referred to it")
    return stored
