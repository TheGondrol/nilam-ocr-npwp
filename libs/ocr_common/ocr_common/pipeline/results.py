"""Reading what an earlier stage stored, for hand-offs by reference.

With `PIPELINE_HANDOFF_BY_REFERENCE` the sender leaves the big parts (OCR blocks, structured fields)
out of the hand-off body and the outbox row; the receiving stage reads them from `<prefix>_results`
of the shared database instead. The payload then only carries `request_id`, `document_type` and the
guardrails report.

The SQLAlchemy implementation lives in `results_sql` so that this module, which the package exports,
stays importable by a service without a database (guardrails)."""

from typing import Any, Protocol

from ocr_common.errors import InternalError


class StageResults(Protocol):
    """Reader of an earlier stage's stored result."""

    async def get(self, stage_prefix: str, request_id: str) -> dict[str, Any] | None:
        """The `result` an earlier stage stored for this request, or None when there is none (yet)."""
        ...


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
