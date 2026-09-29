"""Keeps every guardrails verdict in `guardrails_results`, the rejected documents included: they never reach a
stage table, so without this a rejection left no trace here. The row also keeps the request's
pipeline_name_sequence, so `GET /v1/extract-ocr/{request_id}` can answer for a request that never reached a
stage (guardrails only, or rejected by guardrails).

Best-effort both ways: a write or a read that fails or takes longer than `timeout` is logged, and the request
is answered as if there were no row, because the entry point must not go down with the audit trail."""

import asyncio
import logging
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any, Protocol, TypedDict

from sqlalchemy import MetaData, select

from ocr_common.pipeline.database import get_engine
from ocr_common.pipeline.tables import guardrails_results_table

logger = logging.getLogger(__name__)

SOURCE_REQUEST = "request"  # the central orchestrator sent the threshold with the request
SOURCE_SERVICE = "service"  # the guardrails service used its own


class GuardrailsVerdict(TypedDict):
    """The last verdict kept for a request_id: the guardrails report and the sequence it was judged for."""

    report: dict[str, Any]
    sequence: list[str] | None


class GuardrailsLog(Protocol):
    async def record(
        self,
        request_id: str,
        report: dict[str, Any],
        *,
        threshold_from_request: bool,
        sequence: Sequence[str] | None = None,
    ) -> None: ...

    async def latest(self, request_id: str) -> GuardrailsVerdict | None: ...


class NoGuardrailsLog:
    """Without DATABASE_URL (local runs, tests): nothing is kept."""

    async def record(
        self,
        request_id: str,
        report: dict[str, Any],
        *,
        threshold_from_request: bool,
        sequence: Sequence[str] | None = None,
    ) -> None:
        return None

    async def latest(self, request_id: str) -> GuardrailsVerdict | None:
        return None


class SqlGuardrailsLog:
    def __init__(self, database_url: str, *, table_prefix: str = "", timeout: float = 2.0):
        self._url = database_url
        self._table = guardrails_results_table(MetaData(), table_prefix)
        self._timeout = timeout

    async def record(
        self,
        request_id: str,
        report: dict[str, Any],
        *,
        threshold_from_request: bool,
        sequence: Sequence[str] | None = None,
    ) -> None:
        try:
            await asyncio.wait_for(self._insert(request_id, report, threshold_from_request, sequence), self._timeout)
        except Exception:  # noqa: BLE001 - best-effort, see the module docstring
            logger.exception("guardrails verdict of %s not recorded", request_id)

    async def latest(self, request_id: str) -> GuardrailsVerdict | None:
        """The last verdict of `request_id` (a request_id sent again is judged again); None when there is none,
        or when it cannot be read."""
        try:
            return await asyncio.wait_for(self._select_latest(request_id), self._timeout)
        except Exception:  # noqa: BLE001 - best-effort, see the module docstring
            logger.exception("guardrails verdict of %s not readable", request_id)
            return None

    async def _insert(
        self, request_id: str, report: dict[str, Any], threshold_from_request: bool, sequence: Sequence[str] | None
    ) -> None:
        document = report.get("document") or {}
        now = datetime.now(UTC)
        async with get_engine(self._url).begin() as conn:
            await conn.execute(
                self._table.insert().values(
                    request_id=request_id,
                    passed=bool(report.get("passed")),
                    verdict=document.get("verdict"),
                    confidence=document.get("confidence"),
                    threshold=document.get("threshold"),
                    threshold_target=document.get("threshold_target"),
                    threshold_source=SOURCE_REQUEST if threshold_from_request else SOURCE_SERVICE,
                    n_pages=document.get("n_pages"),
                    reason=report.get("reason"),
                    pipeline_name_sequence=list(sequence) if sequence else None,
                    report=report,
                    created_at=now,
                    ds=now.strftime("%Y%m%d"),
                )
            )

    async def _select_latest(self, request_id: str) -> GuardrailsVerdict | None:
        table = self._table
        async with get_engine(self._url).connect() as conn:
            row = (
                await conn.execute(
                    select(table.c.report, table.c.pipeline_name_sequence)
                    .where(table.c.request_id == request_id)
                    .order_by(table.c.id.desc())
                    .limit(1)
                )
            ).one_or_none()
        if row is None:
            return None
        sequence = row.pipeline_name_sequence
        return {"report": dict(row.report), "sequence": list(sequence) if isinstance(sequence, list) else None}
