"""Keeps every guardrails verdict in `guardrails_results`, the rejected documents included: they never reach a
stage table, so without this a rejection left no trace here. Best-effort: a write that fails or takes longer
than `timeout` is logged and dropped, and the request is answered as if nothing happened, because the entry
point must not go down with the audit trail."""

import asyncio
import logging
from datetime import UTC, datetime
from typing import Any, Protocol

from sqlalchemy import MetaData

from ocr_common.pipeline.database import get_engine
from ocr_common.pipeline.tables import guardrails_results_table

logger = logging.getLogger(__name__)

SOURCE_REQUEST = "request"  # the central orchestrator sent the threshold with the request
SOURCE_SERVICE = "service"  # the guardrails service used its own


class GuardrailsLog(Protocol):
    async def record(self, request_id: str, report: dict[str, Any], *, threshold_from_request: bool) -> None: ...


class NoGuardrailsLog:
    """Without DATABASE_URL (local runs, tests): nothing is kept."""

    async def record(self, request_id: str, report: dict[str, Any], *, threshold_from_request: bool) -> None:
        return None


class SqlGuardrailsLog:
    def __init__(self, database_url: str, *, table_prefix: str = "", timeout: float = 2.0):
        self._url = database_url
        self._table = guardrails_results_table(MetaData(), table_prefix)
        self._timeout = timeout

    async def record(self, request_id: str, report: dict[str, Any], *, threshold_from_request: bool) -> None:
        try:
            await asyncio.wait_for(self._insert(request_id, report, threshold_from_request), self._timeout)
        except Exception:  # noqa: BLE001 - best-effort, see the module docstring
            logger.exception("guardrails verdict of %s not recorded", request_id)

    async def _insert(self, request_id: str, report: dict[str, Any], threshold_from_request: bool) -> None:
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
                    report=report,
                    created_at=now,
                    ds=now.strftime("%Y%m%d"),
                )
            )
