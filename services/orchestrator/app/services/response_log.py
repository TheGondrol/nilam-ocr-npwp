"""Logs every answer to `POST /v1/extract-ocr` in `nilam_ocr_results` (`-test`: `nilam_testing_ocr_results`), just
before it is sent: 200, 202, 400, 422, 5xx, the refusals before anything runs included. With the result callbacks
the stages log when they deliver them, the table holds every answer the central orchestrator got.

Best-effort, like the guardrails log: a write that fails or takes longer than `timeout` is logged, and the answer
goes out anyway, because the entry point must not go down with the log."""

import asyncio
import logging
from typing import Any, Protocol

from sqlalchemy import MetaData

from ocr_common.pipeline.database import get_engine
from ocr_common.pipeline.ocr_results_sql import write_ocr_result
from ocr_common.pipeline.tables import ocr_results_table

logger = logging.getLogger(__name__)


class ResponseLog(Protocol):
    async def record(self, request_id: str, status_code: int, body: dict[str, Any], *, guardrails: int | None) -> None:
        """Append the answer `body` (the extract-ocr envelope) sent with HTTP `status_code`."""
        ...


class NoResponseLog:
    """Without DATABASE_URL (local runs, tests): nothing is kept."""

    async def record(self, request_id: str, status_code: int, body: dict[str, Any], *, guardrails: int | None) -> None:
        return None


class SqlResponseLog:
    def __init__(self, database_url: str, *, table_prefix: str = "", timeout: float = 2.0):
        self._url = database_url
        self.table = ocr_results_table(MetaData(), table_prefix)
        self._timeout = timeout

    async def record(self, request_id: str, status_code: int, body: dict[str, Any], *, guardrails: int | None) -> None:
        try:
            await asyncio.wait_for(self._insert(request_id, status_code, body, guardrails), self._timeout)
        except Exception:  # noqa: BLE001 - best-effort, see the module docstring
            logger.exception("nilam_ocr_results: answer %d to %s not recorded", status_code, request_id)

    async def _insert(self, request_id: str, status_code: int, body: dict[str, Any], guardrails: int | None) -> None:
        async with get_engine(self._url).begin() as conn:
            await write_ocr_result(
                conn,
                self.table,
                request_id,
                status_code,
                body.get("message"),
                data=body.get("data"),
                errors=body.get("errors"),
                guardrails=guardrails,
                pipeline_last_stage=body.get("pipeline_last_stage"),
            )
