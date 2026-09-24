"""`StageResults` on the shared database: the only part of hand-offs by reference that needs SQLAlchemy."""

from typing import Any

from sqlalchemy import MetaData, Table, select

from ocr_common.pipeline.database import get_engine
from ocr_common.pipeline.tables import pipeline_tables


class SqlStageResults:
    """`StageResults` on the shared database."""

    def __init__(self, database_url: str, table_prefix: str = ""):
        """`table_prefix` is put before every stage's prefix: `testing_` reads `testing_ocr_results`, ..."""
        self._url = database_url
        self._table_prefix = table_prefix
        self._tables: dict[str, Table] = {}

    def _results(self, stage_prefix: str) -> Table:
        if stage_prefix not in self._tables:
            _, results = pipeline_tables(f"{self._table_prefix}{stage_prefix}", MetaData())
            self._tables[stage_prefix] = results
        return self._tables[stage_prefix]

    async def get(self, stage_prefix: str, request_id: str) -> dict[str, Any] | None:
        """See `StageResults.get`."""
        results = self._results(stage_prefix)
        async with get_engine(self._url).connect() as conn:
            row = (await conn.execute(select(results.c.result).where(results.c.request_id == request_id))).one_or_none()
        return None if row is None else row.result
