"""Drop the per-stage schemas (ocr, structuring, scoring) left from before the tables moved to public

Revision ID: 0005_drop_legacy_schemas
Revises: 0004_jobs_input
Create Date: 2026-09-22

The dev database still carried `ocr.jobs` / `ocr.results` (and the same for structuring and scoring)
from the first design. No code reads them any more: every service uses `public.<stage>_jobs` and
`public.<stage>_results`. This migration drops those schemas, but only when each one holds nothing
but the expected `jobs` and `results` tables; anything else in there means somebody put it to a new
use, and the migration stops instead of dropping it. The row counts are logged before the drop.
There is no downgrade: the rows are gone with the schema.
"""

import logging
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0005_drop_legacy_schemas"
down_revision: str | None = "0004_jobs_input"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

log = logging.getLogger("alembic.runtime.migration")

LEGACY_SCHEMAS = ("ocr", "structuring", "scoring")
EXPECTED_TABLES = {"jobs", "results"}


def upgrade() -> None:
    conn = op.get_bind()
    if conn.dialect.name != "postgresql":
        return
    for schema in LEGACY_SCHEMAS:
        if not conn.execute(sa.text("SELECT 1 FROM pg_namespace WHERE nspname = :s"), {"s": schema}).scalar():
            continue
        tables = {
            row[0]
            for row in conn.execute(sa.text("SELECT tablename FROM pg_tables WHERE schemaname = :s"), {"s": schema})
        }
        views = [
            row[0]
            for row in conn.execute(sa.text("SELECT viewname FROM pg_views WHERE schemaname = :s"), {"s": schema})
        ]
        unexpected = sorted(tables - EXPECTED_TABLES) + views
        if unexpected:
            raise RuntimeError(
                f"schema {schema!r} holds objects this migration does not know about ({unexpected}); it only "
                "drops the legacy jobs/results tables, so look at them first and drop the schema by hand"
            )
        counts = {
            table: conn.execute(sa.text(f'SELECT count(*) FROM "{schema}"."{table}"')).scalar() for table in tables
        }
        log.info("dropping legacy schema %s (rows left behind: %s)", schema, counts)
        op.execute(sa.text(f'DROP SCHEMA "{schema}" CASCADE'))


def downgrade() -> None:
    # The legacy schemas and their rows are gone for good; the current tables in `public` are untouched.
    pass
