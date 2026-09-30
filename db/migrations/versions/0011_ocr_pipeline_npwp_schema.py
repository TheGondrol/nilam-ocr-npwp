"""Move every table of this repository from the schema `ocr_pipeline` to `ocr_pipeline_npwp`, rows included

Revision ID: 0011_ocr_pipeline_npwp_schema
Revises: 0010_ocr_pipeline_schema
Create Date: 2026-09-30

The schema is named after the document type, like the rest of this repository (`ocr_npwp_alembic_version`,
the release `nilam-ocr-npwp`). 0010 already ran on the dev database, so the new name comes as its own revision:
the same `ALTER TABLE ... SET SCHEMA` as 0010 (rows, indexes, constraints and id sequences move with the table,
nothing is copied, one transaction), then the emptied `ocr_pipeline` is dropped. The drop has no CASCADE: if
something else was put in `ocr_pipeline`, the migration stops instead of taking it along.

The version table is moved by `env.py` before any revision runs, as for 0010. Pods on an image that still
addresses `ocr_pipeline` fail their queries between this migration and their replacement.
"""

import logging
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0011_ocr_pipeline_npwp_schema"
down_revision: str | None = "0010_ocr_pipeline_schema"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

log = logging.getLogger("alembic.runtime.migration")

OLD_SCHEMA = "ocr_pipeline"
NEW_SCHEMA = "ocr_pipeline_npwp"

TABLES = tuple(
    f"{lane}{name}"
    for lane in ("", "testing_")
    for name in (
        "ocr_jobs",
        "ocr_results",
        "structuring_jobs",
        "structuring_results",
        "scoring_jobs",
        "scoring_results",
        "pipeline_outbox",
        "guardrails_results",
    )
)


def _move(source: str, target: str) -> None:
    conn = op.get_bind()
    op.execute(sa.text(f'CREATE SCHEMA IF NOT EXISTS "{target}"'))
    for table in TABLES:
        if conn.execute(sa.text("SELECT to_regclass(:name)"), {"name": f'"{target}"."{table}"'}).scalar():
            raise RuntimeError(
                f"{target}.{table} already exists next to {source}.{table}; compare the two by hand before "
                "running this migration again"
            )
        op.execute(sa.text(f'ALTER TABLE "{source}"."{table}" SET SCHEMA "{target}"'))
        rows = conn.execute(sa.text(f'SELECT count(*) FROM "{target}"."{table}"')).scalar()
        log.info("moved %s.%s to %s (%s rows)", source, table, target, rows)


def upgrade() -> None:
    _move(OLD_SCHEMA, NEW_SCHEMA)
    op.execute(sa.text(f'DROP SCHEMA "{OLD_SCHEMA}"'))


def downgrade() -> None:
    # The new schema stays: env.py keeps the version table in it.
    _move(NEW_SCHEMA, OLD_SCHEMA)
