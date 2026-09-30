"""Move every table of this repository from `public` to the schema `ocr_pipeline`, rows included

Revision ID: 0010_ocr_pipeline_schema
Revises: 0009_guardrails_results_sequence
Create Date: 2026-09-30

`public` is shared with other teams (their `alembic_version` lives there), so this repository's tables get a
schema of their own. `ALTER TABLE ... SET SCHEMA` moves the table itself: the rows, indexes, constraints and
the sequences of the `id` columns go with it, nothing is copied, and it all happens in the migration's one
transaction. The row count of every moved table is logged.

The version table `ocr_npwp_alembic_version` is moved by `env.py` before any revision runs, because Alembic
reads it before this revision and writes it after.

Running pods address the tables by the new schema only from the image that ships this revision: an older pod
fails its queries between this migration and its replacement (jobs left PROCESSING are picked up again by the
stale-job reaper of the new pods).
"""

import logging
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0010_ocr_pipeline_schema"
down_revision: str | None = "0009_guardrails_results_sequence"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

log = logging.getLogger("alembic.runtime.migration")

SCHEMA = "ocr_pipeline"

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
    op.execute(sa.text(f'CREATE SCHEMA IF NOT EXISTS "{SCHEMA}"'))
    _move("public", SCHEMA)


def downgrade() -> None:
    # The schema stays: env.py keeps the version table in it.
    _move(SCHEMA, "public")
