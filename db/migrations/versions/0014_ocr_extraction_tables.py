"""The extraction stage's tables renamed `nilam_ocr_extraction_jobs` / `nilam_ocr_extraction_result`, and a new
`nilam_ocr_results` for the request's final answer

Revision ID: 0014_ocr_extraction_tables
Revises: 0013_nilam_naming
Create Date: 2026-10-07

At the client's request:

1. `nilam_ocr_jobs` becomes `nilam_ocr_extraction_jobs` and `nilam_ocr_results` becomes `nilam_ocr_extraction_result`
   (the testing twins `nilam_testing_ocr_*` the same way), rows included: `RENAME` changes the catalog only. The
   names derived from the table name follow it as in 0013: the indexes (`idx_nilam_ocr_jobs_status` ->
   `idx_nilam_ocr_extraction_jobs_status`), the primary keys and the foreign key of the results table.
2. The freed name `nilam_ocr_results` (and `nilam_testing_ocr_results`) is a new table: the request's final answer in
   the shape of the extract-ocr answer, one row per request_id, written by the service that ends the request.

Pods on an image that still addresses `nilam_ocr_jobs` fail their queries between this migration and their
replacement, and anything outside this repository that reads these tables (the central orchestrator team's
queries, the dashboards) has to use the new names from then on.

The names are written out rather than taken from `tables.py`, so this revision keeps acting on the names it was
written for.
"""

import logging
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0014_ocr_extraction_tables"
down_revision: str | None = "0013_nilam_naming"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

log = logging.getLogger("alembic.runtime.migration")

SCHEMA = "nilam_ocr_npwp"
LANES = ("nilam_", "nilam_testing_")
RENAMES = {"ocr_jobs": "ocr_extraction_jobs", "ocr_results": "ocr_extraction_result"}
NEW_TABLE = "ocr_results"

# Every object named after the table: its indexes (renaming the index of a primary key renames the constraint), its
# other constraints (the foreign key) and the sequences it owns.
DERIVED_NAMES = sa.text(
    """
    SELECT 'INDEX' AS kind, i.relname AS name
      FROM pg_index x JOIN pg_class i ON i.oid = x.indexrelid
     WHERE x.indrelid = CAST(:table AS regclass)
    UNION ALL
    SELECT 'CONSTRAINT', c.conname
      FROM pg_constraint c
     WHERE c.conrelid = CAST(:table AS regclass) AND c.contype NOT IN ('p', 'u', 'x')
    UNION ALL
    SELECT 'SEQUENCE', s.relname
      FROM pg_depend d JOIN pg_class s ON s.oid = d.objid
     WHERE d.refobjid = CAST(:table AS regclass) AND s.relkind = 'S' AND d.deptype IN ('a', 'i')
    """
)


def _rename(old: str, new: str) -> None:
    """Renames table `old` to `new` in SCHEMA, and `old` to `new` in the names of the objects named after it."""
    conn = op.get_bind()
    if conn.execute(sa.text("SELECT to_regclass(:name)"), {"name": f'"{SCHEMA}"."{new}"'}).scalar():
        raise RuntimeError(f"{SCHEMA}.{new} already exists next to {SCHEMA}.{old}; compare the two by hand first")
    op.execute(sa.text(f'ALTER TABLE "{SCHEMA}"."{old}" RENAME TO "{new}"'))
    for kind, name in conn.execute(DERIVED_NAMES, {"table": f'"{SCHEMA}"."{new}"'}).all():
        if old not in name:
            continue
        renamed = name.replace(old, new, 1)
        if kind == "CONSTRAINT":
            op.execute(sa.text(f'ALTER TABLE "{SCHEMA}"."{new}" RENAME CONSTRAINT "{name}" TO "{renamed}"'))
        else:
            op.execute(sa.text(f'ALTER {kind} "{SCHEMA}"."{name}" RENAME TO "{renamed}"'))
    rows = conn.execute(sa.text(f'SELECT count(*) FROM "{SCHEMA}"."{new}"')).scalar()
    log.info("renamed %s.%s to %s (%s rows)", SCHEMA, old, new, rows)


def upgrade() -> None:
    for lane in LANES:
        for old, new in RENAMES.items():
            _rename(f"{lane}{old}", f"{lane}{new}")
        op.create_table(
            f"{lane}{NEW_TABLE}",
            sa.Column("status_code", sa.Integer(), nullable=False),
            sa.Column("status_desc", sa.Text(), nullable=False),
            sa.Column("message", sa.Text(), nullable=True),
            sa.Column("data", postgresql.JSONB(none_as_null=True), nullable=True),
            sa.Column("errors", sa.Text(), nullable=True),
            sa.Column("request_id", sa.Text(), nullable=False),
            sa.Column("guardrails", sa.Integer(), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
            sa.Column("update_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
            sa.PrimaryKeyConstraint("request_id"),
            schema=SCHEMA,
        )


def downgrade() -> None:
    for lane in LANES:
        op.drop_table(f"{lane}{NEW_TABLE}", schema=SCHEMA)
        for old, new in RENAMES.items():
            _rename(f"{lane}{new}", f"{lane}{old}")
