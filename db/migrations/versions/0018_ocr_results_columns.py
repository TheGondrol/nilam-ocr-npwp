"""`nilam_ocr_results` without `update_at`, its columns in the client's order

Revision ID: 0018_ocr_results_columns
Revises: 0017_ocr_extraction_results
Create Date: 2026-10-07

The table is append-only, so `update_at` only ever repeated `created_at`; the client dropped it and asked for the
columns in this order: id, request_id, status_code, status_desc, message, data, errors, pipeline_last_stage,
guardrails, created_at. PostgreSQL cannot reorder columns, so each table (and its testing twin) is rebuilt:
a new table in that order, every row copied with its id (the id sequence goes on after them), the old table dropped
and the new one renamed, then the `request_id` index and the append-only trigger are created again. All in the
migration's one transaction.

Pods on an image that still writes `update_at` fail their inserts between this migration and their replacement:
stop the services before it, deploy right after. The downgrade adds `update_at` back (= `created_at`) at the end of
the table; the order is not restored.
"""

import logging
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0018_ocr_results_columns"
down_revision: str | None = "0017_ocr_extraction_results"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

log = logging.getLogger("alembic.runtime.migration")

SCHEMA = "nilam_ocr_npwp"
TABLES = ("nilam_ocr_results", "nilam_testing_ocr_results")
FUNCTION = "nilam_append_only"
COLUMNS = "id, request_id, status_code, status_desc, message, data, errors, pipeline_last_stage, guardrails, created_at"


def _trigger(table: str) -> str:
    return (
        f'CREATE TRIGGER "{table}_append_only" BEFORE UPDATE OR DELETE ON "{SCHEMA}"."{table}" '
        f'FOR EACH ROW EXECUTE FUNCTION "{SCHEMA}"."{FUNCTION}"()'
    )


def upgrade() -> None:
    conn = op.get_bind()
    for table in TABLES:
        new = f"{table}_new"
        op.execute(
            sa.text(
                f'CREATE TABLE "{SCHEMA}"."{new}" ('
                "id BIGSERIAL PRIMARY KEY, "
                "request_id TEXT NOT NULL, "
                "status_code INTEGER NOT NULL, "
                "status_desc TEXT NOT NULL, "
                "message TEXT, "
                "data JSONB, "
                "errors TEXT, "
                "pipeline_last_stage TEXT, "
                "guardrails INTEGER, "
                "created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now())"
            )
        )
        op.execute(
            sa.text(
                f'INSERT INTO "{SCHEMA}"."{new}" ({COLUMNS}) ' f'SELECT {COLUMNS} FROM "{SCHEMA}"."{table}" ORDER BY id'
            )
        )
        op.execute(
            sa.text(
                f"SELECT setval(pg_get_serial_sequence('\"{SCHEMA}\".\"{new}\"', 'id'), "
                f'COALESCE(MAX(id), 1), MAX(id) IS NOT NULL) FROM "{SCHEMA}"."{new}"'
            )
        )
        rows = conn.execute(sa.text(f'SELECT count(*) FROM "{SCHEMA}"."{new}"')).scalar()
        op.execute(sa.text(f'DROP TABLE "{SCHEMA}"."{table}"'))  # its trigger, index and id sequence with it
        op.execute(sa.text(f'ALTER TABLE "{SCHEMA}"."{new}" RENAME TO "{table}"'))
        op.execute(sa.text(f'ALTER SEQUENCE "{SCHEMA}"."{new}_id_seq" RENAME TO "{table}_id_seq"'))
        op.execute(sa.text(f'ALTER INDEX "{SCHEMA}"."{new}_pkey" RENAME TO "{table}_pkey"'))
        op.create_index(f"idx_{table}_request_id", table, ["request_id"], schema=SCHEMA)
        op.execute(sa.text(_trigger(table)))
        log.info("rebuilt %s.%s without update_at (%s rows)", SCHEMA, table, rows)


def downgrade() -> None:
    for table in TABLES:
        qualified = f'"{SCHEMA}"."{table}"'
        op.execute(sa.text(f"ALTER TABLE {qualified} ADD COLUMN update_at TIMESTAMP WITH TIME ZONE"))
        # The append-only trigger refuses the UPDATE that fills it.
        op.execute(sa.text(f'ALTER TABLE {qualified} DISABLE TRIGGER "{table}_append_only"'))
        op.execute(sa.text(f"UPDATE {qualified} SET update_at = created_at"))
        op.execute(sa.text(f'ALTER TABLE {qualified} ENABLE TRIGGER "{table}_append_only"'))
        op.execute(sa.text(f"ALTER TABLE {qualified} ALTER COLUMN update_at SET DEFAULT now()"))
        op.execute(sa.text(f"ALTER TABLE {qualified} ALTER COLUMN update_at SET NOT NULL"))
