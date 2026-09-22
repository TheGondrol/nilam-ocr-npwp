"""Baseline: the tables that already existed before migrations

Revision ID: 0001_baseline
Revises:
Create Date: 2026-09-22

"""

from collections.abc import Sequence

from alembic import op

from ocr_common.tables import PIPELINE_TABLE_PREFIXES

revision: str = "0001_baseline"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

STAGE_STATEMENTS = (
    """
    CREATE TABLE IF NOT EXISTS {prefix}_jobs (
        request_id      TEXT PRIMARY KEY,
        status          TEXT NOT NULL,
        error_message   TEXT,
        attempts        INT NOT NULL DEFAULT 1,
        created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
        updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
        ds              TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_{prefix}_jobs_status ON {prefix}_jobs (status)",
    "CREATE INDEX IF NOT EXISTS idx_{prefix}_jobs_ds ON {prefix}_jobs (ds)",
    """
    CREATE TABLE IF NOT EXISTS {prefix}_results (
        request_id      TEXT PRIMARY KEY REFERENCES {prefix}_jobs (request_id),
        result          JSONB NOT NULL,
        created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
        updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
        ds              TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_{prefix}_results_ds ON {prefix}_results (ds)",
)

LEGACY_STATEMENTS = (
    """
    CREATE TABLE IF NOT EXISTS ocr_npwp_requests (
        request_id      TEXT PRIMARY KEY,
        status          TEXT NOT NULL,
        result          JSONB,
        guardrails      DOUBLE PRECISION,
        error_message   TEXT,
        file_name       TEXT,
        file_size_bytes INT,
        created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
        updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
        ds              TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_ocr_npwp_requests_status ON ocr_npwp_requests (status)",
    "CREATE INDEX IF NOT EXISTS idx_ocr_npwp_requests_created_at ON ocr_npwp_requests (created_at)",
    "CREATE INDEX IF NOT EXISTS idx_ocr_npwp_requests_ds ON ocr_npwp_requests (ds)",
)


def upgrade() -> None:
    for prefix in PIPELINE_TABLE_PREFIXES:
        for statement in STAGE_STATEMENTS:
            op.execute(statement.format(prefix=prefix))
    for statement in LEGACY_STATEMENTS:
        op.execute(statement)


def downgrade() -> None:
    raise NotImplementedError(
        "the baseline is not reversible: it adopts tables that already held data. Drop them by hand if you mean it"
    )
