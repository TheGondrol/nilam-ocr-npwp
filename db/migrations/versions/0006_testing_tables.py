"""Testing tables: copies of the stage tables and the outbox for the `-test` endpoints

Revision ID: 0006_testing_tables
Revises: 0005_drop_legacy_schemas
Create Date: 2026-09-24

The ML team load-tests the pipeline through `/v1/extract-ocr-test` (`TESTING_ENDPOINTS`). Those requests
run the same code as the live ones but store everything in these `testing_*` tables, so a burst never
mixes with the orchestrator's data. Same columns and indexes as `<stage>_jobs`, `<stage>_results` and
`pipeline_outbox` as they stand after 0004. They hold nothing of value: empty them after a test run with
`TRUNCATE testing_ocr_jobs, testing_ocr_results, ..., testing_pipeline_outbox`.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0006_testing_tables"
down_revision: str | None = "0005_drop_legacy_schemas"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

STAGE_PREFIXES = ("testing_ocr", "testing_structuring", "testing_scoring")
OUTBOX = "testing_pipeline_outbox"


def _timestamps() -> list[sa.Column]:
    return [
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("ds", sa.Text(), nullable=False),
    ]


def upgrade() -> None:
    for prefix in STAGE_PREFIXES:
        op.create_table(
            f"{prefix}_jobs",
            sa.Column("request_id", sa.Text(), nullable=False),
            sa.Column("status", sa.Text(), nullable=False),
            sa.Column("error_message", sa.Text(), nullable=True),
            sa.Column("attempts", sa.Integer(), server_default=sa.text("1"), nullable=False),
            sa.Column("input", postgresql.JSONB(none_as_null=True), nullable=True),
            *_timestamps(),
            sa.PrimaryKeyConstraint("request_id"),
        )
        op.create_index(f"idx_{prefix}_jobs_status", f"{prefix}_jobs", ["status"])
        op.create_index(f"idx_{prefix}_jobs_ds", f"{prefix}_jobs", ["ds"])
        op.create_table(
            f"{prefix}_results",
            sa.Column("request_id", sa.Text(), nullable=False),
            sa.Column("result", postgresql.JSONB(none_as_null=True), nullable=False),
            *_timestamps(),
            sa.ForeignKeyConstraint(["request_id"], [f"{prefix}_jobs.request_id"]),
            sa.PrimaryKeyConstraint("request_id"),
        )
        op.create_index(f"idx_{prefix}_results_ds", f"{prefix}_results", ["ds"])

    op.create_table(
        OUTBOX,
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("request_id", sa.Text(), nullable=False),
        sa.Column("stage", sa.Text(), nullable=False),
        sa.Column("kind", sa.Text(), nullable=False),
        sa.Column("payload", postgresql.JSONB(none_as_null=True), nullable=False),
        sa.Column("attempts", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("next_attempt_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("failed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_error", sa.Text(), nullable=True),
        *_timestamps(),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        f"idx_{OUTBOX}_due",
        OUTBOX,
        ["stage", "next_attempt_at", "id"],
        postgresql_where=sa.text("failed_at IS NULL"),
    )
    op.create_index(f"idx_{OUTBOX}_dead", OUTBOX, ["stage"], postgresql_where=sa.text("failed_at IS NOT NULL"))
    op.create_index(f"idx_{OUTBOX}_request_id", OUTBOX, ["request_id"])


def downgrade() -> None:
    op.drop_table(OUTBOX)
    for prefix in reversed(STAGE_PREFIXES):
        op.drop_table(f"{prefix}_results")
        op.drop_table(f"{prefix}_jobs")
