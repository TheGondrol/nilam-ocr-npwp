"""guardrails_results: every guardrails verdict, the rejected documents included

Revision ID: 0008_guardrails_results
Revises: 0007_drop_ocr_npwp_requests
Create Date: 2026-09-29

The orchestrator NPWP writes one row per guardrails check: passed or not, the document confidence, the
threshold that decided and whose it was (the central orchestrator's, sent with the request, or the
guardrails service's own), and the full report with the per-page probabilities. Before this, a document
rejected by guardrails left no trace in this database. Append-only: sending the same request_id again
judges it again. `testing_guardrails_results` is the same table for the `-test` endpoints.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0008_guardrails_results"
down_revision: str | None = "0007_drop_ocr_npwp_requests"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

TABLES = ("guardrails_results", "testing_guardrails_results")


def upgrade() -> None:
    for name in TABLES:
        op.create_table(
            name,
            sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
            sa.Column("request_id", sa.Text(), nullable=False),
            sa.Column("passed", sa.Boolean(), nullable=False),
            sa.Column("verdict", sa.Text(), nullable=True),
            sa.Column("confidence", sa.Float(), nullable=True),
            sa.Column("threshold", sa.Float(), nullable=True),
            sa.Column("threshold_target", sa.Text(), nullable=True),
            sa.Column("threshold_source", sa.Text(), nullable=False),
            sa.Column("n_pages", sa.Integer(), nullable=True),
            sa.Column("reason", sa.Text(), nullable=True),
            sa.Column("report", postgresql.JSONB(none_as_null=True), nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
            sa.Column("ds", sa.Text(), nullable=False),
            sa.PrimaryKeyConstraint("id"),
        )
        op.create_index(f"idx_{name}_request_id", name, ["request_id"])
        op.create_index(f"idx_{name}_ds", name, ["ds"])


def downgrade() -> None:
    for name in reversed(TABLES):
        op.drop_index(f"idx_{name}_ds", table_name=name)
        op.drop_index(f"idx_{name}_request_id", table_name=name)
        op.drop_table(name)
