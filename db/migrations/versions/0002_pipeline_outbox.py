"""Outbox for the stage callbacks and hand-offs

Revision ID: 0002_pipeline_outbox
Revises: 0001_baseline
Create Date: 2026-09-22

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0002_pipeline_outbox"
down_revision: str | None = "0001_baseline"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "pipeline_outbox",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("request_id", sa.Text(), nullable=False),
        sa.Column("stage", sa.Text(), nullable=False),
        sa.Column("kind", sa.Text(), nullable=False),
        sa.Column("payload", postgresql.JSONB(none_as_null=True), nullable=False),
        sa.Column("attempts", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("next_attempt_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("ds", sa.Text(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_pipeline_outbox_due", "pipeline_outbox", ["next_attempt_at", "id"])
    op.create_index("idx_pipeline_outbox_request_id", "pipeline_outbox", ["request_id"])


def downgrade() -> None:
    op.drop_index("idx_pipeline_outbox_request_id", table_name="pipeline_outbox")
    op.drop_index("idx_pipeline_outbox_due", table_name="pipeline_outbox")
    op.drop_table("pipeline_outbox")
