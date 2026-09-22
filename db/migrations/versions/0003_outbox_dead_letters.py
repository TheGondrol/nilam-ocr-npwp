"""Outbox: keep the messages that were given up on as dead letters

Revision ID: 0003_outbox_dead_letters
Revises: 0002_pipeline_outbox
Create Date: 2026-09-22

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0003_outbox_dead_letters"
down_revision: str | None = "0002_pipeline_outbox"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("pipeline_outbox", sa.Column("failed_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("pipeline_outbox", sa.Column("last_error", sa.Text(), nullable=True))
    op.drop_index("idx_pipeline_outbox_due", table_name="pipeline_outbox")
    op.create_index(
        "idx_pipeline_outbox_due",
        "pipeline_outbox",
        ["stage", "next_attempt_at", "id"],
        postgresql_where=sa.text("failed_at IS NULL"),
    )
    op.create_index(
        "idx_pipeline_outbox_dead",
        "pipeline_outbox",
        ["stage"],
        postgresql_where=sa.text("failed_at IS NOT NULL"),
    )


def downgrade() -> None:
    op.drop_index("idx_pipeline_outbox_dead", table_name="pipeline_outbox")
    op.drop_index("idx_pipeline_outbox_due", table_name="pipeline_outbox")
    op.create_index("idx_pipeline_outbox_due", "pipeline_outbox", ["next_attempt_at", "id"])
    op.drop_column("pipeline_outbox", "last_error")
    op.drop_column("pipeline_outbox", "failed_at")
