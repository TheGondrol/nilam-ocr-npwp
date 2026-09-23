"""Jobs: keep the request input, so a job a dead process left PROCESSING can be run again

Revision ID: 0004_jobs_input
Revises: 0003_outbox_dead_letters
Create Date: 2026-09-22

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0004_jobs_input"
down_revision: str | None = "0003_outbox_dead_letters"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

TABLES = ("ocr_jobs", "structuring_jobs", "scoring_jobs")


def upgrade() -> None:
    for table in TABLES:
        op.add_column(table, sa.Column("input", postgresql.JSONB(none_as_null=True), nullable=True))


def downgrade() -> None:
    for table in TABLES:
        op.drop_column(table, "input")
