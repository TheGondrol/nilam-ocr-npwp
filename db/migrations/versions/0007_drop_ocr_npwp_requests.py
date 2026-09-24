"""Drop ocr_npwp_requests, the table of the removed legacy synchronous contract

Revision ID: 0007_drop_ocr_npwp_requests
Revises: 0006_testing_tables
Create Date: 2026-09-24

Only ekstraksi's legacy contract (`generate-request-id` -> `extract-ocr` -> `get-ocr-result`) wrote this
table, and that contract has been removed; no service reads or writes it any more. The row count is
logged before the drop so a non-empty table leaves a trace in the migration job's log. `downgrade`
recreates the empty table as 0001 created it; the rows themselves are not restored.
"""

import logging
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0007_drop_ocr_npwp_requests"
down_revision: str | None = "0006_testing_tables"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

log = logging.getLogger("alembic.runtime.migration")

TABLE = "ocr_npwp_requests"


def upgrade() -> None:
    conn = op.get_bind()
    if not sa.inspect(conn).has_table(TABLE):
        return
    rows = conn.execute(sa.text(f"SELECT count(*) FROM {TABLE}")).scalar()
    log.info("dropping %s (rows left behind: %s)", TABLE, rows)
    op.drop_table(TABLE)


def downgrade() -> None:
    op.create_table(
        TABLE,
        sa.Column("request_id", sa.Text(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("result", postgresql.JSONB(none_as_null=True), nullable=True),
        sa.Column("guardrails", sa.Double(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("file_name", sa.Text(), nullable=True),
        sa.Column("file_size_bytes", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("ds", sa.Text(), nullable=False),
        sa.PrimaryKeyConstraint("request_id"),
    )
    op.create_index("idx_ocr_npwp_requests_status", TABLE, ["status"])
    op.create_index("idx_ocr_npwp_requests_created_at", TABLE, ["created_at"])
    op.create_index("idx_ocr_npwp_requests_ds", TABLE, ["ds"])
