"""`pipeline_last_stage` on `nilam_ocr_results`: the service that ended the request

Revision ID: 0015_ocr_results_last_stage
Revises: 0014_ocr_extraction_tables
Create Date: 2026-10-07

At the client's request. The service as pipeline_name_sequence names it (`guardrails`, `extraction`, `structuring`,
`scoring`), on every final answer, the success included; for a failed hand-off the stage that never received the
job. Nullable: the rows written before this revision keep null. Adding a nullable column without a default changes
the catalog only, and code without the column keeps working, so this can run before the deploy that fills it.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0015_ocr_results_last_stage"
down_revision: str | None = "0014_ocr_extraction_tables"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

SCHEMA = "nilam_ocr_npwp"
TABLES = ("nilam_ocr_results", "nilam_testing_ocr_results")


def upgrade() -> None:
    for table in TABLES:
        op.add_column(table, sa.Column("pipeline_last_stage", sa.Text(), nullable=True), schema=SCHEMA)


def downgrade() -> None:
    for table in TABLES:
        op.drop_column(table, "pipeline_last_stage", schema=SCHEMA)
