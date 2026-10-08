"""`nilam_ocr_results.guardrails` a float: the accepted probability when the request sent no guardrails threshold

Revision ID: 0019_results_guardrails_float
Revises: 0018_ocr_results_columns
Create Date: 2026-10-08

Agreed with the central orchestrator (8 Oct 2026): a request without `guardrails_confidence_threshold` is accepted
whatever the guardrails model says, and its answer's `guardrails` is the model's accepted probability (a float)
instead of 0 / 1. The column (and its testing twin) goes from INTEGER to DOUBLE PRECISION; the 0 / 1 already in it
stay as they are. ALTER TABLE does not fire the append-only trigger (a row trigger on UPDATE / DELETE).

The downgrade rounds back to INTEGER, which turns every probability into 0 or 1.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0019_results_guardrails_float"
down_revision: str | None = "0018_ocr_results_columns"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

SCHEMA = "nilam_ocr_npwp"
TABLES = ("nilam_ocr_results", "nilam_testing_ocr_results")


def upgrade() -> None:
    for table in TABLES:
        op.execute(
            sa.text(
                f'ALTER TABLE "{SCHEMA}"."{table}" ALTER COLUMN guardrails TYPE DOUBLE PRECISION '
                "USING guardrails::double precision"
            )
        )


def downgrade() -> None:
    for table in TABLES:
        op.execute(
            sa.text(
                f'ALTER TABLE "{SCHEMA}"."{table}" ALTER COLUMN guardrails TYPE INTEGER '
                "USING round(guardrails)::integer"
            )
        )
