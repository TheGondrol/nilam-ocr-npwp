"""guardrails_results: drop threshold_target, the guardrails threshold has a single side now

Revision ID: 0012_drop_threshold_target
Revises: 0011_ocr_pipeline_npwp_schema
Create Date: 2026-10-01

The guardrails threshold used to name the side of the model's answer it applied to (`accept` or `reject`), and
the verdict recorded it in `threshold_target`. There is one threshold now, on the accepted probability (a page is
accepted when `proba_approve >= threshold`), so the column says nothing. Rows already written lose the value; the
`threshold` column stays, and `report` keeps the old report as it was written.

The downgrade brings the column back empty.

The revision id stays within Alembic's `version_num` (varchar 32), and the schema is written out rather than taken
from `PIPELINE_SCHEMA`, so this revision keeps acting on the schema it was written for.
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0012_drop_threshold_target"
down_revision: str | None = "0011_ocr_pipeline_npwp_schema"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

SCHEMA = "ocr_pipeline_npwp"
TABLES = ("guardrails_results", "testing_guardrails_results")


def upgrade() -> None:
    for name in TABLES:
        op.execute(f'ALTER TABLE "{SCHEMA}"."{name}" DROP COLUMN IF EXISTS threshold_target')


def downgrade() -> None:
    for name in TABLES:
        op.execute(f'ALTER TABLE "{SCHEMA}"."{name}" ADD COLUMN IF NOT EXISTS threshold_target TEXT')
