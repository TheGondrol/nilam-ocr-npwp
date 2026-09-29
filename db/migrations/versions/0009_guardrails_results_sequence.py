"""guardrails_results.pipeline_name_sequence: the sequence the verdict was given for

Revision ID: 0009_guardrails_results_sequence
Revises: 0008_guardrails_results
Create Date: 2026-09-29

`GET /v1/extract-ocr/{request_id}` answers a request that never reached a stage (guardrails only, or rejected
by guardrails) from its last guardrails verdict, and needs the request's pipeline_name_sequence to tell a
guardrails-only request (200 with the report) from one that passed but whose hand-off to extraction failed
(404). The column was first added by editing 0008 after 0008 had already run on the dev database, so dev never
got it; it lives here instead. `IF NOT EXISTS`: a database that ran the edited 0008 (local ones) already has it.
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0009_guardrails_results_sequence"
down_revision: str | None = "0008_guardrails_results"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

TABLES = ("guardrails_results", "testing_guardrails_results")


def upgrade() -> None:
    for name in TABLES:
        op.execute(f"ALTER TABLE {name} ADD COLUMN IF NOT EXISTS pipeline_name_sequence JSONB")


def downgrade() -> None:
    for name in TABLES:
        op.execute(f"ALTER TABLE {name} DROP COLUMN IF EXISTS pipeline_name_sequence")
