"""`nilam_ocr_results` append-only: an `id` key instead of `request_id`, and a trigger that refuses UPDATE / DELETE

Revision ID: 0016_ocr_results_append_only
Revises: 0015_ocr_results_last_stage
Create Date: 2026-10-07

At the client's request every final state is kept: a request_id run again (FAILED, then DONE after the re-run)
gets another row instead of overwriting the one it had. So, on `nilam_ocr_results` and `nilam_testing_ocr_results`:

1. a `bigserial` column `id` becomes the primary key (the rows already there are numbered in their own order),
   `request_id` loses it and gets an index (`idx_<table>_request_id`);
2. the function `nilam_append_only()` and a trigger `<table>_append_only` refuse every UPDATE and DELETE. TRUNCATE
   is left possible: copy_database --replace empties the target that way when moving to another database.

Deploy the code that only INSERTs (and does not ask for the id back) before this revision: the earlier code
upserts on `request_id`, which needs the key this revision removes. The downgrade keeps the newest row of each
request_id and gives `request_id` its key back.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0016_ocr_results_append_only"
down_revision: str | None = "0015_ocr_results_last_stage"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

SCHEMA = "nilam_ocr_npwp"
TABLES = ("nilam_ocr_results", "nilam_testing_ocr_results")
FUNCTION = "nilam_append_only"


def upgrade() -> None:
    op.execute(
        sa.text(
            f'CREATE OR REPLACE FUNCTION "{SCHEMA}"."{FUNCTION}"() RETURNS trigger LANGUAGE plpgsql AS $$ '
            "BEGIN RAISE EXCEPTION '% is append-only: % is not allowed', TG_TABLE_NAME, TG_OP; END $$"
        )
    )
    for table in TABLES:
        qualified = f'"{SCHEMA}"."{table}"'
        op.execute(sa.text(f"ALTER TABLE {qualified} ADD COLUMN id BIGSERIAL"))
        op.execute(sa.text(f'ALTER TABLE {qualified} DROP CONSTRAINT "{table}_pkey"'))
        op.execute(sa.text(f'ALTER TABLE {qualified} ADD CONSTRAINT "{table}_pkey" PRIMARY KEY (id)'))
        op.create_index(f"idx_{table}_request_id", table, ["request_id"], schema=SCHEMA)
        op.execute(
            sa.text(
                f'CREATE TRIGGER "{table}_append_only" BEFORE UPDATE OR DELETE ON {qualified} '
                f'FOR EACH ROW EXECUTE FUNCTION "{SCHEMA}"."{FUNCTION}"()'
            )
        )


def downgrade() -> None:
    for table in TABLES:
        qualified = f'"{SCHEMA}"."{table}"'
        op.execute(sa.text(f'DROP TRIGGER "{table}_append_only" ON {qualified}'))
        # Back to one row per request_id: its newest one.
        op.execute(
            sa.text(
                f"DELETE FROM {qualified} t USING {qualified} newer "
                "WHERE newer.request_id = t.request_id AND newer.id > t.id"
            )
        )
        op.drop_index(f"idx_{table}_request_id", table_name=table, schema=SCHEMA)
        op.execute(sa.text(f'ALTER TABLE {qualified} DROP CONSTRAINT "{table}_pkey"'))
        op.execute(sa.text(f"ALTER TABLE {qualified} DROP COLUMN id"))
        op.execute(sa.text(f'ALTER TABLE {qualified} ADD CONSTRAINT "{table}_pkey" PRIMARY KEY (request_id)'))
    op.execute(sa.text(f'DROP FUNCTION "{SCHEMA}"."{FUNCTION}"()'))
