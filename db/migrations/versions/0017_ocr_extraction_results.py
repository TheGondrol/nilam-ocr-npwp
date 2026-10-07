"""`nilam_ocr_extraction_result` renamed `nilam_ocr_extraction_results` (plural, like the other stages' result tables)

Revision ID: 0017_ocr_extraction_results
Revises: 0016_ocr_results_append_only
Create Date: 2026-10-07

0014 named the extraction stage's result table `nilam_ocr_extraction_result`; the client's name is the plural. The
table (and its testing twin) is renamed with its rows, and so are the names derived from it: the primary key
(`..._pkey`), the `ds` index (`idx_..._ds`) and the foreign key to the jobs table (`..._request_id_fkey`).

Pods on an image that still addresses `nilam_ocr_extraction_result` fail their queries between this migration and
their replacement: stop the services (scale 0) before it, deploy the new images after it. Anything outside this
repository that reads the table has to use the new name.

The names are written out rather than taken from `tables.py`, so this revision keeps acting on the names it was
written for.
"""

import logging
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0017_ocr_extraction_results"
down_revision: str | None = "0016_ocr_results_append_only"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

log = logging.getLogger("alembic.runtime.migration")

SCHEMA = "nilam_ocr_npwp"
RENAMES = {f"{lane}ocr_extraction_result": f"{lane}ocr_extraction_results" for lane in ("nilam_", "nilam_testing_")}

# Every object named after the table: its indexes (renaming the index of a primary key renames the constraint) and
# its other constraints (the foreign key).
DERIVED_NAMES = sa.text(
    """
    SELECT 'INDEX' AS kind, i.relname AS name
      FROM pg_index x JOIN pg_class i ON i.oid = x.indexrelid
     WHERE x.indrelid = CAST(:table AS regclass)
    UNION ALL
    SELECT 'CONSTRAINT', c.conname
      FROM pg_constraint c
     WHERE c.conrelid = CAST(:table AS regclass) AND c.contype NOT IN ('p', 'u', 'x')
    """
)


def _rename(old: str, new: str) -> None:
    """Renames table `old` to `new` in SCHEMA, and `old` to `new` in the names of the objects named after it."""
    conn = op.get_bind()
    if conn.execute(sa.text("SELECT to_regclass(:name)"), {"name": f'"{SCHEMA}"."{new}"'}).scalar():
        raise RuntimeError(f"{SCHEMA}.{new} already exists next to {SCHEMA}.{old}; compare the two by hand first")
    op.execute(sa.text(f'ALTER TABLE "{SCHEMA}"."{old}" RENAME TO "{new}"'))
    for kind, name in conn.execute(DERIVED_NAMES, {"table": f'"{SCHEMA}"."{new}"'}).all():
        # `old` is a prefix of `new`: rename only the names not renamed yet.
        if old not in name or new in name:
            continue
        renamed = name.replace(old, new, 1)
        if kind == "CONSTRAINT":
            op.execute(sa.text(f'ALTER TABLE "{SCHEMA}"."{new}" RENAME CONSTRAINT "{name}" TO "{renamed}"'))
        else:
            op.execute(sa.text(f'ALTER INDEX "{SCHEMA}"."{name}" RENAME TO "{renamed}"'))
    rows = conn.execute(sa.text(f'SELECT count(*) FROM "{SCHEMA}"."{new}"')).scalar()
    log.info("renamed %s.%s to %s (%s rows)", SCHEMA, old, new, rows)


def upgrade() -> None:
    for old, new in RENAMES.items():
        _rename(old, new)


def downgrade() -> None:
    for old, new in RENAMES.items():
        _rename_back(new, old)


def _rename_back(current: str, target: str) -> None:
    """The reverse of `_rename`: `target` is a prefix of `current`, so match the derived names on `current`."""
    conn = op.get_bind()
    op.execute(sa.text(f'ALTER TABLE "{SCHEMA}"."{current}" RENAME TO "{target}"'))
    for kind, name in conn.execute(DERIVED_NAMES, {"table": f'"{SCHEMA}"."{target}"'}).all():
        if current not in name:
            continue
        renamed = name.replace(current, target, 1)
        if kind == "CONSTRAINT":
            op.execute(sa.text(f'ALTER TABLE "{SCHEMA}"."{target}" RENAME CONSTRAINT "{name}" TO "{renamed}"'))
        else:
            op.execute(sa.text(f'ALTER INDEX "{SCHEMA}"."{name}" RENAME TO "{renamed}"'))
