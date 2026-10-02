"""The client's naming: schema `nilam_ocr_npwp`, every table name prefixed with `nilam_`, rows included

Revision ID: 0013_nilam_naming
Revises: 0012_drop_threshold_target
Create Date: 2026-10-02

Every table of this repository moves from `ocr_pipeline_npwp` to `nilam_ocr_npwp` and gets the `nilam_` prefix:
`ocr_pipeline_npwp.ocr_jobs` becomes `nilam_ocr_npwp.nilam_ocr_jobs`, `testing_ocr_jobs` becomes
`nilam_testing_ocr_jobs`. As in 0010 and 0011 nothing is copied: `ALTER TABLE ... SET SCHEMA` and `RENAME` change
the catalog only, in the migration's one transaction. The names derived from the table name follow it, so they
keep matching what PostgreSQL and `tables.py` would give a new table: the indexes (`idx_ocr_jobs_status` ->
`idx_nilam_ocr_jobs_status`, `ocr_jobs_pkey` -> `nilam_ocr_jobs_pkey`), the foreign keys and the sequences of the
`id` columns (the column defaults point at the sequence itself, not its name, so the ids carry on). Then the
emptied `ocr_pipeline_npwp` is dropped, without CASCADE: if something else was put in it, the migration stops.

The version table is moved and renamed by `env.py` before any revision runs (`nilam_ocr_npwp_alembic_version`).
Pods on an image that still addresses `ocr_pipeline_npwp` fail their queries between this migration and their
replacement, and anything outside this repository that reads these tables (the central orchestrator team's
queries, the dashboards) has to use the new names from then on. Grants given on `ocr_pipeline_npwp` do not carry
over to the new schema.

The schema and table names are written out rather than taken from `tables.py`, so this revision keeps acting on
the names it was written for.
"""

import logging
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0013_nilam_naming"
down_revision: str | None = "0012_drop_threshold_target"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

log = logging.getLogger("alembic.runtime.migration")

OLD_SCHEMA = "ocr_pipeline_npwp"
NEW_SCHEMA = "nilam_ocr_npwp"
PREFIX = "nilam_"

TABLES = tuple(
    f"{lane}{name}"
    for lane in ("", "testing_")
    for name in (
        "ocr_jobs",
        "ocr_results",
        "structuring_jobs",
        "structuring_results",
        "scoring_jobs",
        "scoring_results",
        "pipeline_outbox",
        "guardrails_results",
    )
)

# Every object named after the table: its indexes (renaming the index of a primary key or unique constraint renames
# the constraint), its other constraints (the foreign keys) and the sequences it owns.
DERIVED_NAMES = sa.text(
    """
    SELECT 'INDEX' AS kind, i.relname AS name
      FROM pg_index x JOIN pg_class i ON i.oid = x.indexrelid
     WHERE x.indrelid = CAST(:table AS regclass)
    UNION ALL
    SELECT 'CONSTRAINT', c.conname
      FROM pg_constraint c
     WHERE c.conrelid = CAST(:table AS regclass) AND c.contype NOT IN ('p', 'u', 'x')
    UNION ALL
    SELECT 'SEQUENCE', s.relname
      FROM pg_depend d JOIN pg_class s ON s.oid = d.objid
     WHERE d.refobjid = CAST(:table AS regclass) AND s.relkind = 'S' AND d.deptype IN ('a', 'i')
    """
)


def _rename_derived(schema: str, table: str, old: str, new: str) -> None:
    """Puts `new` in place of `old` in the names of the objects named after the table, now called `table`."""
    conn = op.get_bind()
    for kind, name in conn.execute(DERIVED_NAMES, {"table": f'"{schema}"."{table}"'}).all():
        if old not in name:
            continue
        renamed = name.replace(old, new, 1)
        if kind == "CONSTRAINT":
            op.execute(sa.text(f'ALTER TABLE "{schema}"."{table}" RENAME CONSTRAINT "{name}" TO "{renamed}"'))
        else:
            op.execute(sa.text(f'ALTER {kind} "{schema}"."{name}" RENAME TO "{renamed}"'))


def _move(source: str, target: str, names: dict[str, str]) -> None:
    """Moves every table from `source` to `target` and renames it after `names` (old name -> new name)."""
    conn = op.get_bind()
    op.execute(sa.text(f'CREATE SCHEMA IF NOT EXISTS "{target}"'))
    for old, new in names.items():
        if conn.execute(sa.text("SELECT to_regclass(:name)"), {"name": f'"{target}"."{new}"'}).scalar():
            raise RuntimeError(
                f"{target}.{new} already exists next to {source}.{old}; compare the two by hand before running "
                "this migration again"
            )
        op.execute(sa.text(f'ALTER TABLE "{source}"."{old}" SET SCHEMA "{target}"'))
        op.execute(sa.text(f'ALTER TABLE "{target}"."{old}" RENAME TO "{new}"'))
        _rename_derived(target, new, old, new)
        rows = conn.execute(sa.text(f'SELECT count(*) FROM "{target}"."{new}"')).scalar()
        log.info("moved %s.%s to %s.%s (%s rows)", source, old, target, new, rows)


def upgrade() -> None:
    _move(OLD_SCHEMA, NEW_SCHEMA, {table: f"{PREFIX}{table}" for table in TABLES})
    op.execute(sa.text(f'DROP SCHEMA "{OLD_SCHEMA}"'))


def downgrade() -> None:
    # The new schema stays: env.py keeps the version table in it.
    _move(NEW_SCHEMA, OLD_SCHEMA, {f"{PREFIX}{table}": table for table in TABLES})
