"""CI check of migrations 0010, 0011 and 0013: a database at 0009, with its rows and its version table in `public`,
ends up with every table and every row in `nilam_ocr_npwp` under the `nilam_` names, keeps counting its ids where it
left off, gets there from 0012 (the version table in `ocr_pipeline_npwp`, as on dev) and from 0010 (in
`ocr_pipeline`) too, and survives downgrades and upgrades again.

    DATABASE_URL=postgresql+asyncpg://... python db/check_schema_move.py   # an EMPTY database: it is rebuilt
"""

import asyncio
import os
import subprocess
import sys

import asyncpg

ALEMBIC = [sys.executable, "-m", "alembic", "-c", os.path.join(os.path.dirname(__file__), "alembic.ini")]
SCHEMA = "nilam_ocr_npwp"
PREFIX = "nilam_"
OLD_SCHEMAS = ("ocr_pipeline", "ocr_pipeline_npwp")
VERSION_TABLE = "nilam_ocr_npwp_alembic_version"
OLD_VERSION_TABLE = "ocr_npwp_alembic_version"
# The names before 0013; from 0013 on each one carries PREFIX.
TABLES = [
    f"{lane}{name}"
    for lane in ("", "testing_")
    for name in (
        *(f"{stage}_{kind}" for stage in ("ocr", "structuring", "scoring") for kind in ("jobs", "results")),
        "pipeline_outbox",
        "guardrails_results",
    )
]
NEW_TABLES = [f"{PREFIX}{table}" for table in TABLES]


def alembic(*args: str) -> None:
    subprocess.run([*ALEMBIC, *args], check=True)


async def connect() -> asyncpg.Connection:
    return await asyncpg.connect(os.environ["DATABASE_URL"].replace("+asyncpg", ""))


async def tables_in(conn: asyncpg.Connection, schema: str) -> set[str]:
    rows = await conn.fetch("SELECT tablename FROM pg_tables WHERE schemaname = $1", schema)
    return {row["tablename"] for row in rows}


async def put_version_table(conn: asyncpg.Connection, schema: str) -> None:
    """Puts the version table where, and under the name, the migration image of an older revision kept it."""
    await conn.execute(f"ALTER TABLE {SCHEMA}.{VERSION_TABLE} SET SCHEMA {schema}")
    await conn.execute(f"ALTER TABLE {schema}.{VERSION_TABLE} RENAME TO {OLD_VERSION_TABLE}")
    await conn.execute(f"ALTER INDEX {schema}.{VERSION_TABLE}_pkc RENAME TO {OLD_VERSION_TABLE}_pkc")


async def seed(conn: asyncpg.Connection) -> None:
    for lane in ("", "testing_"):
        for stage in ("ocr", "structuring", "scoring"):
            await conn.execute(
                f"INSERT INTO public.{lane}{stage}_jobs (request_id, status, input, ds) "
                "VALUES ('REQ_1', 'DONE', '{\"document_type\": \"NPWP\"}', '20260930'), "
                "('REQ_2', 'FAILED', NULL, '20260930')"
            )
            await conn.execute(
                f"INSERT INTO public.{lane}{stage}_results (request_id, result, ds) "
                "VALUES ('REQ_1', '{\"ok\": 1}', '20260930')"
            )
        await conn.execute(
            f"INSERT INTO public.{lane}pipeline_outbox (request_id, stage, kind, payload, ds) "
            "VALUES ('REQ_1', 'OCR', 'handoff', '{}', '20260930'), ('REQ_2', 'OCR', 'callback', '{}', '20260930')"
        )
        await conn.execute(
            f"INSERT INTO public.{lane}guardrails_results (request_id, passed, threshold_source, report, ds) "
            "VALUES ('REQ_1', true, 'service', '{}', '20260930')"
        )


async def counts(conn: asyncpg.Connection, schema: str, prefix: str = "") -> dict[str, int]:
    """Rows per table, keyed by the name before 0013 whatever the table is called in `schema`."""
    return {table: int(await conn.fetchval(f'SELECT count(*) FROM "{schema}"."{prefix}{table}"')) for table in TABLES}


async def check_head(before: dict[str, int]) -> None:
    alembic("upgrade", "head")
    alembic("check")
    conn = await connect()
    try:
        old_names = set(TABLES) | set(NEW_TABLES) | {VERSION_TABLE, OLD_VERSION_TABLE}
        assert not old_names & await tables_in(conn, "public"), "tables left in public"
        for schema in OLD_SCHEMAS:
            assert not await conn.fetchval("SELECT 1 FROM pg_namespace WHERE nspname = $1", schema), f"{schema} left"
        assert await tables_in(conn, SCHEMA) == set(NEW_TABLES) | {VERSION_TABLE}
        assert await counts(conn, SCHEMA, PREFIX) == before, "rows lost in the move"
        # The id sequences moved and were renamed with their tables, and carry on after the rows already there.
        sequence = await conn.fetchval(f"SELECT pg_get_serial_sequence('{SCHEMA}.{PREFIX}pipeline_outbox', 'id')")
        assert sequence == f"{SCHEMA}.{PREFIX}pipeline_outbox_id_seq", sequence
        last_id = await conn.fetchval(f"SELECT max(id) FROM {SCHEMA}.{PREFIX}pipeline_outbox")
        new_id = await conn.fetchval(
            f"INSERT INTO {SCHEMA}.{PREFIX}pipeline_outbox (request_id, stage, kind, payload, ds) "
            "VALUES ('REQ_3', 'OCR', 'handoff', '{}', '20260930') RETURNING id"
        )
        assert new_id > last_id, (new_id, last_id)
        await conn.execute(f"DELETE FROM {SCHEMA}.{PREFIX}pipeline_outbox WHERE request_id = 'REQ_3'")
        # Every index, constraint and sequence is named after the table it now belongs to.
        stale = await conn.fetch(
            "SELECT relname FROM pg_class WHERE relnamespace = $1::regnamespace AND relkind IN ('i', 'S') "
            "AND strpos(relname, $2) = 0",
            SCHEMA,
            PREFIX,
        )
        assert not stale, [row["relname"] for row in stale]
        constraints = await conn.fetch(
            "SELECT conname FROM pg_constraint WHERE connamespace = $1::regnamespace AND strpos(conname, $2) <> 1",
            SCHEMA,
            PREFIX,
        )
        assert not constraints, [row["conname"] for row in constraints]
        # The foreign key moved too: a result without its job is still refused.
        try:
            await conn.execute(
                f"INSERT INTO {SCHEMA}.{PREFIX}ocr_results (request_id, result, ds) VALUES ('NOPE', '{{}}', '')"
            )
        except asyncpg.ForeignKeyViolationError:
            pass
        else:
            raise AssertionError("nilam_ocr_results lost its foreign key to nilam_ocr_jobs")
    finally:
        await conn.close()


async def downgrade_to(revision: str, schema: str, version_schema: str | None, before: dict[str, int]) -> None:
    """Downgrades to `revision`, checks the rows are back in `schema` under the old names, and puts the version
    table where that revision's image kept it (`None`: where env.py keeps it now)."""
    alembic("downgrade", revision)
    conn = await connect()
    try:
        assert await counts(conn, schema) == before, f"rows lost moving back to {schema}"
        if version_schema:
            await put_version_table(conn, version_schema)
    finally:
        await conn.close()


async def main() -> None:
    alembic("upgrade", "0009_guardrails_results_sequence")
    conn = await connect()
    try:
        assert set(TABLES) <= await tables_in(conn, "public"), "0009 should leave the tables in public"
        await put_version_table(conn, "public")  # where the migration image up to 0009 kept it
        await seed(conn)
        before = await counts(conn, "public")
    finally:
        await conn.close()

    await check_head(before)
    # Back to 0012, with the version table where the 0012 image keeps it (the dev database), then up again.
    await downgrade_to("0012_drop_threshold_target", "ocr_pipeline_npwp", "ocr_pipeline_npwp", before)
    await check_head(before)
    # The same from 0010 (version table in ocr_pipeline) and from 0009 (in public, already tested from the image's
    # placement above, so here from where env.py keeps it).
    await downgrade_to("0010_ocr_pipeline_schema", "ocr_pipeline", "ocr_pipeline", before)
    await check_head(before)
    await downgrade_to("0009_guardrails_results_sequence", "public", None, before)
    await check_head(before)
    print(f"0010, 0011 + 0013 move every table and row to {SCHEMA} under {PREFIX}*, from public, ocr_pipeline and "
          "ocr_pipeline_npwp, and back")


asyncio.run(main())
