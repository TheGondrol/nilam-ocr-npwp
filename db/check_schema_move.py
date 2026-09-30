"""CI check of migrations 0010 and 0011: a database at 0009, with its rows and its version table in `public`, ends
up with every table and every row in `ocr_pipeline_npwp`, keeps counting its ids where it left off, gets there
from 0010 too (the version table in `ocr_pipeline`, as on dev), and survives downgrades and upgrades again.

    DATABASE_URL=postgresql+asyncpg://... python db/check_schema_move.py   # an EMPTY database: it is rebuilt
"""

import asyncio
import os
import subprocess
import sys

import asyncpg

ALEMBIC = [sys.executable, "-m", "alembic", "-c", os.path.join(os.path.dirname(__file__), "alembic.ini")]
SCHEMA = "ocr_pipeline_npwp"
OLD_SCHEMA = "ocr_pipeline"
VERSION_TABLE = "ocr_npwp_alembic_version"
TABLES = [
    f"{lane}{name}"
    for lane in ("", "testing_")
    for name in (
        *(f"{stage}_{kind}" for stage in ("ocr", "structuring", "scoring") for kind in ("jobs", "results")),
        "pipeline_outbox",
        "guardrails_results",
    )
]


def alembic(*args: str) -> None:
    subprocess.run([*ALEMBIC, *args], check=True)


async def connect() -> asyncpg.Connection:
    return await asyncpg.connect(os.environ["DATABASE_URL"].replace("+asyncpg", ""))


async def tables_in(conn: asyncpg.Connection, schema: str) -> set[str]:
    rows = await conn.fetch("SELECT tablename FROM pg_tables WHERE schemaname = $1", schema)
    return {row["tablename"] for row in rows}


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


async def counts(conn: asyncpg.Connection, schema: str) -> dict[str, int]:
    return {table: await conn.fetchval(f'SELECT count(*) FROM "{schema}"."{table}"') for table in TABLES}


async def main() -> None:
    alembic("upgrade", "0009_guardrails_results_sequence")
    conn = await connect()
    try:
        assert set(TABLES) <= await tables_in(conn, "public"), "0009 should leave the tables in public"
        # Where the migration image up to 0009 kept the version table.
        await conn.execute(f"ALTER TABLE {SCHEMA}.{VERSION_TABLE} SET SCHEMA public")
        await seed(conn)
        before = await counts(conn, "public")
    finally:
        await conn.close()

    alembic("upgrade", "head")
    alembic("check")
    conn = await connect()
    try:
        for schema in ("public", OLD_SCHEMA):
            assert not (set(TABLES) | {VERSION_TABLE}) & await tables_in(conn, schema), f"tables left in {schema}"
        assert not await conn.fetchval("SELECT 1 FROM pg_namespace WHERE nspname = $1", OLD_SCHEMA), "schema left"
        assert set(TABLES) | {VERSION_TABLE} <= await tables_in(conn, SCHEMA)
        assert await counts(conn, SCHEMA) == before, "rows lost in the move"
        # The id sequences moved with their tables and carry on after the rows already there.
        new_id = await conn.fetchval(
            f"INSERT INTO {SCHEMA}.pipeline_outbox (request_id, stage, kind, payload, ds) "
            "VALUES ('REQ_3', 'OCR', 'handoff', '{}', '20260930') RETURNING id"
        )
        assert new_id == 3, new_id
        # The foreign key moved too: a result without its job is still refused.
        try:
            await conn.execute(f"INSERT INTO {SCHEMA}.ocr_results (request_id, result, ds) VALUES ('NOPE', '{{}}', '')")
        except asyncpg.ForeignKeyViolationError:
            pass
        else:
            raise AssertionError("ocr_results lost its foreign key to ocr_jobs")
        await conn.execute(f"DELETE FROM {SCHEMA}.pipeline_outbox WHERE request_id = 'REQ_3'")
    finally:
        await conn.close()

    # Back to 0010, with the version table where the 0010 image kept it (the dev database), then up again.
    alembic("downgrade", "0010_ocr_pipeline_schema")
    conn = await connect()
    try:
        assert await counts(conn, OLD_SCHEMA) == before, f"rows lost moving back to {OLD_SCHEMA}"
        await conn.execute(f"ALTER TABLE {SCHEMA}.{VERSION_TABLE} SET SCHEMA {OLD_SCHEMA}")
    finally:
        await conn.close()
    alembic("upgrade", "head")
    alembic("check")

    alembic("downgrade", "0009_guardrails_results_sequence")
    conn = await connect()
    try:
        assert await counts(conn, "public") == before, "rows lost moving back to public"
    finally:
        await conn.close()
    alembic("upgrade", "head")
    alembic("check")
    print(f"0010 + 0011 move every table and row to {SCHEMA}, from public and from {OLD_SCHEMA}, and back")


asyncio.run(main())
