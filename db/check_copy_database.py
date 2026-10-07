"""CI check of copy_database.py (the move to Cloud SQL): from a migrated source with rows in every table into an
empty database, the copy gets every row and leaves the source as it was; a second run adds only what is new;
--replace makes the target an exact copy, a changed row included; the target's ids continue after the copied
ones; and `alembic check` passes on the target (copy_database.py runs it).

    DATABASE_URL=postgresql+asyncpg://.../<any database> python db/check_copy_database.py
    # creates (and first drops) the databases copy_source and copy_target on that server
"""

import asyncio
import json
import os
import subprocess
import sys

import asyncpg

HERE = os.path.dirname(os.path.abspath(__file__))
SCHEMA = "nilam_ocr_npwp"
SERVER = os.environ["DATABASE_URL"].rsplit("/", 1)[0]
SOURCE, TARGET = f"{SERVER}/copy_source", f"{SERVER}/copy_target"


def plain(url: str) -> str:
    return url.replace("+asyncpg", "")


def run(*args: str, **env: str) -> None:
    subprocess.run([sys.executable, *args], check=True, env={**os.environ, **env})


def copy(*args: str) -> None:
    run(os.path.join(HERE, "copy_database.py"), *args, SOURCE_DATABASE_URL=SOURCE, TARGET_DATABASE_URL=TARGET)


async def fetch(url: str, query: str) -> list:
    conn = await asyncpg.connect(plain(url))
    try:
        return await conn.fetch(query)
    finally:
        await conn.close()


async def execute(url: str, *queries: str) -> None:
    conn = await asyncpg.connect(plain(url))
    try:
        for query in queries:
            await conn.execute(query)
    finally:
        await conn.close()


# The jobs and results tables of each stage (the extraction stage's since 0014).
STAGE_TABLES = {
    "ocr": ("ocr_extraction_jobs", "ocr_extraction_result"),
    "structuring": ("structuring_jobs", "structuring_results"),
    "scoring": ("scoring_jobs", "scoring_results"),
}


async def seed() -> None:
    conn = await asyncpg.connect(plain(SOURCE))
    try:
        for lane in ("", "testing_"):
            for stage, (jobs, results) in STAGE_TABLES.items():
                for n in range(1200 if (lane, stage) == ("", "ocr") else 3):  # more than one batch of 1000
                    rid = f"REQ_{lane}{stage}_{n:05d}"
                    await conn.execute(
                        f'INSERT INTO {SCHEMA}."nilam_{lane}{jobs}" (request_id, status, input, ds) '
                        "VALUES ($1, 'DONE', $2, '20261005')",
                        rid,
                        json.dumps({"n": n}),
                    )
                    await conn.execute(
                        f'INSERT INTO {SCHEMA}."nilam_{lane}{results}" (request_id, result, ds) '
                        "VALUES ($1, $2, '20261005')",
                        rid,
                        json.dumps({"fields": {"nomor_npwp": {"value": str(n)}}}),
                    )
            for n in range(4):
                await conn.execute(
                    f'INSERT INTO {SCHEMA}."nilam_{lane}pipeline_outbox" (request_id, stage, kind, payload, ds) '
                    "VALUES ($1, 'OCR', 'callback', '{}', '20261005')",
                    f"REQ_{n}",
                )
                await conn.execute(
                    f'INSERT INTO {SCHEMA}."nilam_{lane}guardrails_results" '
                    "(request_id, passed, threshold_source, report, ds) VALUES ($1, true, 'service', '{}', '20261005')",
                    f"REQ_{n}",
                )
                await conn.execute(
                    f'INSERT INTO {SCHEMA}."nilam_{lane}ocr_results" (status_code, status_desc, data, request_id) '
                    "VALUES (200, 'OK', '{}', $1)",
                    f"REQ_{n}",
                )
    finally:
        await conn.close()


async def digest(url: str) -> list:
    """Every table's rows as one hash each: equal digests are equal contents. Rows are compared as jsonb (keyed by
    column name), because the column order differs: migrations appended some columns, the target has the order
    of tables.py."""
    tables = await fetch(
        url, f"SELECT table_name FROM information_schema.tables WHERE table_schema = '{SCHEMA}' ORDER BY 1"
    )
    hashes = []
    for (name,) in tables:
        row = "to_jsonb(t)::text"
        [(value,)] = await fetch(
            url, f"SELECT md5(coalesce(string_agg({row}, ',' ORDER BY {row}), '')) FROM {SCHEMA}.\"{name}\" t"
        )
        hashes.append((name, value))
    return hashes


async def main() -> None:
    await execute(
        SERVER + "/postgres",
        "DROP DATABASE IF EXISTS copy_source",
        "DROP DATABASE IF EXISTS copy_target",
        "CREATE DATABASE copy_source",
        "CREATE DATABASE copy_target",
    )
    run("-m", "alembic", "-c", os.path.join(HERE, "alembic.ini"), "upgrade", "head", DATABASE_URL=SOURCE)
    await seed()
    source_before = await digest(SOURCE)

    copy("--check")
    copy()
    assert await digest(TARGET) == source_before, "the copy differs from the source"
    assert await digest(SOURCE) == source_before, "the source changed"

    # Without --replace a second run adds the new row and leaves the changed one; --replace takes both.
    await execute(
        SOURCE,
        f"UPDATE {SCHEMA}.nilam_scoring_jobs SET status = 'FAILED' WHERE request_id = 'REQ_scoring_00000'",
        f"INSERT INTO {SCHEMA}.nilam_pipeline_outbox (request_id, stage, kind, payload, ds) "
        "VALUES ('REQ_new', 'OCR', 'handoff', '{}', '20261005')",
    )
    copy()
    [(status,)] = await fetch(
        TARGET, f"SELECT status FROM {SCHEMA}.nilam_scoring_jobs WHERE request_id = 'REQ_scoring_00000'"
    )
    assert status == "DONE", "without --replace a row the target has is left as it is"
    copy("--replace")
    assert await digest(TARGET) == await digest(SOURCE), "--replace is not an exact copy"

    [(max_id,)] = await fetch(TARGET, f"SELECT max(id) FROM {SCHEMA}.nilam_pipeline_outbox")
    [(next_id,)] = await fetch(
        TARGET,
        f"INSERT INTO {SCHEMA}.nilam_pipeline_outbox (request_id, stage, kind, payload, ds) "
        "VALUES ('REQ_after', 'OCR', 'callback', '{}', '20261005') RETURNING id",
    )
    assert next_id == max_id + 1, f"the id sequence did not continue: {next_id} after {max_id}"
    print("copy_database: OK")


asyncio.run(main())
