"""Copies every table of this repository (schema `nilam_ocr_npwp`, live and testing_) from one database to another,
e.g. to move to Cloud SQL, without changing the source: it is read in one read-only snapshot.

    SOURCE_DATABASE_URL=... TARGET_DATABASE_URL=... python db/copy_database.py [--check | --replace]

Either URL may be a Cloud SQL URL (`cloudsql_instance`, IAM login; see ocr_common.pipeline.database).

1. The source must be at the migrations' head. The target gets the schema and the tables from the code's table
   definitions (no CREATE in `public` needed, which PostgreSQL 15+ only allows the database owner), is stamped
   at that head, and `alembic check` then proves it matches the migrations.
2. Rows are copied table by table in primary-key order. Without --replace a row the target already has is left
   as it is, so a run can be repeated; with --replace the target tables are emptied first, for an exact copy
   (the cutover, with every service stopped). The id sequences then continue after the copied ids.
3. Every table's row count must be the same in the source snapshot and the target, else the exit code is 1.

--check only connects to both, compares the versions and prints the counts: nothing is created or copied, and
only a version mismatch fails it (the target is expected to be empty then).
"""

import argparse
import asyncio
import io
import os
import subprocess
import sys
from typing import Any, cast

from alembic.config import Config
from alembic.script import ScriptDirectory
from sqlalchemy import Integer, Table, func, make_url, select, text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncConnection

from ocr_common.pipeline.database import PIPELINE_SCHEMA, dispose_engines, get_engine
from ocr_common.pipeline.tables import repo_metadata

ALEMBIC_INI = os.path.join(os.path.dirname(os.path.abspath(__file__)), "alembic.ini")
VERSION_TABLE = f'"{PIPELINE_SCHEMA}".nilam_ocr_npwp_alembic_version'
BATCH = 1000


def shown(url: str) -> str:
    """The URL without its password, for the log."""
    return make_url(url).render_as_string(hide_password=True)


def head() -> str:
    return str(ScriptDirectory.from_config(Config(ALEMBIC_INI)).get_current_head())


async def version(conn: AsyncConnection) -> str | None:
    """The database's migration version; None when it has no version table (yet)."""
    if not (await conn.execute(text(f"SELECT to_regclass('{VERSION_TABLE}')"))).scalar():
        return None
    return (await conn.execute(text(f"SELECT version_num FROM {VERSION_TABLE}"))).scalar()


async def count(conn: AsyncConnection, table: Table) -> int | None:
    """The table's rows; None when the table does not exist."""
    if not (await conn.execute(text(f'SELECT to_regclass(\'"{table.schema}"."{table.name}"\')'))).scalar():
        return None
    return int((await conn.execute(select(func.count()).select_from(table))).scalar_one())


async def copy_table(source: AsyncConnection, target_url: str, table: Table) -> None:
    """Every row of `table`, in primary-key order, `BATCH` rows per INSERT; a row the target has is skipped."""
    [key] = table.primary_key.columns
    last: Any = None
    while True:
        query = select(table).order_by(key).limit(BATCH)
        if last is not None:
            query = query.where(key > last)
        rows = cast(list[dict[str, Any]], [dict(row) for row in (await source.execute(query)).mappings()])
        if not rows:
            return
        async with get_engine(target_url).begin() as target:
            await target.execute(insert(table).on_conflict_do_nothing(), rows)
        last = rows[-1][key.name]


async def continue_sequences(target: AsyncConnection, tables: list[Table]) -> None:
    """The id sequences go on after the copied ids, so the services' next INSERT does not collide."""
    for table in tables:
        for column in table.primary_key.columns:
            if column.autoincrement is True and isinstance(column.type, Integer):
                qualified = f'"{table.schema}"."{table.name}"'
                await target.execute(
                    text(
                        f"SELECT setval(pg_get_serial_sequence(:table, :column), "
                        f'COALESCE(MAX("{column.name}"), 1), MAX("{column.name}") IS NOT NULL) FROM {qualified}'
                    ),
                    {"table": qualified, "column": column.name},
                )


async def run(source_url: str, target_url: str, *, check: bool, replace: bool) -> bool:
    metadata = repo_metadata()
    tables = list(metadata.sorted_tables)  # jobs before the results that refer to them
    expected = head()
    try:
        async with get_engine(source_url).connect() as source:
            # One consistent, read-only snapshot of the source for the copy and its counts.
            await source.execute(text("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY"))
            source_version = await version(source)
            if source_version != expected:
                print(f"the source is at {source_version}, not at the head {expected}: migrate it first")
                return False
            async with get_engine(target_url).begin() as target:
                target_version = await version(target)
            if target_version not in (None, expected):
                print(f"the target is at {target_version}, not at the head {expected}: refusing to copy into it")
                return False

            if not check:
                async with get_engine(target_url).begin() as target:
                    await target.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{PIPELINE_SCHEMA}"'))
                    await target.run_sync(metadata.create_all)
                    if replace:
                        names = ", ".join(f'"{table.schema}"."{table.name}"' for table in tables)
                        await target.execute(text(f"TRUNCATE {names}"))
                for table in tables:
                    await copy_table(source, target_url, table)
                    print(f"copied {table.name}")
                async with get_engine(target_url).begin() as target:
                    await continue_sequences(target, tables)

            matched = True
            print(f"\n{'table':42} {'source':>8} {'target':>8}")
            async with get_engine(target_url).connect() as target:
                for table in tables:
                    source_rows, target_rows = await count(source, table), await count(target, table)
                    same = source_rows == target_rows
                    matched = matched and same
                    print(f"{table.name:42} {source_rows!s:>8} {target_rows!s:>8}{'' if same else '  <- differs'}")
            return matched or check
    finally:
        await dispose_engines()


def alembic(target_url: str, *args: str) -> None:
    subprocess.run(
        [sys.executable, "-m", "alembic", "-c", ALEMBIC_INI, *args],
        check=True,
        env={**os.environ, "DATABASE_URL": target_url},
    )


def main() -> int:
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(line_buffering=True)  # in order with the alembic subprocess output
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="connect, compare versions and count; change nothing")
    mode.add_argument("--replace", action="store_true", help="empty the target tables first: an exact copy")
    args = parser.parse_args()

    source_url, target_url = os.environ.get("SOURCE_DATABASE_URL"), os.environ.get("TARGET_DATABASE_URL")
    if not source_url or not target_url:
        raise SystemExit("set SOURCE_DATABASE_URL (the database copied from) and TARGET_DATABASE_URL (copied to)")
    if make_url(source_url) == make_url(target_url):
        raise SystemExit("SOURCE_DATABASE_URL and TARGET_DATABASE_URL are the same database")
    print(f"source: {shown(source_url)}\ntarget: {shown(target_url)}\n")

    matched = asyncio.run(run(source_url, target_url, check=args.check, replace=args.replace))
    if not args.check:
        alembic(target_url, "stamp", "head")
        alembic(target_url, "check")
    if args.check:
        print("\ncheck only: nothing was created or copied")
    else:
        print("\nrow counts match" if matched else "\nrow counts DIFFER")
    return 0 if matched else 1


if __name__ == "__main__":
    sys.exit(main())
