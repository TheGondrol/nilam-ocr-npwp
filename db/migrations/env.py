import asyncio
import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from ocr_common.pipeline.database import PIPELINE_SCHEMA
from ocr_common.pipeline.tables import repo_metadata

VERSION_TABLE = "ocr_npwp_alembic_version"

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = repo_metadata()


def database_url() -> str:
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise SystemExit("DATABASE_URL is not set: point it at the database these migrations own")
    return url


def include_name(name, type_, parent_names) -> bool:
    # Only our schema is compared; `public` and the orchestrator's `ocr` are other teams' business.
    return name == PIPELINE_SCHEMA if type_ == "schema" else True


def include_object(obj, name, type_, reflected, compare_to) -> bool:
    return not (type_ == "table" and reflected and compare_to is None)


def configure(**kwargs) -> None:
    context.configure(
        target_metadata=target_metadata,
        version_table=VERSION_TABLE,
        version_table_schema=PIPELINE_SCHEMA,
        include_schemas=True,
        include_name=include_name,
        include_object=include_object,
        compare_type=True,
        **kwargs,
    )


def move_version_table(connection) -> None:
    """Up to revision 0009 the version table lived in `public`; Alembic now reads it from `PIPELINE_SCHEMA`
    and would otherwise take such a database for an empty one. Idempotent, committed on its own."""
    connection.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{PIPELINE_SCHEMA}"'))
    old = connection.execute(text(f"SELECT to_regclass('public.{VERSION_TABLE}')")).scalar()
    new = connection.execute(text(f"SELECT to_regclass('{PIPELINE_SCHEMA}.{VERSION_TABLE}')")).scalar()
    if old and not new:
        connection.execute(text(f'ALTER TABLE public.{VERSION_TABLE} SET SCHEMA "{PIPELINE_SCHEMA}"'))
    connection.commit()


def run_migrations_offline() -> None:
    configure(url=database_url(), literal_binds=True, dialect_opts={"paramstyle": "named"})
    with context.begin_transaction():
        context.run_migrations()


def run_migrations(connection) -> None:
    move_version_table(connection)
    configure(connection=connection)
    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    engine = create_async_engine(database_url())
    async with engine.connect() as connection:
        await connection.run_sync(run_migrations)
    await engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())
