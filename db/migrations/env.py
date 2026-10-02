import asyncio
import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from ocr_common.pipeline.database import PIPELINE_SCHEMA
from ocr_common.pipeline.tables import repo_metadata

VERSION_TABLE = "nilam_ocr_npwp_alembic_version"
# Where earlier revisions kept the version table, newest first: up to 0012 in `ocr_pipeline_npwp`, up to 0010 in
# `ocr_pipeline`, up to 0009 in `public`, all three under the name it had before the `nilam_` prefix.
OLD_VERSION_TABLES = (
    ("ocr_pipeline_npwp", "ocr_npwp_alembic_version"),
    ("ocr_pipeline", "ocr_npwp_alembic_version"),
    ("public", "ocr_npwp_alembic_version"),
)

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
    """Earlier revisions kept the version table elsewhere and under another name (`OLD_VERSION_TABLES`); Alembic
    now reads it as `PIPELINE_SCHEMA.VERSION_TABLE` and would otherwise take such a database for an empty one.
    Moved and renamed with its primary key, rows kept. Idempotent, committed on its own."""
    connection.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{PIPELINE_SCHEMA}"'))
    if not connection.execute(text(f"SELECT to_regclass('{PIPELINE_SCHEMA}.{VERSION_TABLE}')")).scalar():
        for old_schema, old_name in OLD_VERSION_TABLES:
            if connection.execute(text(f"SELECT to_regclass('{old_schema}.{old_name}')")).scalar():
                connection.execute(text(f'ALTER TABLE "{old_schema}"."{old_name}" SET SCHEMA "{PIPELINE_SCHEMA}"'))
                connection.execute(text(f'ALTER TABLE "{PIPELINE_SCHEMA}"."{old_name}" RENAME TO "{VERSION_TABLE}"'))
                connection.execute(
                    text(f'ALTER INDEX IF EXISTS "{PIPELINE_SCHEMA}"."{old_name}_pkc" RENAME TO "{VERSION_TABLE}_pkc"')
                )
                break
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
