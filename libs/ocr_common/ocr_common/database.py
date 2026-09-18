"""
Koneksi database (SQLAlchemy 2.0 async), pola sama dengan ocr-orchestration.

Dipakai service yang menyimpan status (ekstraksi, structuring, scoring).
Engine dibuat per URL saat pertama dipakai dan dibuang saat shutdown. Skema
dipasang manual dari services/<nama>/db/schema.sql; service tidak
membuat/migrasi tabel sendiri. Butuh extra `ocr-common[db]`.
"""

from sqlalchemy import JSON, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

# JSONB di PostgreSQL, JSON generik di dialek lain (SQLite untuk test).
# none_as_null: tanpa ini None tersimpan sebagai JSON `null`, bukan SQL NULL.
JSON_TYPE = JSON(none_as_null=True).with_variant(JSONB(none_as_null=True), "postgresql")


class Base(DeclarativeBase):
    pass


_engines: dict[str, AsyncEngine] = {}
_factories: dict[str, async_sessionmaker] = {}


def get_engine(url: str) -> AsyncEngine:
    if url not in _engines:
        # hide_parameters=True: tanpa ini DBAPIError menempelkan "[parameters: (...)]"
        # ke traceback, dan parameter itu isi baris yang sedang ditulis (hasil OCR).
        _engines[url] = create_async_engine(url, pool_pre_ping=True, hide_parameters=True)
    return _engines[url]


def get_session_factory(url: str) -> async_sessionmaker:
    if url not in _factories:
        _factories[url] = async_sessionmaker(get_engine(url), expire_on_commit=False)
    return _factories[url]


async def check_connection(url: str) -> None:
    """Gagal keras saat startup kalau database tidak terjangkau, bukan di request pertama."""
    async with get_engine(url).connect() as conn:
        await conn.execute(text("SELECT 1"))


async def dispose_engines() -> None:
    for engine in _engines.values():
        await engine.dispose()
    _engines.clear()
    _factories.clear()
