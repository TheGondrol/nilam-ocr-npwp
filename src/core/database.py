"""
Koneksi database (SQLAlchemy 2.0 async), pola sama dengan ocr-orchestration.

Bedanya: di sini database OPSIONAL. Tanpa DATABASE_URL, status request_id
disimpan in-memory seperti mock ocr-*; dengan DATABASE_URL, disimpan di
PostgreSQL lewat repository SQL. Karena itu engine dibuat saat pertama
dipakai, bukan saat import. Skema dipasang manual dari db/schema.sql;
aplikasi ini tidak membuat/migrasi tabel sendiri.
"""

from functools import lru_cache

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from src.core.config import get_settings


class Base(DeclarativeBase):
    pass


def _require_url() -> str:
    url = get_settings().database_url
    if not url:
        raise RuntimeError("DATABASE_URL is not set; the SQL request repository cannot be used")
    return url


@lru_cache
def get_engine() -> AsyncEngine:
    # hide_parameters=True: tanpa ini DBAPIError menempelkan "[parameters: (...)]"
    # ke traceback, dan parameter itu isi baris yang sedang ditulis (hasil OCR).
    return create_async_engine(_require_url(), pool_pre_ping=True, hide_parameters=True)


@lru_cache
def get_session_factory() -> async_sessionmaker:
    return async_sessionmaker(get_engine(), expire_on_commit=False)


async def check_connection() -> None:
    """Gagal keras saat startup kalau database tidak terjangkau, bukan di request pertama."""
    async with get_engine().connect() as conn:
        await conn.execute(text("SELECT 1"))


async def dispose_engine() -> None:
    if get_engine.cache_info().currsize:
        await get_engine().dispose()
    get_engine.cache_clear()
    get_session_factory.cache_clear()
