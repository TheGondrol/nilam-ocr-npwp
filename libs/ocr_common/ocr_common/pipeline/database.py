from sqlalchemy import JSON, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

JSON_TYPE = JSON(none_as_null=True).with_variant(JSONB(none_as_null=True), "postgresql")


class Base(DeclarativeBase):
    pass


_engines: dict[str, AsyncEngine] = {}
_factories: dict[str, async_sessionmaker] = {}


def get_engine(url: str) -> AsyncEngine:
    if url not in _engines:
        _engines[url] = create_async_engine(url, pool_pre_ping=True, hide_parameters=True)
    return _engines[url]


def get_session_factory(url: str) -> async_sessionmaker:
    if url not in _factories:
        _factories[url] = async_sessionmaker(get_engine(url), expire_on_commit=False)
    return _factories[url]


async def check_connection(url: str) -> None:
    async with get_engine(url).connect() as conn:
        await conn.execute(text("SELECT 1"))


async def dispose_engines() -> None:
    for engine in _engines.values():
        await engine.dispose()
    _engines.clear()
    _factories.clear()
