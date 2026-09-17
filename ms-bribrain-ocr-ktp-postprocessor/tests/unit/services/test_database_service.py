"""Tests for src/services/database_service.py."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.schemas.api_schema import LogEntry
from src.services import database_service


@pytest.fixture(autouse=True)
def _reset_globals():
    """Reset module globals before and after each test."""
    database_service._async_engine = None
    database_service._async_session_factory = None
    yield
    database_service._async_engine = None
    database_service._async_session_factory = None


def _make_entry(**overrides):
    defaults = dict(
        request_id="req-1",
        response_code=200,
        payload={"x": 1},
        error_message="",
        result={"ok": True},
        processing_time=0.1,
    )
    defaults.update(overrides)
    return LogEntry(**defaults)


class TestInitEngine:
    @pytest.mark.asyncio
    async def test_init_with_database_url(self, monkeypatch):
        monkeypatch.setattr(database_service, "DATABASE_URL", "postgresql+asyncpg://u:p@h/db")
        fake_engine = MagicMock()
        with patch.object(database_service, "create_async_engine", return_value=fake_engine) as ce, \
             patch.object(database_service, "async_sessionmaker", return_value=MagicMock()) as asm:
            await database_service.init_engine()
            ce.assert_called_once()
            asm.assert_called_once()
        assert database_service._async_engine is fake_engine
        assert database_service._async_session_factory is not None

    @pytest.mark.asyncio
    async def test_init_skipped_when_no_database_url(self, monkeypatch):
        monkeypatch.setattr(database_service, "DATABASE_URL", None)
        with patch.object(database_service, "create_async_engine") as ce:
            await database_service.init_engine()
            ce.assert_not_called()
        assert database_service._async_engine is None

    @pytest.mark.asyncio
    async def test_init_skipped_when_already_initialized(self, monkeypatch):
        monkeypatch.setattr(database_service, "DATABASE_URL", "postgresql+asyncpg://u:p@h/db")
        database_service._async_engine = MagicMock()  # pretend already initialized
        with patch.object(database_service, "create_async_engine") as ce:
            await database_service.init_engine()
            ce.assert_not_called()


class TestDisposeEngine:
    @pytest.mark.asyncio
    async def test_dispose_when_initialized(self):
        fake_engine = MagicMock()
        fake_engine.dispose = AsyncMock()
        database_service._async_engine = fake_engine
        database_service._async_session_factory = MagicMock()
        await database_service.dispose_engine()
        fake_engine.dispose.assert_awaited_once()
        assert database_service._async_engine is None
        assert database_service._async_session_factory is None

    @pytest.mark.asyncio
    async def test_dispose_noop_when_not_initialized(self):
        # Should not raise
        await database_service.dispose_engine()
        assert database_service._async_engine is None


class TestInsertLog:
    @pytest.mark.asyncio
    async def test_skips_when_database_disabled(self, monkeypatch):
        monkeypatch.setattr(database_service, "insert_to_database", False)
        # Should return without touching session factory
        database_service._async_session_factory = MagicMock()
        await database_service.insert_log(_make_entry())
        database_service._async_session_factory.assert_not_called()

    @pytest.mark.asyncio
    async def test_skips_when_session_factory_none(self, monkeypatch):
        monkeypatch.setattr(database_service, "insert_to_database", True)
        database_service._async_session_factory = None
        # Just verify no exception
        await database_service.insert_log(_make_entry())

    @pytest.mark.asyncio
    async def test_insert_success(self, monkeypatch):
        monkeypatch.setattr(database_service, "insert_to_database", True)
        # insert_log now encrypts payload/result (needs LOG_ENCRYPTION_KEY);
        # stub encrypt to a passthrough so the DB-write path is exercised.
        monkeypatch.setattr(database_service, "encrypt", lambda v: v)
        session = MagicMock()
        session.add = MagicMock()
        session.commit = AsyncMock()

        class _CtxFactory:
            def __call__(self):
                return self

            async def __aenter__(self_inner):
                return session

            async def __aexit__(self_inner, exc_type, exc, tb):
                return False

        database_service._async_session_factory = _CtxFactory()
        await database_service.insert_log(_make_entry())
        session.add.assert_called_once()
        session.commit.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_insert_logs_on_exception(self, monkeypatch):
        monkeypatch.setattr(database_service, "insert_to_database", True)

        class _BadFactory:
            def __call__(self):
                raise RuntimeError("db down")

        database_service._async_session_factory = _BadFactory()
        # Should not propagate
        await database_service.insert_log(_make_entry())
