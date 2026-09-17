"""Unit tests for src.services.database_service module"""

import asyncio
from unittest.mock import patch, MagicMock, AsyncMock

import pytest

from src.services.database_service import init_engine, dispose_engine, insert_log


@pytest.mark.asyncio
class TestInitEngine:
    async def test_creates_engine_when_url_set(self):
        import src.services.database_service as db_mod
        db_mod._async_engine = None
        db_mod._async_session_factory = None
        original_url = db_mod.DATABASE_URL
        db_mod.DATABASE_URL = "postgresql+asyncpg://test:test@localhost/db"

        with patch("src.services.database_service.create_async_engine") as mock_create:
            mock_engine = MagicMock()
            mock_create.return_value = mock_engine
            await init_engine()
            mock_create.assert_called_once()
            assert db_mod._async_engine is mock_engine

        db_mod.DATABASE_URL = original_url
        db_mod._async_engine = None
        db_mod._async_session_factory = None

    async def test_skips_when_no_url(self):
        import src.services.database_service as db_mod
        db_mod._async_engine = None
        original_url = db_mod.DATABASE_URL
        db_mod.DATABASE_URL = None

        with patch("src.services.database_service.create_async_engine") as mock_create:
            await init_engine()
            mock_create.assert_not_called()

        db_mod.DATABASE_URL = original_url

    async def test_skips_when_already_initialized(self):
        import src.services.database_service as db_mod
        original_engine = db_mod._async_engine
        db_mod._async_engine = MagicMock()  # pretend engine exists

        with patch("src.services.database_service.create_async_engine") as mock_create:
            await init_engine()
            mock_create.assert_not_called()

        db_mod._async_engine = original_engine


@pytest.mark.asyncio
class TestDisposeEngine:
    async def test_disposes_engine(self):
        import src.services.database_service as db_mod
        mock_engine = MagicMock()
        mock_engine.dispose = AsyncMock()
        db_mod._async_engine = mock_engine
        db_mod._async_session_factory = MagicMock()

        await dispose_engine()

        mock_engine.dispose.assert_awaited_once()
        assert db_mod._async_engine is None
        assert db_mod._async_session_factory is None

    async def test_noop_when_no_engine(self):
        import src.services.database_service as db_mod
        db_mod._async_engine = None
        await dispose_engine()  # should not raise


@pytest.mark.asyncio
class TestInsertLog:
    async def test_skips_when_disabled(self):
        import src.services.database_service as db_mod
        original = db_mod.insert_to_database
        db_mod.insert_to_database = False
        await insert_log("req", 200, "file.jpg", "", "ok", 1.0)
        db_mod.insert_to_database = original

    async def test_skips_when_no_url(self):
        import src.services.database_service as db_mod
        original_flag = db_mod.insert_to_database
        original_url = db_mod.DATABASE_URL
        db_mod.insert_to_database = True
        db_mod.DATABASE_URL = None
        await insert_log("req", 200, "file.jpg", "", "ok", 1.0)
        db_mod.insert_to_database = original_flag
        db_mod.DATABASE_URL = original_url

    async def test_skips_when_no_session_factory(self):
        import src.services.database_service as db_mod
        original_flag = db_mod.insert_to_database
        original_url = db_mod.DATABASE_URL
        original_factory = db_mod._async_session_factory
        db_mod.insert_to_database = True
        db_mod.DATABASE_URL = "postgresql+asyncpg://test"
        db_mod._async_session_factory = None
        await insert_log("req", 200, "file.jpg", "", "ok", 1.0)
        db_mod.insert_to_database = original_flag
        db_mod.DATABASE_URL = original_url
        db_mod._async_session_factory = original_factory

    async def test_success(self):
        import src.services.database_service as db_mod
        original_flag = db_mod.insert_to_database
        original_url = db_mod.DATABASE_URL
        original_factory = db_mod._async_session_factory

        db_mod.insert_to_database = True
        db_mod.DATABASE_URL = "postgresql+asyncpg://test"

        mock_session = MagicMock()
        mock_session.add = MagicMock()
        mock_session.commit = AsyncMock()
        mock_factory = MagicMock()
        mock_factory.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_factory.return_value.__aexit__ = AsyncMock(return_value=False)
        db_mod._async_session_factory = mock_factory

        await insert_log("req-1", 200, "file.jpg", "", "result", 1.5)
        mock_session.add.assert_called_once()
        mock_session.commit.assert_awaited_once()

        db_mod.insert_to_database = original_flag
        db_mod.DATABASE_URL = original_url
        db_mod._async_session_factory = original_factory

    async def test_error_logged_not_raised(self):
        import src.services.database_service as db_mod
        original_flag = db_mod.insert_to_database
        original_url = db_mod.DATABASE_URL
        original_factory = db_mod._async_session_factory

        db_mod.insert_to_database = True
        db_mod.DATABASE_URL = "postgresql+asyncpg://test"

        mock_session = MagicMock()
        mock_session.add = MagicMock()
        mock_session.commit = AsyncMock(side_effect=Exception("DB Error"))
        mock_factory = MagicMock()
        mock_factory.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_factory.return_value.__aexit__ = AsyncMock(return_value=False)
        db_mod._async_session_factory = mock_factory

        # Should not raise
        await insert_log("req-2", 500, "file.jpg", "err", "", 0.5)

        db_mod.insert_to_database = original_flag
        db_mod.DATABASE_URL = original_url
        db_mod._async_session_factory = original_factory
