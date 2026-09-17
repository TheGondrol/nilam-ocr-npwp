"""Unit tests for src.services.database_service module."""

from unittest.mock import patch, AsyncMock, MagicMock

import src.services.database_service as db_mod


class TestInitEngine:
    async def test_init_creates_engine(self):
        db_mod._async_engine = None
        db_mod._async_session_factory = None
        original_url = db_mod.DATABASE_URL

        with patch.object(db_mod, "DATABASE_URL", "postgresql+asyncpg://test:test@localhost/db"), \
             patch("src.services.database_service.create_async_engine") as mock_engine, \
             patch("src.services.database_service.async_sessionmaker") as mock_session:
            mock_engine.return_value = MagicMock()
            await db_mod.init_engine()
            mock_engine.assert_called_once()
            mock_session.assert_called_once()

        db_mod._async_engine = None
        db_mod._async_session_factory = None
        db_mod.DATABASE_URL = original_url

    async def test_init_skips_if_no_url(self):
        db_mod._async_engine = None
        original_url = db_mod.DATABASE_URL
        db_mod.DATABASE_URL = None
        await db_mod.init_engine()
        assert db_mod._async_engine is None
        db_mod.DATABASE_URL = original_url


class TestDisposeEngine:
    async def test_dispose_engine(self):
        mock_engine = AsyncMock()
        db_mod._async_engine = mock_engine
        db_mod._async_session_factory = MagicMock()
        await db_mod.dispose_engine()
        mock_engine.dispose.assert_awaited_once()
        assert db_mod._async_engine is None
        assert db_mod._async_session_factory is None

    async def test_dispose_when_none(self):
        db_mod._async_engine = None
        await db_mod.dispose_engine()  # Should not raise


class TestInsertLog:
    async def test_skips_when_disabled(self):
        with patch.object(db_mod, "config") as mock_cfg:
            mock_cfg.log_to_database = False
            await db_mod.insert_log("r1", 200, {}, "", {}, 0.1)

    async def test_skips_when_no_url(self):
        original_url = db_mod.DATABASE_URL
        db_mod.DATABASE_URL = None
        with patch.object(db_mod, "config") as mock_cfg:
            mock_cfg.log_to_database = True
            await db_mod.insert_log("r1", 200, {}, "", {}, 0.1)
        db_mod.DATABASE_URL = original_url

    async def test_skips_when_no_session_factory(self):
        original = db_mod._async_session_factory
        db_mod._async_session_factory = None
        with patch.object(db_mod, "config") as mock_cfg, \
             patch.object(db_mod, "DATABASE_URL", "postgresql+asyncpg://x"):
            mock_cfg.log_to_database = True
            await db_mod.insert_log("r1", 200, {}, "", {}, 0.1)
        db_mod._async_session_factory = original

    async def test_insert_success(self):
        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_session.add = MagicMock()

        mock_factory = MagicMock(return_value=mock_session)
        db_mod._async_session_factory = mock_factory

        with patch.object(db_mod, "config") as mock_cfg, \
             patch.object(db_mod, "DATABASE_URL", "postgresql+asyncpg://x"):
            mock_cfg.log_to_database = True
            await db_mod.insert_log("r1", 200, {"f": "v"}, "", {"r": "v"}, 0.1)

        mock_session.add.assert_called_once()
        mock_session.commit.assert_awaited_once()
        db_mod._async_session_factory = None

    async def test_insert_handles_exception(self):
        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_session.add = MagicMock()
        mock_session.commit = AsyncMock(side_effect=Exception("db error"))

        mock_factory = MagicMock(return_value=mock_session)
        db_mod._async_session_factory = mock_factory

        with patch.object(db_mod, "config") as mock_cfg, \
             patch.object(db_mod, "DATABASE_URL", "postgresql+asyncpg://x"):
            mock_cfg.log_to_database = True
            await db_mod.insert_log("r1", 200, {}, "", {}, 0.1)  # Should not raise

        db_mod._async_session_factory = None
