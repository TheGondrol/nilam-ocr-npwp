"""Unit tests for src.services.database_services module."""

from unittest.mock import patch, MagicMock, AsyncMock

import pytest

from src.services.database_services import init_engine, dispose_engine, insert_log


@pytest.mark.asyncio
class TestInitEngine:
    async def test_skips_when_no_url(self):
        import src.services.database_services as db_mod
        db_mod._async_engine = None
        db_mod._async_session_factory = None

        with patch.dict("os.environ", {}, clear=True):
            with patch("src.services.database_services.create_async_engine") as mock_create:
                await init_engine()
                mock_create.assert_not_called()

    async def test_creates_engine_with_url(self):
        import src.services.database_services as db_mod
        original_engine = db_mod._async_engine
        original_factory = db_mod._async_session_factory
        db_mod._async_engine = None
        db_mod._async_session_factory = None

        mock_engine = MagicMock()
        mock_conn = MagicMock()
        mock_conn.run_sync = AsyncMock()
        mock_ctx = MagicMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)
        mock_engine.begin.return_value = mock_ctx

        with patch.dict("os.environ", {"DATABASE_URL": "postgresql+asyncpg://test:test@localhost/db"}):
            with patch("src.services.database_services.create_async_engine", return_value=mock_engine):
                await init_engine()
                assert db_mod._async_engine is mock_engine
                assert db_mod._async_session_factory is not None

        db_mod._async_engine = original_engine
        db_mod._async_session_factory = original_factory

    async def test_passes_url_directly_to_engine(self):
        """init_engine passes DATABASE_URL as-is to create_async_engine.
        psycopg3 supports postgresql+psycopg:// natively for async."""
        import src.services.database_services as db_mod
        db_mod._async_engine = None
        db_mod._async_session_factory = None

        mock_engine = MagicMock()
        mock_conn = MagicMock()
        mock_conn.run_sync = AsyncMock()
        mock_ctx = MagicMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)
        mock_engine.begin.return_value = mock_ctx

        url = "postgresql+psycopg://test@localhost/db"
        with patch.dict("os.environ", {"DATABASE_URL": url}):
            with patch("src.services.database_services.create_async_engine", return_value=mock_engine) as mock_create:
                await init_engine()
                call_args = mock_create.call_args
                assert call_args[0][0] == url

        db_mod._async_engine = None
        db_mod._async_session_factory = None

    async def test_raises_on_engine_error(self):
        import src.services.database_services as db_mod
        db_mod._async_engine = None
        db_mod._async_session_factory = None

        with patch.dict("os.environ", {"DATABASE_URL": "postgresql+asyncpg://test@localhost/db"}):
            with patch("src.services.database_services.create_async_engine", side_effect=Exception("conn error")):
                with pytest.raises(Exception, match="conn error"):
                    await init_engine()

        db_mod._async_engine = None
        db_mod._async_session_factory = None


@pytest.mark.asyncio
class TestDisposeEngine:
    async def test_disposes_engine(self):
        import src.services.database_services as db_mod
        mock_engine = MagicMock()
        mock_engine.dispose = AsyncMock()
        db_mod._async_engine = mock_engine

        await dispose_engine()
        mock_engine.dispose.assert_awaited_once()

    async def test_noop_when_no_engine(self):
        import src.services.database_services as db_mod
        db_mod._async_engine = None
        await dispose_engine()  # should not raise


@pytest.mark.asyncio
class TestInsertLog:
    async def test_skips_when_no_factory(self):
        import src.services.database_services as db_mod
        original = db_mod._async_session_factory
        db_mod._async_session_factory = None
        # Should not raise
        await insert_log("req", 200, None, "", None, 1.0)
        db_mod._async_session_factory = original

    async def test_success(self):
        import src.services.database_services as db_mod
        original = db_mod._async_session_factory

        mock_session = MagicMock()
        mock_session.add = MagicMock()
        mock_session.commit = AsyncMock()
        mock_factory = MagicMock()
        mock_factory.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_factory.return_value.__aexit__ = AsyncMock(return_value=False)
        db_mod._async_session_factory = mock_factory

        await insert_log("req-1", 200, {"file": "a.jpg"}, "", {"label": "good"}, 1.5)
        mock_session.add.assert_called_once()
        mock_session.commit.assert_awaited_once()

        db_mod._async_session_factory = original

    async def test_error_logged_not_raised(self):
        import src.services.database_services as db_mod
        original = db_mod._async_session_factory

        mock_session = MagicMock()
        mock_session.add = MagicMock()
        mock_session.commit = AsyncMock(side_effect=Exception("DB Error"))
        mock_factory = MagicMock()
        mock_factory.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_factory.return_value.__aexit__ = AsyncMock(return_value=False)
        db_mod._async_session_factory = mock_factory

        # Should not raise
        await insert_log("req-2", 500, None, "err", None, 0.5)

        db_mod._async_session_factory = original
