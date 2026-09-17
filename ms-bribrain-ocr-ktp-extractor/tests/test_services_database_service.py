"""Unit tests for src.services.database_service module"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from src.services.database_service import (
    init_engine,
    dispose_engine,
    insert_log
)


@pytest.mark.asyncio
class TestInitEngine:
    """Test cases for init_engine function"""

    def setup_method(self):
        """Reset global engine before each test"""
        import src.services.database_service
        src.services.database_service._async_engine = None
        src.services.database_service._async_session_factory = None

    async def test_init_engine_creates_new_engine(self, mock_database_url):
        """Test that init_engine creates a new async engine"""
        import src.services.database_service
        src.services.database_service.DATABASE_URL = mock_database_url

        with patch('src.services.database_service.create_async_engine') as mock_create:
            mock_engine = MagicMock()
            mock_create.return_value = mock_engine

            await init_engine()

            mock_create.assert_called_once()
            assert src.services.database_service._async_engine == mock_engine
            assert src.services.database_service._async_session_factory is not None

    async def test_init_engine_returns_cached_engine(self, mock_database_url):
        """Test that init_engine does not recreate engine if already initialized"""
        import src.services.database_service
        src.services.database_service.DATABASE_URL = mock_database_url

        with patch('src.services.database_service.create_async_engine') as mock_create:
            mock_engine = MagicMock()
            mock_create.return_value = mock_engine

            await init_engine()
            await init_engine()

            # Should only create engine once
            assert mock_create.call_count == 1

    async def test_init_engine_no_database_url(self):
        """Test init_engine does nothing when DATABASE_URL not set"""
        import src.services.database_service
        src.services.database_service.DATABASE_URL = None
        src.services.database_service._async_engine = None

        with patch('src.services.database_service.create_async_engine') as mock_create:
            await init_engine()
            mock_create.assert_not_called()

    async def test_init_engine_with_connection_pool_settings(self, mock_database_url):
        """Test that engine is created with correct pool settings"""
        import src.services.database_service
        src.services.database_service.DATABASE_URL = mock_database_url
        src.services.database_service._async_engine = None

        with patch('src.services.database_service.create_async_engine') as mock_create:
            mock_engine = MagicMock()
            mock_create.return_value = mock_engine

            await init_engine()

            call_kwargs = mock_create.call_args[1]
            assert 'pool_size' in call_kwargs
            assert 'max_overflow' in call_kwargs
            assert call_kwargs['pool_pre_ping'] is True
            assert 'pool_recycle' in call_kwargs


@pytest.mark.asyncio
class TestDisposeEngine:
    """Test cases for dispose_engine function"""

    async def test_dispose_engine_disposes_engine(self):
        """Test that dispose_engine disposes the engine"""
        import src.services.database_service

        mock_engine = MagicMock()
        mock_engine.dispose = AsyncMock()
        src.services.database_service._async_engine = mock_engine
        src.services.database_service._async_session_factory = MagicMock()

        await dispose_engine()

        mock_engine.dispose.assert_awaited_once()
        assert src.services.database_service._async_engine is None
        assert src.services.database_service._async_session_factory is None

    async def test_dispose_engine_no_engine(self):
        """Test dispose_engine when no engine exists"""
        import src.services.database_service
        src.services.database_service._async_engine = None

        # Should not raise error
        await dispose_engine()


@pytest.mark.asyncio
class TestInsertLog:
    """Test cases for insert_log function"""

    async def test_insert_log_database_disabled(self, mock_settings):
        """Test that insert_log skips when database logging is disabled"""
        mock_settings.log_to_database = False

        with patch('src.services.database_service.settings', mock_settings):
            # Should not raise error
            await insert_log(
                request_id="test-123",
                response_code=200,
                payload="test.jpg",
                error_message="",
                result="success",
                processing_time=1.5
            )

    async def test_insert_log_no_session_factory(self, mock_settings):
        """Test insert_log when session factory not initialized"""
        mock_settings.log_to_database = True

        import src.services.database_service
        src.services.database_service._async_session_factory = None

        with patch('src.services.database_service.settings', mock_settings):
            # Should not raise error, just log warning
            await insert_log(
                request_id="test-123",
                response_code=200,
                payload="test.jpg",
                error_message="",
                result="success",
                processing_time=1.5
            )

    async def test_insert_log_success(self, mock_settings, mock_database_url):
        """Test successful log insertion"""
        import src.services.database_service
        mock_settings.log_to_database = True

        mock_session = AsyncMock()
        mock_session_factory = MagicMock()
        mock_session_factory.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_factory.return_value.__aexit__ = AsyncMock(return_value=False)
        src.services.database_service._async_session_factory = mock_session_factory

        with patch('src.services.database_service.settings', mock_settings):
            await insert_log(
                request_id="test-123",
                response_code=200,
                payload="test.jpg",
                error_message="",
                result='[{"text": "Test"}]',
                processing_time=1.5
            )

            mock_session.add.assert_called_once()
            mock_session.commit.assert_awaited_once()

    async def test_insert_log_with_error(self, mock_settings, mock_database_url):
        """Test log insertion with error message"""
        import src.services.database_service
        mock_settings.log_to_database = True

        mock_session = AsyncMock()
        mock_session_factory = MagicMock()
        mock_session_factory.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_factory.return_value.__aexit__ = AsyncMock(return_value=False)
        src.services.database_service._async_session_factory = mock_session_factory

        with patch('src.services.database_service.settings', mock_settings):
            await insert_log(
                request_id="test-456",
                response_code=400,
                payload="bad.jpg",
                error_message="Invalid file type",
                result="",
                processing_time=0.5
            )

    async def test_insert_log_database_error(self, mock_settings):
        """Test insert_log when database operation fails - error is logged but not raised"""
        import src.services.database_service
        mock_settings.log_to_database = True

        mock_session = AsyncMock()
        mock_session.commit = AsyncMock(side_effect=Exception("DB Error"))
        mock_session_factory = MagicMock()
        mock_session_factory.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_factory.return_value.__aexit__ = AsyncMock(return_value=False)
        src.services.database_service._async_session_factory = mock_session_factory

        with patch('src.services.database_service.settings', mock_settings):
            # Should not raise - just logs the error
            await insert_log(
                request_id="test-789",
                response_code=200,
                payload="test.jpg",
                error_message="",
                result="success",
                processing_time=1.5
            )
