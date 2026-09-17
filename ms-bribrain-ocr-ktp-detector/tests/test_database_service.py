"""
Unit tests for services.database_service module
"""
import os
from unittest.mock import patch, MagicMock, AsyncMock
import pytest

from src.services.database_service import (
    init_engine,
    dispose_engine,
    insert_log
)


@pytest.mark.unit
@pytest.mark.asyncio
class TestDatabaseService:
    """Test database service functions"""
    
    async def test_init_engine_success(self, clean_env):
        """Test successful engine initialization"""
        import src.services.database_service as db_service
        db_service._async_engine = None
        db_service._async_session_factory = None

        with patch('src.services.database_service.create_async_engine') as mock_create:
            with patch('src.services.database_service.async_sessionmaker'):
                with patch('src.services.database_service.DATABASE_URL',
                           'postgresql+psycopg://user:pass@localhost/db'):
                    mock_engine = MagicMock()
                    mock_create.return_value = mock_engine

                    await init_engine()

                    # Should have created engine with the configured async driver
                    mock_create.assert_called_once()
                    call_args = mock_create.call_args[0][0]
                    assert 'postgresql+psycopg' in call_args
    
    async def test_init_engine_no_database_url(self, clean_env):
        """Test engine initialization with no DATABASE_URL"""
        if 'DATABASE_URL' in os.environ:
            del os.environ['DATABASE_URL']
        
        # Should not raise exception
        await init_engine()
    
    async def test_init_engine_idempotent(self, clean_env):
        """Test that init_engine is idempotent"""
        # Reset global state
        import src.services.database_service as db_service
        db_service._async_engine = None
        db_service._async_session_factory = None

        with patch('src.services.database_service.create_async_engine') as mock_create:
            with patch('src.services.database_service.async_sessionmaker'):
                with patch('src.services.database_service.DATABASE_URL',
                           'postgresql+psycopg://user:pass@localhost/db'):
                    mock_engine = MagicMock()
                    mock_create.return_value = mock_engine

                    # Call twice
                    await init_engine()
                    await init_engine()

                    # Should only create engine once
                    assert mock_create.call_count == 1
    
    async def test_dispose_engine_success(self, clean_env):
        """Test successful engine disposal"""
        import src.services.database_service as db_service
        db_service._async_engine = None
        db_service._async_session_factory = None

        with patch('src.services.database_service.create_async_engine') as mock_create:
            with patch('src.services.database_service.async_sessionmaker'):
                with patch('src.services.database_service.DATABASE_URL',
                           'postgresql+psycopg://user:pass@localhost/db'):
                    mock_engine = AsyncMock()
                    mock_engine.dispose = AsyncMock()
                    mock_create.return_value = mock_engine

                    await init_engine()
                    await dispose_engine()

                    mock_engine.dispose.assert_called_once()
    
    async def test_dispose_engine_not_initialized(self, clean_env):
        """Test disposing engine when not initialized"""
        import src.services.database_service as db_service
        db_service._async_engine = None
        db_service._async_session_factory = None
        
        # Should not raise exception
        await dispose_engine()
    
    async def test_insert_log_success(self, clean_env):
        """Test successful log insertion"""
        with patch('src.services.database_service._async_session_factory') as mock_factory:
            with patch('src.services.database_service.insert_to_database', True):
                mock_session = AsyncMock()
                mock_factory.return_value.__aenter__.return_value = mock_session
                
                await insert_log(
                    request_id="test-123",
                    response_code=200,
                    payload="test payload",
                    error_message="",
                    result={"detected": True},
                    processing_time=0.5
                )
                
                mock_session.add.assert_called_once()
                mock_session.commit.assert_called_once()
    
    async def test_insert_log_database_disabled(self, clean_env):
        """Test log insertion when database is disabled"""
        with patch('src.services.database_service.insert_to_database', False):
            # Should not raise exception and should return early
            await insert_log(
                request_id="test-123",
                response_code=200,
                payload="test",
                error_message="",
                result=None,
                processing_time=0.5
            )
    
    async def test_insert_log_not_initialized(self, clean_env):
        """Test log insertion when database not initialized"""
        with patch('src.services.database_service._async_session_factory', None):
            with patch('src.services.database_service.insert_to_database', True):
                # Should not raise exception
                await insert_log(
                    request_id="test-123",
                    response_code=200,
                    payload="test",
                    error_message="",
                    result=None,
                    processing_time=0.5
                )
    
    async def test_insert_log_database_error(self, clean_env):
        """Test log insertion with database error"""
        with patch('src.services.database_service._async_session_factory') as mock_factory:
            with patch('src.services.database_service.insert_to_database', True):
                mock_session = AsyncMock()
                mock_session.commit.side_effect = Exception("Database error")
                mock_factory.return_value.__aenter__.return_value = mock_session
                
                # Should not raise exception (error is logged)
                await insert_log(
                    request_id="test-123",
                    response_code=200,
                    payload="test",
                    error_message="",
                    result=None,
                    processing_time=0.5
                )
