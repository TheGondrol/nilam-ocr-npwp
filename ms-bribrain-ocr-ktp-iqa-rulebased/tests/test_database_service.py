"""
Unit tests for database_service.
Tests database operations with async mocking.
"""
import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from sqlalchemy.ext.asyncio import AsyncSession

from src.services.database_service import (
    init_engine,
    dispose_engine,
    insert_log
)


class TestInitEngine:
    """Test cases for init_engine function."""
    
    @pytest.mark.asyncio
    @patch('src.services.database_service.DATABASE_URL', 'postgresql+asyncpg://user:pass@localhost/db')
    @patch('src.services.database_service.create_async_engine')
    @patch('src.services.database_service.async_sessionmaker')
    async def test_init_engine_success(self, mock_sessionmaker, mock_create_engine):
        """Test successful engine initialization."""
        mock_engine = AsyncMock()
        mock_create_engine.return_value = mock_engine
        mock_sessionmaker.return_value = MagicMock()
        
        # Reset global state
        import src.services.database_service as db_service
        db_service._async_engine = None
        db_service._async_session_factory = None
        
        await init_engine()
        
        # Verify engine was created
        assert mock_create_engine.called
        assert mock_sessionmaker.called
    
    @pytest.mark.asyncio
    @patch('src.services.database_service.DATABASE_URL', None)
    async def test_init_engine_no_database_url(self):
        """Test engine initialization with no DATABASE_URL."""
        import src.services.database_service as db_service
        db_service._async_engine = None
        db_service._async_session_factory = None
        
        await init_engine()
        
        # Should not create engine without DATABASE_URL
        assert db_service._async_engine is None
    
    @pytest.mark.asyncio
    @patch('src.services.database_service.DATABASE_URL', 'postgresql+asyncpg://localhost/db')
    @patch('src.services.database_service.create_async_engine')
    async def test_init_engine_already_initialized(self, mock_create_engine):
        """Test that engine is not re-initialized if already exists."""
        import src.services.database_service as db_service
        
        # Set engine as already initialized
        mock_existing_engine = AsyncMock()
        db_service._async_engine = mock_existing_engine
        
        await init_engine()
        
        # Should not create new engine
        assert not mock_create_engine.called
        assert db_service._async_engine is mock_existing_engine


class TestDisposeEngine:
    """Test cases for dispose_engine function."""
    
    @pytest.mark.asyncio
    async def test_dispose_engine_success(self):
        """Test successful engine disposal."""
        import src.services.database_service as db_service
        
        # Set up mock engine
        mock_engine = AsyncMock()
        mock_engine.dispose = AsyncMock()
        db_service._async_engine = mock_engine
        db_service._async_session_factory = MagicMock()
        
        await dispose_engine()
        
        # Verify engine was disposed
        assert mock_engine.dispose.called
        assert db_service._async_engine is None
        assert db_service._async_session_factory is None
    
    @pytest.mark.asyncio
    async def test_dispose_engine_no_engine(self):
        """Test disposing when no engine exists."""
        import src.services.database_service as db_service
        db_service._async_engine = None
        db_service._async_session_factory = None
        
        # Should not raise error
        await dispose_engine()
        
        assert db_service._async_engine is None


class TestInsertLog:
    """Test cases for insert_log function."""
    
    @pytest.mark.asyncio
    @patch('src.services.database_service.get_config')
    async def test_insert_log_disabled(self, mock_get_config):
        """Test when database logging is disabled."""
        # Mock config to disable logging
        mock_config = MagicMock()
        mock_config.logging.log_to_database = False
        mock_get_config.return_value = mock_config
        
        await insert_log(
            request_id="test-123",
            response_code=200,
            payload={"test": "data"},
            error_message="",
            result={"success": True},
            processing_time=1.5
        )
        
        # Should return early without attempting database operation
        # No exception should be raised
    
    @pytest.mark.asyncio
    @patch('src.services.database_service.get_config')
    async def test_insert_log_no_session_factory(self, mock_get_config):
        """Test when session factory is not initialized."""
        mock_config = MagicMock()
        mock_config.logging.log_to_database = True
        mock_get_config.return_value = mock_config
        
        import src.services.database_service as db_service
        db_service._async_session_factory = None
        
        await insert_log(
            request_id="test-123",
            response_code=200,
            payload={"test": "data"},
            error_message="",
            result={"success": True},
            processing_time=1.5
        )
        
        # Should return early with warning
        # No exception should be raised
    
    @pytest.mark.asyncio
    @patch('src.services.database_service.get_config')
    @patch('src.services.database_service._async_session_factory')
    async def test_insert_log_success(self, mock_session_factory, mock_get_config):
        """Test successful log insertion."""
        # Mock config
        mock_config = MagicMock()
        mock_config.logging.log_to_database = True
        mock_get_config.return_value = mock_config
        
        # Mock session
        mock_session = AsyncMock(spec=AsyncSession)
        mock_session.add = MagicMock()
        mock_session.commit = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        
        # Configure session factory
        import src.services.database_service as db_service
        db_service._async_session_factory = MagicMock(return_value=mock_session)
        
        await insert_log(
            request_id="test-123",
            response_code=200,
            payload={"file_name": "test.jpg"},
            error_message="",
            result={"low_confidence": False},
            processing_time=1.5
        )
        
        # Verify session operations were called
        assert mock_session.add.called
        assert mock_session.commit.called
    
    @pytest.mark.asyncio
    @patch('src.services.database_service.get_config')
    @patch('src.services.database_service._async_session_factory')
    async def test_insert_log_with_error_message(self, mock_session_factory, mock_get_config):
        """Test log insertion with error message."""
        mock_config = MagicMock()
        mock_config.logging.log_to_database = True
        mock_get_config.return_value = mock_config
        
        mock_session = AsyncMock(spec=AsyncSession)
        mock_session.add = MagicMock()
        mock_session.commit = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        
        import src.services.database_service as db_service
        db_service._async_session_factory = MagicMock(return_value=mock_session)
        
        await insert_log(
            request_id="test-456",
            response_code=500,
            payload={"file_name": "error.jpg"},
            error_message="Processing failed",
            result={},
            processing_time=0.5
        )
        
        assert mock_session.add.called
        assert mock_session.commit.called
    
    @pytest.mark.asyncio
    @patch('src.services.database_service.get_config')
    @patch('src.services.database_service._async_session_factory')
    async def test_insert_log_exception(self, mock_session_factory, mock_get_config):
        """Test log insertion with database exception."""
        mock_config = MagicMock()
        mock_config.logging.log_to_database = True
        mock_get_config.return_value = mock_config
        
        mock_session = AsyncMock(spec=AsyncSession)
        mock_session.add = MagicMock()
        mock_session.commit = AsyncMock(side_effect=Exception("Database error"))
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        
        import src.services.database_service as db_service
        db_service._async_session_factory = MagicMock(return_value=mock_session)
        
        # Should not raise exception, just log error
        await insert_log(
            request_id="test-789",
            response_code=200,
            payload={"file_name": "test.jpg"},
            error_message="",
            result={"success": True},
            processing_time=1.0
        )
        
        # Should handle exception gracefully
    
    @pytest.mark.asyncio
    @patch('src.services.database_service.get_config')
    @patch('src.services.database_service._async_session_factory')
    async def test_insert_log_with_string_result(self, mock_session_factory, mock_get_config):
        """Test log insertion with string result (alternative format)."""
        mock_config = MagicMock()
        mock_config.logging.log_to_database = True
        mock_get_config.return_value = mock_config
        
        mock_session = AsyncMock(spec=AsyncSession)
        mock_session.add = MagicMock()
        mock_session.commit = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        
        import src.services.database_service as db_service
        db_service._async_session_factory = MagicMock(return_value=mock_session)
        
        await insert_log(
            request_id="test-str",
            response_code=200,
            payload={"test": "data"},
            error_message="",
            result="String result",  # String instead of dict
            processing_time=1.0
        )
        
        assert mock_session.add.called
        assert mock_session.commit.called
    
    @pytest.mark.asyncio
    @patch('src.services.database_service.get_config')
    @patch('src.services.database_service._async_session_factory')
    async def test_insert_log_various_response_codes(self, mock_session_factory, mock_get_config):
        """Test log insertion with various response codes."""
        mock_config = MagicMock()
        mock_config.logging.log_to_database = True
        mock_get_config.return_value = mock_config
        
        mock_session = AsyncMock(spec=AsyncSession)
        mock_session.add = MagicMock()
        mock_session.commit = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        
        import src.services.database_service as db_service
        db_service._async_session_factory = MagicMock(return_value=mock_session)
        
        response_codes = [200, 400, 404, 500]
        
        for code in response_codes:
            await insert_log(
                request_id=f"test-{code}",
                response_code=code,
                payload={"test": "data"},
                error_message="" if code == 200 else f"Error {code}",
                result={} if code != 200 else {"success": True},
                processing_time=1.0
            )
        
        # Should handle all response codes
        assert mock_session.add.call_count == len(response_codes)
        assert mock_session.commit.call_count == len(response_codes)
