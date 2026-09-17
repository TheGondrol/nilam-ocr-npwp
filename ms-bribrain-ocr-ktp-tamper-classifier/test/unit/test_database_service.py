"""
Unit tests for database_service
"""

import pytest
from typing import cast
from unittest.mock import AsyncMock, Mock, patch

from src.services.database_service import (
    init_engine,
    dispose_engine,
    insert_log
)


@pytest.mark.unit
class TestInitEngine:
    """Tests for init_engine function"""
    
    @pytest.mark.asyncio
    @patch("src.services.database_service.create_async_engine")
    @patch("src.services.database_service.async_sessionmaker")
    @patch("src.services.database_service.DATABASE_URL", "postgresql+asyncpg://test:test@localhost/testdb")
    async def test_init_engine_creates_engine(self, mock_sessionmaker, mock_create_engine):
        """Test that init_engine creates engine and session factory"""
        mock_engine = Mock()
        mock_create_engine.return_value = mock_engine
        
        # Reset global state
        import src.services.database_service as db_service
        db_service._async_engine = None
        db_service._async_session_factory = None
        
        await init_engine()
        
        mock_create_engine.assert_called_once()
        mock_sessionmaker.assert_called_once()
        assert db_service._async_engine == mock_engine
    
    @pytest.mark.asyncio
    @patch("src.services.database_service.DATABASE_URL", None)
    async def test_init_engine_with_no_database_url(self):
        """Test that init_engine handles missing DATABASE_URL"""
        import src.services.database_service as db_service
        db_service._async_engine = None
        
        await init_engine()
        
        # Should not crash and engine should remain None
        assert db_service._async_engine is None
    
    @pytest.mark.asyncio
    @patch("src.services.database_service.create_async_engine")
    @patch("src.services.database_service.DATABASE_URL", "postgresql+asyncpg://test:test@localhost/testdb")
    async def test_init_engine_not_called_twice(self, mock_create_engine):
        """Test that init_engine doesn't recreate engine if already exists"""
        import src.services.database_service as db_service
        from sqlalchemy.ext.asyncio import AsyncEngine
        db_service._async_engine = cast(AsyncEngine, Mock())  # Simulate already initialized
        
        await init_engine()
        
        # Should not create new engine
        mock_create_engine.assert_not_called()


@pytest.mark.unit
class TestDisposeEngine:
    """Tests for dispose_engine function"""
    
    @pytest.mark.asyncio
    async def test_dispose_engine_disposes_existing_engine(self):
        """Test that dispose_engine disposes existing engine"""
        import src.services.database_service as db_service
        
        mock_engine = AsyncMock()
        from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, AsyncSession
        db_service._async_engine = cast(AsyncEngine, mock_engine)
        db_service._async_session_factory = cast(async_sessionmaker[AsyncSession], Mock())
        
        await dispose_engine()
        
        mock_engine.dispose.assert_called_once()
        assert db_service._async_engine is None
        assert db_service._async_session_factory is None
    
    @pytest.mark.asyncio
    async def test_dispose_engine_with_no_engine(self):
        """Test that dispose_engine handles no existing engine"""
        import src.services.database_service as db_service
        db_service._async_engine = None
        
        # Should not crash
        await dispose_engine()
        
        assert db_service._async_engine is None


@pytest.mark.unit
class TestInsertLog:
    """Tests for insert_log function"""
    
    @pytest.mark.asyncio
    @patch("src.services.database_service.insert_to_database", False)
    async def test_insert_log_skips_when_disabled(self):
        """Test that insert_log skips insertion when database is disabled"""
        # Should not raise any errors
        await insert_log(
            request_id="test-123",
            response_code=200,
            payload={"test": "data"},
            error_message="",
            result={"prediction": "authentic"},
            processing_time=100.5
        )
        # No assertions needed, just verifying it doesn't crash
    
    @pytest.mark.asyncio
    @patch("src.services.database_service.insert_to_database", True)
    async def test_insert_log_skips_when_not_initialized(self):
        """Test that insert_log skips when session factory not initialized"""
        import src.services.database_service as db_service
        db_service._async_session_factory = None
        
        # Should not raise any errors
        await insert_log(
            request_id="test-123",
            response_code=200,
            payload={"test": "data"},
            error_message="",
            result={"prediction": "authentic"},
            processing_time=100.5
        )
    
    @pytest.mark.asyncio
    @patch("src.services.database_service.insert_to_database", True)
    async def test_insert_log_inserts_to_database(self):
        """Test that insert_log inserts data to database"""
        import src.services.database_service as db_service
        
        # Create mock session with proper async methods
        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session.add = Mock()  # add is synchronous
        mock_session.commit = AsyncMock()  # commit is async
        
        # Create mock session factory
        from sqlalchemy.ext.asyncio import async_sessionmaker, AsyncSession
        mock_session_factory = Mock(return_value=mock_session)
        db_service._async_session_factory = cast(async_sessionmaker[AsyncSession], mock_session_factory)
        
        await insert_log(
            request_id="test-123",
            response_code=200,
            payload={"test": "data"},
            error_message="",
            result={"prediction": "authentic"},
            processing_time=100.5
        )
        
        mock_session_factory.assert_called_once()
        mock_session.add.assert_called_once()
        mock_session.commit.assert_called_once()
    
    @pytest.mark.asyncio
    @patch("src.services.database_service.insert_to_database", True)
    async def test_insert_log_handles_database_error(self):
        """Test that insert_log handles database errors gracefully"""
        import src.services.database_service as db_service
        
        # Create mock session that raises error
        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session.add = Mock()  # add is synchronous
        mock_session.commit = AsyncMock(side_effect=Exception("Database error"))
        
        from sqlalchemy.ext.asyncio import async_sessionmaker, AsyncSession
        mock_session_factory = Mock(return_value=mock_session)
        db_service._async_session_factory = cast(async_sessionmaker[AsyncSession], mock_session_factory)
        
        # Should not raise exception (error is logged)
        await insert_log(
            request_id="test-123",
            response_code=200,
            payload={"test": "data"},
            error_message="",
            result={"prediction": "authentic"},
            processing_time=100.5
        )
