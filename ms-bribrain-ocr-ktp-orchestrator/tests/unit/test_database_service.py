"""
Unit tests for database service module.
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from sqlalchemy.ext.asyncio import AsyncSession

from src.services.database_service import (
    init_engine,
    dispose_engine,
    insert_log
)


class TestDatabaseService:
    """Tests for database service."""

    @pytest.mark.asyncio
    async def test_init_engine(self):
        """Test database engine initialization."""
        with patch("src.services.database_service.DATABASE_URL", "postgresql+asyncpg://test:test@localhost/test"), \
             patch("src.services.database_service.create_async_engine") as mock_create_engine, \
             patch("src.services.database_service.async_sessionmaker") as mock_sessionmaker:
            
            mock_engine = MagicMock()
            mock_create_engine.return_value = mock_engine
            mock_sessionmaker.return_value = MagicMock()

            await init_engine()

            mock_create_engine.assert_called_once()
            mock_sessionmaker.assert_called_once()

    @pytest.mark.asyncio
    async def test_init_engine_no_database_url(self):
        """Test init_engine with no DATABASE_URL."""
        with patch("src.services.database_service.DATABASE_URL", None):
            # Should not raise exception
            await init_engine()

    @pytest.mark.asyncio
    async def test_dispose_engine(self):
        """Test database engine disposal."""
        mock_engine = AsyncMock()
        mock_engine.dispose = AsyncMock()

        with patch("src.services.database_service._async_engine", mock_engine):
            await dispose_engine()
            mock_engine.dispose.assert_called_once()

    @pytest.mark.asyncio
    async def test_dispose_engine_not_initialized(self):
        """Test dispose_engine when engine is not initialized."""
        with patch("src.services.database_service._async_engine", None):
            # Should not raise exception
            await dispose_engine()

    @pytest.mark.asyncio
    async def test_insert_log_success(self, request_id):
        """Test successful log insertion."""
        mock_session = AsyncMock(spec=AsyncSession)
        mock_session_factory = MagicMock()
        mock_session_factory.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_factory.return_value.__aexit__ = AsyncMock(return_value=None)

        with patch("src.services.database_service.insert_to_database", True), \
             patch("src.services.database_service._async_session_factory", mock_session_factory):
            
            await insert_log(
                request_id=request_id,
                response_code=200,
                payload={"file": "test.jpg"},
                error_message="",
                result={"nama": "Test"},
                processing_time=1.5
            )

            mock_session.add.assert_called_once()
            mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_insert_log_disabled(self, request_id):
        """Test insert_log when database is disabled."""
        with patch("src.services.database_service.insert_to_database", False):
            # Should not raise exception and return early
            await insert_log(
                request_id=request_id,
                response_code=200,
                payload={"file": "test.jpg"},
                error_message="",
                result={"nama": "Test"},
                processing_time=1.5
            )

    @pytest.mark.asyncio
    async def test_insert_log_not_initialized(self, request_id):
        """Test insert_log when database is not initialized."""
        with patch("src.services.database_service.insert_to_database", True), \
             patch("src.services.database_service._async_session_factory", None):
            
            # Should not raise exception
            await insert_log(
                request_id=request_id,
                response_code=200,
                payload={"file": "test.jpg"},
                error_message="",
                result={"nama": "Test"},
                processing_time=1.5
            )

    @pytest.mark.asyncio
    async def test_insert_log_database_error(self, request_id):
        """Test insert_log with database error."""
        mock_session = AsyncMock(spec=AsyncSession)
        mock_session.commit = AsyncMock(side_effect=Exception("Database error"))
        
        mock_session_factory = MagicMock()
        mock_session_factory.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_factory.return_value.__aexit__ = AsyncMock(return_value=None)

        with patch("src.services.database_service.insert_to_database", True), \
             patch("src.services.database_service._async_session_factory", mock_session_factory):
            
            # Should not raise exception, just log error
            await insert_log(
                request_id=request_id,
                response_code=200,
                payload={"file": "test.jpg"},
                error_message="",
                result={"nama": "Test"},
                processing_time=1.5
            )
