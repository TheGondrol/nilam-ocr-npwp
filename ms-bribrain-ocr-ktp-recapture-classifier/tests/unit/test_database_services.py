"""Tests for database services."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


class TestDatabaseServices:
    """Test cases for database service functions."""

    @pytest.mark.asyncio
    async def test_insert_log_database_disabled(self):
        """Test log insertion when database is disabled."""
        from src.services import database_services
        
        # Save original value
        original_value = database_services.insert_to_database
        
        try:
            database_services.insert_to_database = False
            
            # Should return without error
            await database_services.insert_log(
                request_id="test-123",
                response_code=200,
                payload={"test": "data"},
                error_message="",
                result={"prediction": "ORIGINAL"},
                processing_time=0.5
            )
            # No error means success
            assert True
        finally:
            # Restore original value
            database_services.insert_to_database = original_value

    @pytest.mark.asyncio
    async def test_insert_log_no_session_factory(self):
        """Test log insertion when session factory is not initialized."""
        from src.services import database_services
        
        # Save original values
        original_insert = database_services.insert_to_database
        original_factory = database_services._async_session_factory
        
        try:
            database_services.insert_to_database = True
            database_services._async_session_factory = None
            
            # Should return without error
            await database_services.insert_log(
                request_id="test-456",
                response_code=400,
                payload={"file": "test.jpg"},
                error_message="Invalid file type",
                result=None,
                processing_time=0.1
            )
            # No error means success
            assert True
        finally:
            # Restore original values
            database_services.insert_to_database = original_insert
            database_services._async_session_factory = original_factory

    @pytest.mark.asyncio
    async def test_insert_log_with_mock_session(self):
        """Test log insertion with mock session factory."""
        from src.services import database_services
        
        # Save original values
        original_insert = database_services.insert_to_database
        original_factory = database_services._async_session_factory
        
        try:
            database_services.insert_to_database = True
            
            # Create mock session
            mock_session = AsyncMock()
            mock_session.add = MagicMock()
            mock_session.commit = AsyncMock()
            
            # Create mock context manager
            mock_factory = MagicMock()
            mock_factory.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_factory.return_value.__aexit__ = AsyncMock(return_value=None)
            
            database_services._async_session_factory = mock_factory
            
            await database_services.insert_log(
                request_id="test-789",
                response_code=200,
                payload={},
                error_message="",
                result={"prediction": "ORIGINAL"},
                processing_time=0.2
            )
            
            mock_session.add.assert_called_once()
            mock_session.commit.assert_called_once()
        finally:
            # Restore original values
            database_services.insert_to_database = original_insert
            database_services._async_session_factory = original_factory

    @pytest.mark.asyncio
    async def test_init_engine_no_database_url(self):
        """Test engine initialization when DATABASE_URL is not set."""
        from src.services import database_services
        
        # Save original value
        original_url = database_services.DATABASE_URL
        original_engine = database_services._async_engine
        
        try:
            database_services.DATABASE_URL = None
            database_services._async_engine = None
            
            await database_services.init_engine()
            
            # Should not create engine without DATABASE_URL
            assert database_services._async_engine is None
        finally:
            # Restore original values
            database_services.DATABASE_URL = original_url
            database_services._async_engine = original_engine

    @pytest.mark.asyncio
    async def test_dispose_engine_with_no_engine(self):
        """Test engine disposal when no engine exists."""
        from src.services import database_services
        
        # Save original value
        original_engine = database_services._async_engine
        
        try:
            database_services._async_engine = None
            
            # Should not raise error
            await database_services.dispose_engine()
            
            assert database_services._async_engine is None
        finally:
            # Restore original value
            database_services._async_engine = original_engine

    @pytest.mark.asyncio
    async def test_init_engine_is_idempotent(self):
        """Calling init_engine twice should not recreate the engine (guard on `is None`)."""
        from src.services import database_services

        original_url = database_services.DATABASE_URL
        original_engine = database_services._async_engine
        original_factory = database_services._async_session_factory

        try:
            sentinel_engine = MagicMock()
            sentinel_engine.dispose = AsyncMock()
            database_services._async_engine = sentinel_engine
            database_services._async_session_factory = MagicMock()
            database_services.DATABASE_URL = "postgresql+asyncpg://localhost/db"

            with patch.object(database_services, "create_async_engine") as create_mock:
                await database_services.init_engine()
                create_mock.assert_not_called()

            # Engine reference unchanged
            assert database_services._async_engine is sentinel_engine
        finally:
            database_services.DATABASE_URL = original_url
            database_services._async_engine = original_engine
            database_services._async_session_factory = original_factory

    @pytest.mark.asyncio
    async def test_dispose_engine_with_existing_engine(self):
        """Dispose should call engine.dispose() and clear globals."""
        from src.services import database_services

        original_engine = database_services._async_engine
        original_factory = database_services._async_session_factory

        try:
            engine = MagicMock()
            engine.dispose = AsyncMock()
            database_services._async_engine = engine
            database_services._async_session_factory = MagicMock()

            await database_services.dispose_engine()

            engine.dispose.assert_awaited_once()
            assert database_services._async_engine is None
            assert database_services._async_session_factory is None
        finally:
            database_services._async_engine = original_engine
            database_services._async_session_factory = original_factory

    @pytest.mark.asyncio
    async def test_insert_log_handles_session_add_exception(self):
        """Exception in session.add (not commit) must be swallowed and logged."""
        from src.services import database_services

        original_insert = database_services.insert_to_database
        original_factory = database_services._async_session_factory

        try:
            database_services.insert_to_database = True

            mock_session = AsyncMock()
            mock_session.add = MagicMock(side_effect=Exception("add failed"))
            mock_session.commit = AsyncMock()

            mock_factory = MagicMock()
            mock_factory.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_factory.return_value.__aexit__ = AsyncMock(return_value=None)

            database_services._async_session_factory = mock_factory

            await database_services.insert_log(
                request_id="err-add",
                response_code=500,
                payload={},
                error_message="",
                result=None,
                processing_time=0.01,
            )
            mock_session.commit.assert_not_awaited()
        finally:
            database_services.insert_to_database = original_insert
            database_services._async_session_factory = original_factory

    @pytest.mark.asyncio
    async def test_insert_log_with_none_payload_and_result(self):
        """Nullable payload/result columns should be passed through to the model unchanged."""
        from src.services import database_services

        original_insert = database_services.insert_to_database
        original_factory = database_services._async_session_factory

        try:
            database_services.insert_to_database = True

            mock_session = AsyncMock()
            mock_session.add = MagicMock()
            mock_session.commit = AsyncMock()

            factory = MagicMock()
            factory.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            factory.return_value.__aexit__ = AsyncMock(return_value=None)
            database_services._async_session_factory = factory

            await database_services.insert_log(
                request_id="nulls",
                response_code=400,
                payload=None,
                error_message="boom",
                result=None,
                processing_time=0.0,
            )

            log_obj = mock_session.add.call_args.args[0]
            assert log_obj.payload is None
            assert log_obj.result is None
            assert log_obj.error_message == "boom"
        finally:
            database_services.insert_to_database = original_insert
            database_services._async_session_factory = original_factory

    @pytest.mark.asyncio
    async def test_insert_log_handles_exception(self):
        """Test log insertion handles database exceptions gracefully."""
        from src.services import database_services
        
        # Save original values
        original_insert = database_services.insert_to_database
        original_factory = database_services._async_session_factory
        
        try:
            database_services.insert_to_database = True
            
            # Create mock session that raises exception on commit
            mock_session = AsyncMock()
            mock_session.add = MagicMock()
            mock_session.commit = AsyncMock(side_effect=Exception("DB Error"))
            
            # Create mock context manager
            mock_factory = MagicMock()
            mock_factory.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_factory.return_value.__aexit__ = AsyncMock(return_value=None)
            
            database_services._async_session_factory = mock_factory
            
            # Should not raise error
            await database_services.insert_log(
                request_id="test-error",
                response_code=500,
                payload={},
                error_message="",
                result=None,
                processing_time=0.1
            )
            # No error raised means success
            assert True
        finally:
            # Restore original values
            database_services.insert_to_database = original_insert
            database_services._async_session_factory = original_factory


class TestDatabaseSchema:
    """Test cases for database schema models."""

    def test_log_table_model_creation(self):
        """Test OcrKtpLog model creation."""
        from src.schemas.database_schema import OcrKtpLog
        
        log = OcrKtpLog(
            request_id="test-123",
            response_code=200,
            payload={"test": "data"},
            error_message="",
            result={"prediction": "ORIGINAL"},
            processing_time=0.5
        )
        
        assert log.request_id == "test-123"
        assert log.response_code == 200
        assert log.processing_time == 0.5

    def test_log_table_with_none_result(self):
        """Test OcrKtpLog with None result."""
        from src.schemas.database_schema import OcrKtpLog
        
        log = OcrKtpLog(
            request_id="test-456",
            response_code=400,
            payload={},
            error_message="Error",
            result=None,
            processing_time=0.1
        )
        
        assert log.result is None
        assert log.error_message == "Error"
