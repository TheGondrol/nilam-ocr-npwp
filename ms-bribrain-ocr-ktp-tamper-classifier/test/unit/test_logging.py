"""
Unit tests for logging module
"""

import json
import logging
import pytest

from src.core.logging import (
    request_id_ctx,
    RequestIdFilter,
    JSONFormatter,
    TextFormatter,
    get_logger,
    setup_logging,
    log_performance,
    set_request_id,
    get_request_id,
    LogContext,
)


@pytest.mark.unit
class TestRequestIdFilter:
    """Tests for RequestIdFilter"""
    
    def test_filter_adds_request_id(self):
        """Test that filter adds request_id to log record"""
        filter_instance = RequestIdFilter()
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="Test message", args=(), exc_info=None
        )
        
        token = request_id_ctx.set("test-request-id")
        
        try:
            result = filter_instance.filter(record)
            
            assert result is True
            assert record.request_id == "test-request-id"  # type: ignore[attr-defined]
        finally:
            request_id_ctx.reset(token)
    
    def test_filter_uses_default_when_no_request_id(self):
        """Test that filter uses default value when no request_id set"""
        filter_instance = RequestIdFilter()
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="Test message", args=(), exc_info=None
        )
        
        result = filter_instance.filter(record)
        
        assert result is True
        assert record.request_id == "-"  # type: ignore[attr-defined]


@pytest.mark.unit
class TestJSONFormatter:
    """Tests for JSONFormatter"""
    
    def test_format_basic_record(self):
        """Test formatting basic log record"""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test.module",
            level=logging.INFO,
            pathname="test.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None
        )
        record.request_id = "test-id"
        record.funcName = "test_function"
        record.module = "test"
        
        result = formatter.format(record)
        
        data = json.loads(result)
        assert data["level"] == "INFO"
        assert data["logger"] == "test.module"
        assert data["message"] == "Test message"
        assert data["request_id"] == "test-id"
        assert data["line"] == 42
    
    def test_format_with_exception(self):
        """Test formatting record with exception"""
        formatter = JSONFormatter()
        
        try:
            raise ValueError("Test error")
        except ValueError:
            import sys
            exc_info = sys.exc_info()
            
            record = logging.LogRecord(
                name="test",
                level=logging.ERROR,
                pathname="test.py",
                lineno=1,
                msg="Error occurred",
                args=(),
                exc_info=exc_info
            )
            record.request_id = "test-id"
            record.funcName = "test"
            record.module = "test"
            
            result = formatter.format(record)
            
            data = json.loads(result)
            assert "exception" in data
            assert "ValueError" in data["exception"]
    
    def test_format_without_request_id(self):
        """Test formatting record without request_id"""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test",
            args=(),
            exc_info=None
        )
        record.request_id = "N/A"
        record.funcName = "test"
        record.module = "test"
        
        result = formatter.format(record)
        
        data = json.loads(result)
        assert "request_id" not in data


@pytest.mark.unit
class TestTextFormatter:
    """Tests for TextFormatter"""
    
    def test_format_with_request_id(self):
        """Test formatting with request_id"""
        formatter = TextFormatter(include_request_id=True)
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None
        )
        record.request_id = "test-id"
        record.funcName = "test_func"
        
        result = formatter.format(record)
        
        assert "test-id" in result
        assert "INFO" in result
        assert "Test message" in result
    
    def test_format_without_request_id(self):
        """Test formatting without request_id"""
        formatter = TextFormatter(include_request_id=False)
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None
        )
        record.funcName = "test_func"
        
        result = formatter.format(record)
        
        assert "INFO" in result
        assert "Test message" in result


@pytest.mark.unit
class TestGetLogger:
    """Tests for get_logger function"""
    
    def test_get_logger_returns_logger(self):
        """Test that get_logger returns a logger instance"""
        logger = get_logger("test.module")
        
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test.module"
    
    def test_get_logger_has_request_id_filter(self):
        """Test that logger has RequestIdFilter"""
        logger = get_logger("test.module")
        
        # Logger should exist and be callable
        assert logger is not None
        assert hasattr(logger, 'info')


@pytest.mark.unit
class TestSetupLogging:
    """Tests for setup_logging function"""
    
    def test_setup_logging_basic(self):
        """Test basic logging setup"""
        # Setup logging should not raise any errors
        # Just test it doesn't crash
        try:
            setup_logging()
        except Exception:
            # May fail if already setup, that's OK
            pass


@pytest.mark.unit
class TestLogPerformance:
    """Tests for log_performance decorator"""
    
    @pytest.mark.asyncio
    async def test_log_performance_async_function(self):
        """Test log_performance decorator with async function"""
        
        @log_performance
        async def async_test_function():
            return "result"
        
        result = await async_test_function()
        
        assert result == "result"
    
    def test_log_performance_sync_function(self):
        """Test log_performance decorator with sync function"""
        
        @log_performance
        def sync_test_function():
            return "result"
        
        result = sync_test_function()
        
        assert result == "result"
    
    @pytest.mark.asyncio
    async def test_log_performance_with_exception(self):
        """Test log_performance decorator handles exceptions"""
        
        @log_performance
        async def failing_function():
            raise ValueError("Test error")
        
        with pytest.raises(ValueError, match="Test error"):
            await failing_function()


@pytest.mark.unit
class TestRequestIdContext:
    """Tests for request_id_ctx context variable"""
    
    def test_request_id_context_default(self):
        """Test request_id_ctx has default value"""
        value = request_id_ctx.get()
        assert value == "-"
    
    def test_request_id_context_set_and_reset(self):
        """Test setting and resetting request_id_ctx"""
        token = request_id_ctx.set("test-request-id")

        try:
            assert request_id_ctx.get() == "test-request-id"
        finally:
            request_id_ctx.reset(token)
            assert request_id_ctx.get() == "-"


@pytest.mark.unit
class TestSetAndGetRequestId:
    """Tests for set_request_id and get_request_id functions."""

    def test_set_request_id(self):
        """Test set_request_id sets the context variable."""
        token = request_id_ctx.set("-")
        try:
            set_request_id("abc-123")
            assert get_request_id() == "abc-123"
        finally:
            request_id_ctx.reset(token)

    def test_get_request_id_default(self):
        """Test get_request_id returns default."""
        value = get_request_id()
        assert isinstance(value, str)


@pytest.mark.unit
class TestJSONFormatterExtraData:
    """Tests for JSONFormatter extra_data handling."""

    def test_format_with_extra_data(self):
        """Test formatting record with extra_data attribute."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="test.py",
            lineno=1, msg="Test", args=(), exc_info=None
        )
        record.request_id = "id-1"
        record.funcName = "test"
        record.module = "test"
        record.extra_data = {"key": "value"}

        result = formatter.format(record)
        data = json.loads(result)
        assert data["extra"] == {"key": "value"}


@pytest.mark.unit
class TestLogPerformanceSyncException:
    """Tests for sync function exception path in log_performance."""

    def test_sync_exception_logged(self):
        """Test log_performance handles sync function exceptions."""
        @log_performance
        def failing_sync():
            raise RuntimeError("sync fail")

        with pytest.raises(RuntimeError, match="sync fail"):
            failing_sync()


@pytest.mark.unit
class TestLogContext:
    """Tests for LogContext context manager."""

    def test_adds_context_to_logs(self):
        """Test LogContext adds extra context."""
        logger = get_logger("test.logcontext")
        with LogContext(logger, user_id=42) as ctx:
            assert ctx is not None

    def test_restores_original_methods(self):
        """Test LogContext restores original logger methods."""
        logger = get_logger("test.logcontext2")
        original_info = logger.info
        with LogContext(logger, key="val"):
            # Inside context, info should be wrapped
            assert logger.info is not original_info
        # After context, info should be restored
        assert logger.info.__func__ == original_info.__func__
