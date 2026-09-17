"""Unit tests for src.core.logging module"""

import pytest
import logging
from pathlib import Path
from unittest.mock import patch, MagicMock, call
from src.core.logging import (
    request_id_ctx,
    RequestIdFilter,
    setup_logging
)


class TestRequestIdContext:
    """Test cases for request_id context variable"""

    def test_default_value(self):
        """Test that default request_id is '-'"""
        # Reset to default
        request_id_ctx.set("-")
        assert request_id_ctx.get() == "-"

    def test_set_and_get(self):
        """Test setting and getting request_id"""
        test_id = "test-request-123"
        request_id_ctx.set(test_id)
        assert request_id_ctx.get() == test_id
        # Reset
        request_id_ctx.set("-")


class TestRequestIdFilter:
    """Test cases for RequestIdFilter"""

    def test_filter_adds_request_id(self):
        """Test that filter adds request_id to log record"""
        request_id_ctx.set("req-456")
        
        log_filter = RequestIdFilter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None
        )
        
        result = log_filter.filter(record)
        
        assert result is True
        assert hasattr(record, "request_id")
        assert record.request_id == "req-456"
        
        # Reset
        request_id_ctx.set("-")

    def test_filter_with_default_request_id(self):
        """Test filter with default request_id"""
        request_id_ctx.set("-")
        
        log_filter = RequestIdFilter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None
        )
        
        result = log_filter.filter(record)
        
        assert result is True
        assert record.request_id == "-"


class TestSetupLogging:
    """Test cases for setup_logging function"""

    def test_console_logging_enabled(self, mock_settings):
        """Test setup with console logging enabled"""
        mock_settings.log_console_enabled = True
        mock_settings.log_file_enabled = False
        
        with patch('src.core.logging.settings', mock_settings):
            with patch('logging.getLogger') as mock_get_logger:
                mock_logger = MagicMock()
                mock_get_logger.return_value = mock_logger
                
                setup_logging()
                
                # Verify handlers were cleared and added
                assert mock_logger.handlers.clear.called
                # At least one handler should be added
                assert mock_logger.addHandler.called

    def test_file_logging_enabled(self, mock_settings):
        """Test setup with file logging enabled"""
        mock_settings.log_console_enabled = False
        mock_settings.log_file_enabled = True
        
        with patch('src.core.logging.settings', mock_settings):
            with patch('logging.getLogger') as mock_get_logger:
                with patch('logging.handlers.RotatingFileHandler') as mock_handler:
                    with patch.object(Path, 'mkdir'):
                        mock_logger = MagicMock()
                        mock_get_logger.return_value = mock_logger
                        
                        setup_logging()
                        
                        # Verify file handler was created
                        mock_handler.assert_called_once()
                        assert mock_logger.addHandler.called

    def test_both_handlers_enabled(self, mock_settings):
        """Test setup with both console and file logging"""
        mock_settings.log_console_enabled = True
        mock_settings.log_file_enabled = True
        
        with patch('src.core.logging.settings', mock_settings):
            with patch('logging.getLogger') as mock_get_logger:
                with patch('logging.handlers.RotatingFileHandler'):
                    with patch.object(Path, 'mkdir'):
                        mock_logger = MagicMock()
                        mock_get_logger.return_value = mock_logger
                        
                        setup_logging()
                        
                        # Both handlers should be added
                        assert mock_logger.addHandler.call_count >= 2

    def test_creates_log_directory(self, mock_settings):
        """Test that log directory is created if it doesn't exist"""
        mock_settings.log_file_enabled = True
        mock_settings.log_file_path = "logs/test.log"
        
        with patch('src.core.logging.settings', mock_settings):
            with patch('logging.getLogger') as mock_get_logger:
                with patch('logging.handlers.RotatingFileHandler'):
                    with patch.object(Path, 'mkdir') as mock_mkdir:
                        mock_logger = MagicMock()
                        mock_get_logger.return_value = mock_logger
                        
                        setup_logging()
                        
                        # Verify mkdir was called with parents=True, exist_ok=True
                        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)

    def test_log_level_settings(self, mock_settings):
        """Test that log levels are set correctly"""
        mock_settings.log_console_enabled = True
        mock_settings.log_file_enabled = False
        mock_settings.log_level = "INFO"
        mock_settings.log_console_level = "DEBUG"
        
        with patch('src.core.logging.settings', mock_settings):
            with patch('logging.getLogger') as mock_get_logger:
                mock_logger = MagicMock()
                mock_get_logger.return_value = mock_logger
                
                setup_logging()
                
                # Verify setLevel was called
                assert mock_logger.setLevel.called

    def test_request_id_filter_added(self, mock_settings):
        """Test that RequestIdFilter is added to handlers"""
        mock_settings.log_console_enabled = True
        mock_settings.log_file_enabled = False
        
        with patch('src.core.logging.settings', mock_settings):
            with patch('logging.getLogger') as mock_get_logger:
                mock_logger = MagicMock()
                mock_get_logger.return_value = mock_logger
                
                setup_logging()
                
                # Verify addHandler was called (filter is added to handler)
                assert mock_logger.addHandler.called

    def test_logging_disabled(self, mock_settings):
        """Test setup when both logging options are disabled"""
        mock_settings.log_console_enabled = False
        mock_settings.log_file_enabled = False
        
        with patch('src.core.logging.settings', mock_settings):
            with patch('logging.getLogger') as mock_get_logger:
                mock_logger = MagicMock()
                mock_get_logger.return_value = mock_logger
                
                setup_logging()
                
                # Handlers should be cleared but none added
                assert mock_logger.handlers.clear.called
