"""Unit tests for src.core.logging module."""

import logging
from unittest.mock import patch

from src.core.logging import RequestIdFilter, request_id_ctx, setup_logging


class TestRequestIdCtx:
    def test_default_value(self):
        assert request_id_ctx.get() == "-"

    def test_set_and_get(self):
        token = request_id_ctx.set("test-id-123")
        assert request_id_ctx.get() == "test-id-123"
        request_id_ctx.reset(token)

    def test_reset_restores_default(self):
        token = request_id_ctx.set("temp-id")
        request_id_ctx.reset(token)
        assert request_id_ctx.get() == "-"


class TestRequestIdFilter:
    def test_filter_adds_request_id(self):
        f = RequestIdFilter()
        record = logging.LogRecord("test", logging.INFO, "", 0, "msg", (), None)
        result = f.filter(record)
        assert result is True
        assert hasattr(record, "request_id")

    def test_filter_uses_context_value(self):
        f = RequestIdFilter()
        token = request_id_ctx.set("ctx-id-456")
        record = logging.LogRecord("test", logging.INFO, "", 0, "msg", (), None)
        f.filter(record)
        assert getattr(record, "request_id") == "ctx-id-456"
        request_id_ctx.reset(token)


class TestSetupLogging:
    def test_returns_logger(self):
        logger = setup_logging()
        assert isinstance(logger, logging.Logger)

    def test_root_logger_has_handlers(self):
        """Handlers are added to the root logger; child loggers inherit them."""
        import logging as _logging
        setup_logging()
        assert len(_logging.root.handlers) > 0

    def test_logger_name_from_config(self):
        logger = setup_logging()
        assert logger.name == "quality"

    @patch("src.core.logging.config")
    def test_setup_without_file_handler(self, mock_config):
        mock_config.get.side_effect = lambda key, default=None: {
            "logging.logger_name": "test_logger",
            "logging.level": "DEBUG",
            "logging.format": "%(message)s",
            "logging.file.enabled": False,
        }.get(key, default)

        logger = setup_logging()
        # Should only have console handler
        file_handlers = [
            h for h in logger.handlers
            if hasattr(h, "baseFilename")
        ]
        assert len(file_handlers) == 0
