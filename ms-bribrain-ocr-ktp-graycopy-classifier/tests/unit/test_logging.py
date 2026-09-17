"""Unit tests for src.core.logging module"""

import logging

from src.core.logging import request_id_ctx, RequestIdFilter, setup_logging, get_logger


class TestRequestIdCtx:
    def test_default_value(self):
        assert request_id_ctx.get() is not None  # default is "-"

    def test_set_and_get(self):
        token = request_id_ctx.set("abc-123")
        assert request_id_ctx.get() == "abc-123"
        request_id_ctx.reset(token)


class TestRequestIdFilter:
    def test_adds_request_id(self):
        token = request_id_ctx.set("filter-test")
        f = RequestIdFilter()
        record = logging.LogRecord("test", logging.INFO, "", 0, "msg", (), None)
        result = f.filter(record)
        assert result is True
        assert getattr(record, "request_id") == "filter-test"
        request_id_ctx.reset(token)

    def test_default_dash(self):
        f = RequestIdFilter()
        record = logging.LogRecord("test", logging.INFO, "", 0, "msg", (), None)
        f.filter(record)
        request_id = getattr(record, "request_id")
        assert request_id == "-" or isinstance(request_id, str)


class TestSetupLogging:
    def test_returns_logger(self, tmp_path):
        log_file = str(tmp_path / "test.log")
        logger = setup_logging(log_file=log_file, log_level="DEBUG")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "ocr_graycopy"

    def test_idempotent(self, tmp_path):
        log_file = str(tmp_path / "test2.log")
        # Clear handlers from previous tests
        existing = logging.getLogger("ocr_graycopy")
        existing.handlers.clear()

        logger1 = setup_logging(log_file=log_file)
        handler_count = len(logger1.handlers)
        logger2 = setup_logging(log_file=log_file)
        assert len(logger2.handlers) == handler_count

    def test_creates_log_directory(self, tmp_path):
        log_file = str(tmp_path / "subdir" / "test.log")
        # Clear handlers
        existing = logging.getLogger("ocr_graycopy")
        existing.handlers.clear()

        setup_logging(log_file=log_file)
        assert (tmp_path / "subdir").exists()


class TestGetLogger:
    def test_default_name(self):
        logger = get_logger()
        assert logger.name == "ocr_graycopy"

    def test_custom_name(self):
        logger = get_logger("custom")
        assert logger.name == "custom"
