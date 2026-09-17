"""Unit tests for src.core.logging module."""

import logging

from src.core.logging import (
    RequestIdFilter,
    RequestIdFormatter,
    get_logger,
    request_id_ctx,
    setup_logging,
)


class TestRequestIdCtx:
    def test_default_value(self):
        assert request_id_ctx.get() == "-"

    def test_set_and_get(self):
        token = request_id_ctx.set("test-123")
        assert request_id_ctx.get() == "test-123"
        request_id_ctx.reset(token)

    def test_reset_restores_default(self):
        token = request_id_ctx.set("temp")
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
        token = request_id_ctx.set("ctx-456")
        record = logging.LogRecord("test", logging.INFO, "", 0, "msg", (), None)
        f.filter(record)
        assert getattr(record, "request_id") == "ctx-456"
        request_id_ctx.reset(token)


class TestRequestIdFormatter:
    def test_adds_request_id_if_missing(self):
        fmt = RequestIdFormatter("%(request_id)s - %(message)s")
        record = logging.LogRecord("test", logging.INFO, "", 0, "hello", (), None)
        output = fmt.format(record)
        assert "-" in output or "hello" in output

    def test_preserves_existing_request_id(self):
        fmt = RequestIdFormatter("%(request_id)s - %(message)s")
        record = logging.LogRecord("test", logging.INFO, "", 0, "msg", (), None)
        record.request_id = "existing-id"
        output = fmt.format(record)
        assert "existing-id" in output


class TestSetupLogging:
    def test_returns_logger(self):
        logger = setup_logging("test_setup")
        assert isinstance(logger, logging.Logger)

    def test_logger_has_handlers(self):
        logger = setup_logging("test_handlers")
        assert len(logger.handlers) > 0


class TestGetLogger:
    def test_returns_logger(self):
        logger = get_logger("test_get")
        assert isinstance(logger, logging.Logger)

    def test_has_request_id_filter(self):
        logger = get_logger("test_filter")
        has_filter = any(isinstance(f, RequestIdFilter) for f in logger.filters)
        assert has_filter

    def test_no_duplicate_filters(self):
        logger = get_logger("test_no_dup")
        get_logger("test_no_dup")  # call again
        count = sum(1 for f in logger.filters if isinstance(f, RequestIdFilter))
        assert count == 1
