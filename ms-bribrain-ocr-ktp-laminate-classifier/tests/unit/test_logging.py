"""Unit tests for src.core.logging module."""

import logging

from src.core.logging import request_id_ctx, RequestIdFilter, setup_logging


class TestRequestIdCtx:
    def test_default_value(self):
        assert request_id_ctx.get() == "-"

    def test_set_and_get(self):
        token = request_id_ctx.set("test-123")
        assert request_id_ctx.get() == "test-123"
        request_id_ctx.reset(token)

    def test_reset(self):
        token = request_id_ctx.set("temp")
        request_id_ctx.reset(token)
        assert request_id_ctx.get() == "-"


class TestRequestIdFilter:
    def test_filter_adds_request_id(self):
        f = RequestIdFilter()
        record = logging.LogRecord("test", logging.INFO, "", 0, "msg", (), None)
        token = request_id_ctx.set("req-abc")
        result = f.filter(record)
        assert result is True
        assert getattr(record, "request_id") == "req-abc"
        request_id_ctx.reset(token)

    def test_filter_default_request_id(self):
        f = RequestIdFilter()
        record = logging.LogRecord("test", logging.INFO, "", 0, "msg", (), None)
        result = f.filter(record)
        assert result is True
        assert getattr(record, "request_id") == "-"


class TestSetupLogging:
    def test_returns_logger(self):
        lgr = setup_logging("test_unit_logger")
        assert isinstance(lgr, logging.Logger)

    def test_no_duplicate_handlers(self):
        name = "test_no_dup"
        lgr1 = setup_logging(name)
        count1 = len(lgr1.handlers)
        lgr2 = setup_logging(name)
        assert len(lgr2.handlers) == count1
        assert lgr1 is lgr2
