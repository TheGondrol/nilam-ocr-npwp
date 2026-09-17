"""Unit tests for src.api.routes module."""

import json
from unittest.mock import patch, MagicMock, AsyncMock

import pytest
from fastapi.responses import JSONResponse

from src.api.routes import _envelope


class TestEnvelope:
    def test_success_envelope(self):
        resp = _envelope(200, "req-1", data={"key": "val"})
        assert isinstance(resp, JSONResponse)
        assert resp.status_code == 200
        content = json.loads(bytes(resp.body))
        assert content["status_code"] == 200
        assert content["status_desc"] == "OK"
        assert content["data"]["key"] == "val"
        assert content["request_id"] == "req-1"
        assert content["error_code"] is None

    def test_error_envelope(self):
        resp = _envelope(400, "req-2", message="Bad input", error_code="INVALID_INPUT")
        content = json.loads(bytes(resp.body))
        assert content["status_code"] == 400
        assert content["status_desc"] == "Bad Request"
        assert content["error_code"] == "INVALID_INPUT"

    def test_500_envelope(self):
        resp = _envelope(500, "req-3", message="fail", error_code="ERR")
        content = json.loads(bytes(resp.body))
        assert content["status_code"] == 500
        assert content["status_desc"] == "Internal Server Error"

    def test_unknown_status(self):
        resp = _envelope(418, "req-4")
        content = json.loads(bytes(resp.body))
        assert content["status_desc"] == "Unknown"

    def test_empty_data(self):
        resp = _envelope(200, "req-5")
        content = json.loads(bytes(resp.body))
        assert content["data"] == ""

    def test_errors_field(self):
        resp = _envelope(400, "req-6", errors=["e1", "e2"])
        content = json.loads(bytes(resp.body))
        assert content["errors"] == ["e1", "e2"]

    def test_all_envelope_keys(self):
        resp = _envelope(200, "req-7", data={"x": 1})
        content = json.loads(bytes(resp.body))
        expected = {"status_code", "status_desc", "message", "data", "error_code", "errors", "request_id"}
        assert expected == set(content.keys())
