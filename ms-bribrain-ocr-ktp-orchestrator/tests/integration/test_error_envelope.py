"""Integration tests for the unified top-level error envelope.

The dependencies and rate limiter build a full envelope; these handlers ensure
auth (401), dependency (400), validation (422) and unknown-route (404/405)
errors are returned as the same top-level envelope as the route endpoints,
rather than nested under FastAPI's ``detail`` key.
"""

from unittest.mock import patch, AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routes import router, register_exception_handlers

ENVELOPE_KEYS = {"status_code", "status_desc", "message", "data", "error_code", "errors", "request_id"}
VALID_REQUEST_ID = "OCR_12345678-1234-1234-1234-123456789abc"


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(router)
    register_exception_handlers(app)
    return TestClient(app)


@pytest.mark.integration
class TestErrorEnvelope:
    def test_wrong_api_key_top_level_envelope(self, client):
        """Wrong API key → 401 as a top-level envelope (not nested under 'detail')."""
        with patch.dict("os.environ", {"ORCHESTRATOR_SERVICE_API": "secret"}):
            resp = client.post(
                "/v1/extract-ocr",
                data={"request_id": VALID_REQUEST_ID},
                files={"file": ("t.jpg", b"\xff\xd8\xff\xe0", "image/jpeg")},
                headers={"X-API-Key": "wrong"},
            )
        assert resp.status_code == 401
        body = resp.json()
        assert ENVELOPE_KEYS == set(body.keys())
        assert "detail" not in body
        assert body["error_code"] == "UNAUTHORIZED"

    def test_missing_file_top_level_envelope(self, client):
        """Missing required file field → 422 as a top-level envelope."""
        with patch.dict("os.environ", {"ORCHESTRATOR_SERVICE_API": "secret"}):
            resp = client.post(
                "/v1/extract-ocr",
                data={"request_id": VALID_REQUEST_ID},
                headers={"X-API-Key": "secret"},
            )
        assert resp.status_code == 422
        body = resp.json()
        assert ENVELOPE_KEYS == set(body.keys())
        assert body["error_code"] == "VALIDATION_ERROR"
        assert body["errors"]

    def test_invalid_request_id_top_level_envelope(self, client):
        """Invalid request_id format → 400 envelope un-nested from HTTPException detail."""
        with patch.dict("os.environ", {"ORCHESTRATOR_SERVICE_API": "secret"}), \
             patch("src.api.routes.insert_log", new_callable=AsyncMock):
            resp = client.post(
                "/v1/extract-ocr",
                data={"request_id": "INVALID_FORMAT"},
                files={"file": ("t.jpg", b"\xff\xd8\xff\xe0", "image/jpeg")},
                headers={"X-API-Key": "secret"},
            )
        assert resp.status_code == 400
        body = resp.json()
        assert ENVELOPE_KEYS == set(body.keys())
        assert "detail" not in body
        assert body["error_code"] == "INVALID_REQUEST_ID"
