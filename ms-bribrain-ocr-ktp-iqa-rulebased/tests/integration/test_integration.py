"""Integration tests for IQA Rule-based service."""

import json
from unittest.mock import patch, MagicMock, AsyncMock

import httpx
import pytest

from src.main import app


@pytest.fixture
def transport():
    return httpx.ASGITransport(app=app)  # type: ignore[arg-type]


@pytest.fixture
def mock_quality_result():
    return {
        "low_confidence": False,
        "is_blurry": False,
        "is_glare": False,
        "is_rotated": False,
    }


# ---------------------------------------------------------------------------
# Quality endpoint – success paths
# ---------------------------------------------------------------------------

class TestQualityEndpointSuccess:
    async def test_successful_quality_check(self, transport, image_bytes, ocr_result_str, mock_quality_result):
        with patch("src.api.routes.image_quality", return_value=mock_quality_result), \
             patch("src.api.routes.insert_log", new_callable=AsyncMock):
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post(
                    "/v1/ocr_quality",
                    files={"file": ("test.jpg", image_bytes, "image/jpeg")},
                    data={"ocr_result": ocr_result_str},
                    headers={"X-API-Key": "test"},
                )
        assert resp.status_code == 200
        body = resp.json()
        assert body["status_code"] == 200
        assert body["status_desc"] == "OK"
        assert body["data"]["is_blurry"] is False
        assert body["data"]["is_glare"] is False

    async def test_quality_with_all_issues(self, transport, image_bytes, ocr_result_str):
        result = {
            "low_confidence": True,
            "is_blurry": True,
            "is_glare": True,
            "is_rotated": True,
        }
        with patch("src.api.routes.image_quality", return_value=result), \
             patch("src.api.routes.insert_log", new_callable=AsyncMock):
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post(
                    "/v1/ocr_quality",
                    files={"file": ("test.jpg", image_bytes, "image/jpeg")},
                    data={"ocr_result": ocr_result_str},
                    headers={"X-API-Key": "test"},
                )
        body = resp.json()
        assert body["data"]["low_confidence"] is True
        assert body["data"]["is_blurry"] is True
        assert body["data"]["is_glare"] is True
        assert body["data"]["is_rotated"] is True

    async def test_quality_with_empty_ocr(self, transport, image_bytes, empty_ocr_result_str, mock_quality_result):
        with patch("src.api.routes.image_quality", return_value=mock_quality_result), \
             patch("src.api.routes.insert_log", new_callable=AsyncMock):
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post(
                    "/v1/ocr_quality",
                    files={"file": ("test.jpg", image_bytes, "image/jpeg")},
                    data={"ocr_result": empty_ocr_result_str},
                    headers={"X-API-Key": "test"},
                )
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Quality endpoint – error paths
# ---------------------------------------------------------------------------

class TestQualityEndpointErrors:
    async def test_invalid_json_ocr_result(self, transport, image_bytes):
        with patch("src.api.routes.insert_log", new_callable=AsyncMock):
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post(
                    "/v1/ocr_quality",
                    files={"file": ("test.jpg", image_bytes, "image/jpeg")},
                    data={"ocr_result": "not valid json{{{"},
                    headers={"X-API-Key": "test"},
                )
        assert resp.status_code == 400
        body = resp.json()
        assert body["error_code"] == "INVALID_INPUT"
        assert "Invalid OCR result format" in body["message"]

    async def test_undecodable_image_returns_400(self, transport, ocr_result_str):
        """A non-image upload must be rejected as 400 (not reported as 200 all-clear)."""
        with patch("src.api.routes.insert_log", new_callable=AsyncMock) as mock_log:
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post(
                    "/v1/ocr_quality",
                    files={"file": ("test.jpg", b"definitely not an image", "image/jpeg")},
                    data={"ocr_result": ocr_result_str},
                    headers={"X-API-Key": "test"},
                )
        assert resp.status_code == 400
        body = resp.json()
        assert body["error_code"] == "INVALID_INPUT"
        assert "Invalid image" in body["message"]
        # The bad image is logged with response_code=400, not 200.
        mock_log.assert_awaited_once()
        assert mock_log.call_args.kwargs["response_code"] == 400

    async def test_processing_exception_returns_500(self, transport, image_bytes, ocr_result_str):
        with patch("src.api.routes.image_quality", side_effect=RuntimeError("boom")), \
             patch("src.api.routes.insert_log", new_callable=AsyncMock):
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post(
                    "/v1/ocr_quality",
                    files={"file": ("test.jpg", image_bytes, "image/jpeg")},
                    data={"ocr_result": ocr_result_str},
                    headers={"X-API-Key": "test"},
                )
        assert resp.status_code == 500
        body = resp.json()
        assert body["error_code"] == "INTERNAL_ERROR"

    async def test_missing_file_field(self, transport, ocr_result_str):
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/v1/ocr_quality",
                data={"ocr_result": ocr_result_str},
                headers={"X-API-Key": "test"},
            )
        assert resp.status_code == 422

    async def test_missing_ocr_result_field(self, transport, image_bytes):
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/v1/ocr_quality",
                files={"file": ("test.jpg", image_bytes, "image/jpeg")},
                headers={"X-API-Key": "test"},
            )
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------

class TestAuthentication:
    async def test_missing_api_key_rejected(self, transport, image_bytes, ocr_result_str):
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/v1/ocr_quality",
                files={"file": ("test.jpg", image_bytes, "image/jpeg")},
                data={"ocr_result": ocr_result_str},
            )
        assert resp.status_code in (401, 403)

    async def test_wrong_api_key_rejected(self, transport, image_bytes, ocr_result_str):
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/v1/ocr_quality",
                files={"file": ("test.jpg", image_bytes, "image/jpeg")},
                data={"ocr_result": ocr_result_str},
                headers={"X-API-Key": "wrong-key"},
            )
        assert resp.status_code == 401

    async def test_health_no_auth_required(self, transport):
        with patch("src.main._check_database", return_value={"status": "up"}):
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "healthy"

    async def test_liveness_no_auth_required(self, transport):
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/health/live")
        assert resp.status_code == 200
        assert resp.json()["status"] == "alive"

    async def test_root_no_auth_required(self, transport):
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/")
        assert resp.status_code == 200
        body = resp.json()
        assert "service" in body
        assert body["status"] == "running"


# ---------------------------------------------------------------------------
# Response envelope structure
# ---------------------------------------------------------------------------

class TestResponseEnvelope:
    async def test_success_envelope_has_all_keys(self, transport, image_bytes, ocr_result_str, mock_quality_result):
        with patch("src.api.routes.image_quality", return_value=mock_quality_result), \
             patch("src.api.routes.insert_log", new_callable=AsyncMock):
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post(
                    "/v1/ocr_quality",
                    files={"file": ("test.jpg", image_bytes, "image/jpeg")},
                    data={"ocr_result": ocr_result_str},
                    headers={"X-API-Key": "test"},
                )
        body = resp.json()
        expected_keys = {"status_code", "status_desc", "message", "data", "error_code", "errors", "request_id"}
        assert expected_keys == set(body.keys())
        assert body["error_code"] is None
        assert body["errors"] is None

    async def test_error_envelope_has_all_keys(self, transport, image_bytes):
        with patch("src.api.routes.insert_log", new_callable=AsyncMock):
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post(
                    "/v1/ocr_quality",
                    files={"file": ("test.jpg", image_bytes, "image/jpeg")},
                    data={"ocr_result": "bad json"},
                    headers={"X-API-Key": "test"},
                )
        body = resp.json()
        expected_keys = {"status_code", "status_desc", "message", "data", "error_code", "errors", "request_id"}
        assert expected_keys == set(body.keys())
        assert body["error_code"] == "INVALID_INPUT"


# ---------------------------------------------------------------------------
# Database logging
# ---------------------------------------------------------------------------

class TestDatabaseLogging:
    async def test_insert_log_called_on_success(self, transport, image_bytes, ocr_result_str, mock_quality_result):
        mock_log = AsyncMock()
        with patch("src.api.routes.image_quality", return_value=mock_quality_result), \
             patch("src.api.routes.insert_log", mock_log):
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                await client.post(
                    "/v1/ocr_quality",
                    files={"file": ("test.jpg", image_bytes, "image/jpeg")},
                    data={"ocr_result": ocr_result_str},
                    headers={"X-API-Key": "test"},
                )
        mock_log.assert_awaited_once()
        call_kwargs = mock_log.call_args.kwargs
        assert call_kwargs["response_code"] == 200
        assert call_kwargs["error_message"] == ""

    async def test_insert_log_called_on_error(self, transport, image_bytes):
        mock_log = AsyncMock()
        with patch("src.api.routes.insert_log", mock_log):
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                await client.post(
                    "/v1/ocr_quality",
                    files={"file": ("test.jpg", image_bytes, "image/jpeg")},
                    data={"ocr_result": "bad"},
                    headers={"X-API-Key": "test"},
                )
        mock_log.assert_awaited_once()
        call_kwargs = mock_log.call_args.kwargs
        assert call_kwargs["response_code"] == 400

    async def test_insert_log_called_on_exception(self, transport, image_bytes, ocr_result_str):
        mock_log = AsyncMock()
        with patch("src.api.routes.image_quality", side_effect=Exception("fail")), \
             patch("src.api.routes.insert_log", mock_log):
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                await client.post(
                    "/v1/ocr_quality",
                    files={"file": ("test.jpg", image_bytes, "image/jpeg")},
                    data={"ocr_result": ocr_result_str},
                    headers={"X-API-Key": "test"},
                )
        mock_log.assert_awaited_once()
        call_kwargs = mock_log.call_args.kwargs
        assert call_kwargs["response_code"] == 500


# ---------------------------------------------------------------------------
# Middleware integration
# ---------------------------------------------------------------------------

class TestMiddlewareIntegration:
    async def test_response_has_request_id_header(self, transport, image_bytes, ocr_result_str, mock_quality_result):
        with patch("src.api.routes.image_quality", return_value=mock_quality_result), \
             patch("src.api.routes.insert_log", new_callable=AsyncMock):
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post(
                    "/v1/ocr_quality",
                    files={"file": ("test.jpg", image_bytes, "image/jpeg")},
                    data={"ocr_result": ocr_result_str},
                    headers={"X-API-Key": "test"},
                )
        assert "x-request-id" in resp.headers
        assert len(resp.headers["x-request-id"]) == 36  # UUID

    async def test_request_id_in_envelope_matches_header(self, transport, image_bytes, ocr_result_str, mock_quality_result):
        with patch("src.api.routes.image_quality", return_value=mock_quality_result), \
             patch("src.api.routes.insert_log", new_callable=AsyncMock):
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post(
                    "/v1/ocr_quality",
                    files={"file": ("test.jpg", image_bytes, "image/jpeg")},
                    data={"ocr_result": ocr_result_str},
                    headers={"X-API-Key": "test"},
                )
        body = resp.json()
        assert body["request_id"] == resp.headers["x-request-id"]

    async def test_custom_request_id_preserved(self, transport, image_bytes, ocr_result_str, mock_quality_result):
        with patch("src.api.routes.image_quality", return_value=mock_quality_result), \
             patch("src.api.routes.insert_log", new_callable=AsyncMock):
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post(
                    "/v1/ocr_quality",
                    files={"file": ("test.jpg", image_bytes, "image/jpeg")},
                    data={"ocr_result": ocr_result_str},
                    headers={"X-API-Key": "test", "x-request-id": "my-custom-id"},
                )
        assert resp.headers["x-request-id"] == "my-custom-id"
        assert resp.json()["request_id"] == "my-custom-id"

    async def test_different_requests_different_ids(self, transport, image_bytes, ocr_result_str, mock_quality_result):
        with patch("src.api.routes.image_quality", return_value=mock_quality_result), \
             patch("src.api.routes.insert_log", new_callable=AsyncMock):
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                r1 = await client.post(
                    "/v1/ocr_quality",
                    files={"file": ("test.jpg", image_bytes, "image/jpeg")},
                    data={"ocr_result": ocr_result_str},
                    headers={"X-API-Key": "test"},
                )
                r2 = await client.post(
                    "/v1/ocr_quality",
                    files={"file": ("test.jpg", image_bytes, "image/jpeg")},
                    data={"ocr_result": ocr_result_str},
                    headers={"X-API-Key": "test"},
                )
        assert r1.headers["x-request-id"] != r2.headers["x-request-id"]


# ---------------------------------------------------------------------------
# Info endpoints
# ---------------------------------------------------------------------------

class TestInfoEndpoints:
    async def test_liveness_endpoint(self, transport):
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/health/live")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "alive"
        assert "version" in body

    async def test_readiness_endpoint(self, transport):
        with patch("src.main._check_database", return_value={"status": "up"}):
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/health/ready")
            assert resp.status_code == 200
            body = resp.json()
            assert body["status"] == "ready"
            assert "database" in body["checks"]

    async def test_readiness_not_ready(self, transport):
        with patch("src.main._check_database", return_value={"status": "down", "reason": "not initialised"}):
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/health/ready")
            assert resp.status_code == 503

    async def test_health_endpoint_healthy(self, transport):
        with patch("src.main._check_database", return_value={"status": "up"}):
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/health")
            assert resp.status_code == 200
            body = resp.json()
            assert body["status"] == "healthy"
            assert body["checks"]["database"]["status"] == "up"

    async def test_health_endpoint_unhealthy(self, transport):
        with patch("src.main._check_database", return_value={"status": "down", "reason": "not initialised"}):
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/health")
            assert resp.status_code == 200
            body = resp.json()
            assert body["status"] == "unhealthy"

    async def test_root_endpoint(self, transport):
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "running"
        assert "service" in body
        assert "version" in body
        assert "endpoints" in body
        assert "liveness" in body["endpoints"]
        assert "readiness" in body["endpoints"]


# ---------------------------------------------------------------------------
# Unified error envelope for early rejections (auth / validation)
# ---------------------------------------------------------------------------

_ENVELOPE_KEYS = {"status_code", "status_desc", "message", "data", "error_code", "errors", "request_id"}


class TestEarlyRejectionEnvelope:
    async def test_wrong_api_key_uses_envelope_and_logs(self, transport, image_bytes, ocr_result_str):
        with patch("src.main.insert_log", new_callable=AsyncMock) as mock_log:
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post(
                    "/v1/ocr_quality",
                    files={"file": ("test.jpg", image_bytes, "image/jpeg")},
                    data={"ocr_result": ocr_result_str},
                    headers={"X-API-Key": "wrong-key"},
                )
        assert resp.status_code == 401
        body = resp.json()
        assert _ENVELOPE_KEYS == set(body.keys())
        assert body["status_code"] == 401
        assert body["error_code"] == "UNAUTHORIZED"
        mock_log.assert_awaited_once()
        assert mock_log.call_args.kwargs["response_code"] == 401

    async def test_missing_field_uses_envelope_and_logs(self, transport, image_bytes):
        with patch("src.main.insert_log", new_callable=AsyncMock) as mock_log:
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post(
                    "/v1/ocr_quality",
                    files={"file": ("test.jpg", image_bytes, "image/jpeg")},
                    headers={"X-API-Key": "test"},
                )
        assert resp.status_code == 422
        body = resp.json()
        assert _ENVELOPE_KEYS == set(body.keys())
        assert body["error_code"] == "VALIDATION_ERROR"
        assert body["errors"]  # validation detail populated
        mock_log.assert_awaited_once()
        assert mock_log.call_args.kwargs["response_code"] == 422
