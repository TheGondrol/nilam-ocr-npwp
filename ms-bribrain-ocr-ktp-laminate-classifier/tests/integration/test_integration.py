"""Integration tests for laminate classifier service."""

from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch, AsyncMock, MagicMock

import httpx
import numpy as np
import pytest
from PIL import Image

from src.main import app


@pytest.fixture(autouse=True)
def _fresh_executor():
    """Ensure a fresh ThreadPoolExecutor for each test."""
    import src.api.routes as routes_mod
    routes_mod._executor = ThreadPoolExecutor(max_workers=2)
    yield
    routes_mod._executor.shutdown(wait=False)


@pytest.fixture
def transport():
    return httpx.ASGITransport(app=app)


# ---------------------------------------------------------------------------
# Prediction endpoint – success paths
# ---------------------------------------------------------------------------

class TestPredictEndpointSuccess:
    async def test_successful_prediction_laminated(self, transport, image_bytes):
        mock_predictor = MagicMock()
        mock_predictor.is_loaded.return_value = True
        mock_predictor.predict.return_value = ("LAMINATED", 0.3)
        mock_predictor.threshold = 0.5

        with patch("src.api.routes.predictor", mock_predictor), \
             patch("src.api.routes.insert_log", new_callable=AsyncMock):
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post(
                    "/v1/ocr_laminate",
                    files={"file": ("test.jpg", image_bytes, "image/jpeg")},
                    headers={"X-API-Key": "test"},
                )
        assert resp.status_code == 200
        body = resp.json()
        assert body["status_code"] == 200
        assert body["status_desc"] == "OK"
        assert body["data"]["prediction"] == "LAMINATED"
        assert body["data"]["prob"] == 0.3

    async def test_successful_prediction_unlaminated(self, transport, image_bytes):
        mock_predictor = MagicMock()
        mock_predictor.is_loaded.return_value = True
        mock_predictor.predict.return_value = ("UNLAMINATED", 0.85)
        mock_predictor.threshold = 0.5

        with patch("src.api.routes.predictor", mock_predictor), \
             patch("src.api.routes.insert_log", new_callable=AsyncMock):
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post(
                    "/v1/ocr_laminate",
                    files={"file": ("test.jpg", image_bytes, "image/jpeg")},
                    headers={"X-API-Key": "test"},
                )
        body = resp.json()
        assert body["data"]["prediction"] == "UNLAMINATED"

    async def test_png_image(self, transport):
        import io
        import numpy as np
        from PIL import Image

        img = Image.fromarray(np.ones((50, 50, 3), dtype=np.uint8) * 200)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        png_bytes = buf.getvalue()

        mock_predictor = MagicMock()
        mock_predictor.is_loaded.return_value = True
        mock_predictor.predict.return_value = ("LAMINATED", 0.2)
        mock_predictor.threshold = 0.5

        with patch("src.api.routes.predictor", mock_predictor), \
             patch("src.api.routes.insert_log", new_callable=AsyncMock):
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post(
                    "/v1/ocr_laminate",
                    files={"file": ("test.png", png_bytes, "image/png")},
                    headers={"X-API-Key": "test"},
                )
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Prediction endpoint – error paths
# ---------------------------------------------------------------------------

class TestPredictEndpointErrors:
    async def test_model_not_loaded_503(self, transport, image_bytes):
        mock_predictor = MagicMock()
        mock_predictor.is_loaded.return_value = False

        with patch("src.api.routes.predictor", mock_predictor), \
             patch("src.api.routes.insert_log", new_callable=AsyncMock):
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post(
                    "/v1/ocr_laminate",
                    files={"file": ("test.jpg", image_bytes, "image/jpeg")},
                    headers={"X-API-Key": "test"},
                )
        assert resp.status_code == 503
        body = resp.json()
        assert body["error_code"] == "MODEL_NOT_LOADED"

    async def test_invalid_file_type_400(self, transport):
        with patch("src.api.routes.predictor") as mock_predictor, \
             patch("src.api.routes.insert_log", new_callable=AsyncMock):
            mock_predictor.is_loaded.return_value = True
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post(
                    "/v1/ocr_laminate",
                    files={"file": ("test.txt", b"hello", "text/plain")},
                    headers={"X-API-Key": "test"},
                )
        assert resp.status_code == 400
        assert resp.json()["error_code"] == "INVALID_FILE_TYPE"

    async def test_file_too_large_413(self, transport, large_image_bytes):
        with patch("src.api.routes.predictor") as mock_predictor, \
             patch("src.api.routes.insert_log", new_callable=AsyncMock):
            mock_predictor.is_loaded.return_value = True
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post(
                    "/v1/ocr_laminate",
                    files={"file": ("big.jpg", large_image_bytes, "image/jpeg")},
                    headers={"X-API-Key": "test"},
                )
        assert resp.status_code == 413
        assert resp.json()["error_code"] == "FILE_TOO_LARGE"

    async def test_corrupted_image_400(self, transport):
        with patch("src.api.routes.predictor") as mock_predictor, \
             patch("src.api.routes.insert_log", new_callable=AsyncMock):
            mock_predictor.is_loaded.return_value = True
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post(
                    "/v1/ocr_laminate",
                    files={"file": ("bad.jpg", b"\xff\xd8\xff\x00corrupt", "image/jpeg")},
                    headers={"X-API-Key": "test"},
                )
        assert resp.status_code == 400
        assert resp.json()["error_code"] == "IMAGE_PROCESSING_FAILED"

    async def test_prediction_exception_500(self, transport, image_bytes):
        mock_predictor = MagicMock()
        mock_predictor.is_loaded.return_value = True
        mock_predictor.predict.side_effect = RuntimeError("model crash")

        with patch("src.api.routes.predictor", mock_predictor), \
             patch("src.api.routes.insert_log", new_callable=AsyncMock):
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post(
                    "/v1/ocr_laminate",
                    files={"file": ("test.jpg", image_bytes, "image/jpeg")},
                    headers={"X-API-Key": "test"},
                )
        assert resp.status_code == 500
        assert resp.json()["error_code"] == "PREDICTION_FAILED"

    async def test_missing_file_422(self, transport):
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/v1/ocr_laminate",
                headers={"X-API-Key": "test"},
            )
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------

class TestAuthentication:
    async def test_missing_api_key(self, transport, image_bytes):
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/v1/ocr_laminate",
                files={"file": ("test.jpg", image_bytes, "image/jpeg")},
            )
        assert resp.status_code in (401, 403)

    async def test_wrong_api_key(self, transport, image_bytes):
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/v1/ocr_laminate",
                files={"file": ("test.jpg", image_bytes, "image/jpeg")},
                headers={"X-API-Key": "wrong-key"},
            )
        assert resp.status_code == 401

    async def test_health_no_auth(self, transport):
        with patch("src.api.routes._check_database", return_value={"status": "up"}):
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/health")
        assert resp.status_code == 200

    async def test_liveness_no_auth(self, transport):
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/health/live")
        assert resp.status_code == 200

    async def test_root_no_auth(self, transport):
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Response envelope
# ---------------------------------------------------------------------------

class TestResponseEnvelope:
    async def test_success_envelope_keys(self, transport, image_bytes):
        mock_predictor = MagicMock()
        mock_predictor.is_loaded.return_value = True
        mock_predictor.predict.return_value = ("LAMINATED", 0.3)
        mock_predictor.threshold = 0.5

        with patch("src.api.routes.predictor", mock_predictor), \
             patch("src.api.routes.insert_log", new_callable=AsyncMock):
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post(
                    "/v1/ocr_laminate",
                    files={"file": ("test.jpg", image_bytes, "image/jpeg")},
                    headers={"X-API-Key": "test"},
                )
        body = resp.json()
        expected = {"status_code", "status_desc", "message", "data", "error_code", "errors", "request_id"}
        assert expected == set(body.keys())
        assert body["error_code"] is None

    async def test_error_envelope_keys(self, transport, image_bytes):
        mock_predictor = MagicMock()
        mock_predictor.is_loaded.return_value = False

        with patch("src.api.routes.predictor", mock_predictor), \
             patch("src.api.routes.insert_log", new_callable=AsyncMock):
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post(
                    "/v1/ocr_laminate",
                    files={"file": ("test.jpg", image_bytes, "image/jpeg")},
                    headers={"X-API-Key": "test"},
                )
        body = resp.json()
        expected = {"status_code", "status_desc", "message", "data", "error_code", "errors", "request_id"}
        assert expected == set(body.keys())
        assert body["error_code"] is not None


# ---------------------------------------------------------------------------
# Database logging
# ---------------------------------------------------------------------------

class TestDatabaseLogging:
    async def test_log_called_on_success(self, transport, image_bytes):
        mock_predictor = MagicMock()
        mock_predictor.is_loaded.return_value = True
        mock_predictor.predict.return_value = ("LAMINATED", 0.3)
        mock_predictor.threshold = 0.5
        mock_log = AsyncMock()

        with patch("src.api.routes.predictor", mock_predictor), \
             patch("src.api.routes.insert_log", mock_log):
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                await client.post(
                    "/v1/ocr_laminate",
                    files={"file": ("test.jpg", image_bytes, "image/jpeg")},
                    headers={"X-API-Key": "test"},
                )
        mock_log.assert_awaited_once()
        kwargs = mock_log.call_args.kwargs
        assert kwargs["response_code"] == 200

    async def test_log_called_on_error(self, transport, image_bytes):
        mock_predictor = MagicMock()
        mock_predictor.is_loaded.return_value = False
        mock_log = AsyncMock()

        with patch("src.api.routes.predictor", mock_predictor), \
             patch("src.api.routes.insert_log", mock_log):
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                await client.post(
                    "/v1/ocr_laminate",
                    files={"file": ("test.jpg", image_bytes, "image/jpeg")},
                    headers={"X-API-Key": "test"},
                )
        mock_log.assert_awaited_once()
        kwargs = mock_log.call_args.kwargs
        assert kwargs["response_code"] == 503

    async def test_log_called_on_exception(self, transport, image_bytes):
        mock_predictor = MagicMock()
        mock_predictor.is_loaded.return_value = True
        mock_predictor.predict.side_effect = Exception("fail")
        mock_log = AsyncMock()

        with patch("src.api.routes.predictor", mock_predictor), \
             patch("src.api.routes.insert_log", mock_log):
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                await client.post(
                    "/v1/ocr_laminate",
                    files={"file": ("test.jpg", image_bytes, "image/jpeg")},
                    headers={"X-API-Key": "test"},
                )
        mock_log.assert_awaited_once()
        kwargs = mock_log.call_args.kwargs
        assert kwargs["response_code"] == 500


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------

class TestMiddlewareIntegration:
    async def test_request_id_header(self, transport, image_bytes):
        mock_predictor = MagicMock()
        mock_predictor.is_loaded.return_value = True
        mock_predictor.predict.return_value = ("LAMINATED", 0.2)
        mock_predictor.threshold = 0.5

        with patch("src.api.routes.predictor", mock_predictor), \
             patch("src.api.routes.insert_log", new_callable=AsyncMock):
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post(
                    "/v1/ocr_laminate",
                    files={"file": ("test.jpg", image_bytes, "image/jpeg")},
                    headers={"X-API-Key": "test"},
                )
        assert "x-request-id" in resp.headers
        assert len(resp.headers["x-request-id"]) == 36

    async def test_request_id_matches_envelope(self, transport, image_bytes):
        mock_predictor = MagicMock()
        mock_predictor.is_loaded.return_value = True
        mock_predictor.predict.return_value = ("LAMINATED", 0.2)
        mock_predictor.threshold = 0.5

        with patch("src.api.routes.predictor", mock_predictor), \
             patch("src.api.routes.insert_log", new_callable=AsyncMock):
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post(
                    "/v1/ocr_laminate",
                    files={"file": ("test.jpg", image_bytes, "image/jpeg")},
                    headers={"X-API-Key": "test"},
                )
        body = resp.json()
        assert body["request_id"] == resp.headers["x-request-id"]

    async def test_custom_request_id(self, transport, image_bytes):
        mock_predictor = MagicMock()
        mock_predictor.is_loaded.return_value = True
        mock_predictor.predict.return_value = ("LAMINATED", 0.2)
        mock_predictor.threshold = 0.5

        with patch("src.api.routes.predictor", mock_predictor), \
             patch("src.api.routes.insert_log", new_callable=AsyncMock):
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post(
                    "/v1/ocr_laminate",
                    files={"file": ("test.jpg", image_bytes, "image/jpeg")},
                    headers={"X-API-Key": "test", "x-request-id": "my-id"},
                )
        assert resp.headers["x-request-id"] == "my-id"
        assert resp.json()["request_id"] == "my-id"


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

    async def test_readiness_endpoint_ready(self, transport):
        mock_predictor = MagicMock()
        mock_predictor.is_loaded.return_value = True
        mock_predictor.get_device_string.return_value = "cpu"
        with patch("src.api.routes.predictor", mock_predictor), \
             patch("src.api.routes._check_database", return_value={"status": "up"}):
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/health/ready")
            assert resp.status_code == 200
            body = resp.json()
            assert body["status"] == "ready"
            assert "database" in body["checks"]

    async def test_readiness_endpoint_not_ready(self, transport):
        mock_predictor = MagicMock()
        mock_predictor.is_loaded.return_value = False
        with patch("src.api.routes.predictor", mock_predictor), \
             patch("src.api.routes._check_database", return_value={"status": "up"}):
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/health/ready")
            assert resp.status_code == 503

    async def test_health_endpoint_healthy(self, transport):
        mock_predictor = MagicMock()
        mock_predictor.is_loaded.return_value = True
        mock_predictor.get_device_string.return_value = "cpu"
        with patch("src.api.routes.predictor", mock_predictor), \
             patch("src.api.routes._check_database", return_value={"status": "up"}):
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/health")
            assert resp.status_code == 200
            body = resp.json()
            assert body["status"] == "healthy"
            assert body["model_loaded"] is True
            assert body["checks"]["model"]["status"] == "up"
            assert body["checks"]["database"]["status"] == "up"

    async def test_health_endpoint_degraded(self, transport):
        mock_predictor = MagicMock()
        mock_predictor.is_loaded.return_value = True
        mock_predictor.get_device_string.return_value = "cpu"
        with patch("src.api.routes.predictor", mock_predictor), \
             patch("src.api.routes._check_database", return_value={"status": "down", "reason": "not initialised"}):
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/health")
            assert resp.status_code == 200
            body = resp.json()
            assert body["status"] == "degraded"

    async def test_health_endpoint_unhealthy(self, transport):
        mock_predictor = MagicMock()
        mock_predictor.is_loaded.return_value = False
        mock_predictor.get_device_string.return_value = "cpu"
        with patch("src.api.routes.predictor", mock_predictor), \
             patch("src.api.routes._check_database", return_value={"status": "up"}):
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/health")
            assert resp.status_code == 200
            body = resp.json()
            assert body["status"] == "unhealthy"
            assert body["model_loaded"] is False

    async def test_root_endpoint(self, transport):
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/")
        assert resp.status_code == 200
        body = resp.json()
        assert "message" in body
        assert "version" in body
        assert "endpoints" in body
        assert "liveness" in body["endpoints"]
        assert "readiness" in body["endpoints"]


# ---------------------------------------------------------------------------
# Unified error envelope for early rejections (auth / validation)
# ---------------------------------------------------------------------------

_ENVELOPE_KEYS = {"status_code", "status_desc", "message", "data", "error_code", "errors", "request_id"}


class TestEarlyRejectionEnvelope:
    """401/403/422 use the unified envelope shape and are logged to OcrKtpLog."""

    async def test_wrong_api_key_uses_envelope_and_logs(self, transport, image_bytes):
        mock_log = AsyncMock()
        with patch("src.api.routes.insert_log", mock_log):
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post(
                    "/v1/ocr_laminate",
                    files={"file": ("test.jpg", image_bytes, "image/jpeg")},
                    headers={"X-API-Key": "wrong-key"},
                )
        assert resp.status_code == 401
        body = resp.json()
        assert _ENVELOPE_KEYS == set(body.keys())
        assert body["error_code"] == "UNAUTHORIZED"
        mock_log.assert_awaited_once()
        assert mock_log.call_args.kwargs["response_code"] == 401

    async def test_missing_file_uses_envelope_and_logs(self, transport):
        mock_log = AsyncMock()
        with patch("src.api.routes.insert_log", mock_log):
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post(
                    "/v1/ocr_laminate",
                    headers={"X-API-Key": "test"},  # file field omitted
                )
        assert resp.status_code == 422
        body = resp.json()
        assert _ENVELOPE_KEYS == set(body.keys())
        assert body["error_code"] == "VALIDATION_ERROR"
        assert body["errors"]
        mock_log.assert_awaited_once()
        assert mock_log.call_args.kwargs["response_code"] == 422
