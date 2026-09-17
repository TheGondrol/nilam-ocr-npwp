"""Tests for API routes."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock, AsyncMock
import io
from PIL import Image


class TestLivenessEndpoint:
    """Tests for GET /health/live."""

    @pytest.fixture
    def app_client(self):
        from fastapi import FastAPI
        from src.api.routes import router
        app = FastAPI()
        app.include_router(router)
        client = TestClient(app)
        return client

    def test_liveness_returns_200(self, app_client):
        response = app_client.get("/health/live")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "alive"
        assert "version" in data


class TestReadinessEndpoint:
    """Tests for GET /health/ready."""

    @pytest.fixture
    def app_client(self):
        from fastapi import FastAPI
        from src.api.routes import router
        app = FastAPI()
        app.include_router(router)
        client = TestClient(app)
        return client

    @patch('src.api.routes._check_database', new_callable=AsyncMock, return_value={"status": "up"})
    @patch('src.api.routes.get_model')
    @patch('src.api.routes.get_service_device')
    def test_readiness_ready(self, mock_device, mock_model, mock_db, app_client):
        import torch
        mock_model.return_value = MagicMock()
        mock_device.return_value = torch.device("cpu")
        response = app_client.get("/health/ready")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ready"

    @patch('src.api.routes._check_database', new_callable=AsyncMock, return_value={"status": "down", "reason": "not initialised"})
    @patch('src.api.routes.get_model')
    def test_readiness_not_ready(self, mock_model, mock_db, app_client):
        mock_model.return_value = None
        response = app_client.get("/health/ready")
        assert response.status_code == 503
        data = response.json()
        assert data["status"] == "not_ready"


class TestRootEndpoint:
    """Tests for GET /."""

    @pytest.fixture
    def app_client(self):
        from fastapi import FastAPI
        from src.api.routes import router
        app = FastAPI()
        app.include_router(router)
        client = TestClient(app)
        return client

    def test_root_returns_service_info(self, app_client):
        response = app_client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Screen Recapture Detection API"
        assert "endpoints" in data
        assert "liveness" in data["endpoints"]
        assert "readiness" in data["endpoints"]


class TestAPIRoutes:
    """Test cases for API endpoints."""

    @pytest.fixture
    def app_client(self):
        """Create test client without full app startup."""
        from fastapi import FastAPI
        from src.api.routes import router

        app = FastAPI()
        app.include_router(router)

        client = TestClient(app)
        client.headers["X-API-Key"] = "test"
        return client

    def test_root_endpoint(self, app_client):
        """Test root endpoint returns API information."""
        response = app_client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert data["message"] == "Screen Recapture Detection API"
        assert "endpoints" in data
        assert "version" in data

    @patch('src.api.routes._check_database', new_callable=AsyncMock, return_value={"status": "up"})
    @patch('src.api.routes.get_model')
    @patch('src.api.routes.get_service_device')
    def test_health_check_healthy(self, mock_get_device, mock_get_model, mock_db, app_client):
        """Test health check when model is loaded and DB is up."""
        import torch
        mock_get_model.return_value = MagicMock()
        mock_get_device.return_value = torch.device("cpu")

        response = app_client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True
        assert "checks" in data

    @patch('src.api.routes._check_database', new_callable=AsyncMock, return_value={"status": "down", "reason": "connection refused"})
    @patch('src.api.routes.get_model')
    @patch('src.api.routes.get_service_device')
    def test_health_check_degraded(self, mock_get_device, mock_get_model, mock_db, app_client):
        """Test health check when model is loaded but DB is down."""
        import torch
        mock_get_model.return_value = MagicMock()
        mock_get_device.return_value = torch.device("cpu")

        response = app_client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "degraded"
        assert data["model_loaded"] is True

    @patch('src.api.routes._check_database', new_callable=AsyncMock, return_value={"status": "up"})
    @patch('src.api.routes.get_model')
    def test_health_check_unhealthy(self, mock_get_model, mock_db, app_client):
        """Test health check when model is not loaded."""
        mock_get_model.return_value = None

        response = app_client.get("/health")

        assert response.status_code == 503
        data = response.json()
        assert data["status"] == "unhealthy"
        assert data["model_loaded"] is False

    @patch('src.api.routes.insert_log', new_callable=AsyncMock)
    @patch('src.api.routes.get_model')
    @patch('src.api.routes.process_and_predict_sync')
    def test_predict_endpoint_success_original(self, mock_predict, mock_get_model, mock_insert_log, app_client):
        """Test successful prediction for ORIGINAL class."""
        mock_get_model.return_value = MagicMock()
        mock_predict.return_value = (1, 0.25, 0.75, (800, 600))

        img = Image.new('RGB', (100, 100), color='blue')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        img_bytes.seek(0)

        response = app_client.post(
            "/v1/ocr_recapture",
            files={"file": ("test.jpg", img_bytes, "image/jpeg")}
        )

        assert response.status_code == 200
        body = response.json()
        assert body["status_code"] == 200
        data = body["data"]
        assert data["prediction"] == "ORIGINAL"
        assert data["confidence"] == 0.75
        assert data["probability_recaptured"] == 0.25
        assert data["filename"] == "test.jpg"

    @patch('src.api.routes.insert_log', new_callable=AsyncMock)
    @patch('src.api.routes.get_model')
    @patch('src.api.routes.process_and_predict_sync')
    def test_predict_endpoint_success_recaptured(self, mock_predict, mock_get_model, mock_insert_log, app_client):
        """Test successful prediction for RECAPTURED class."""
        mock_get_model.return_value = MagicMock()
        mock_predict.return_value = (0, 0.85, 0.85, (800, 600))

        img = Image.new('RGB', (100, 100), color='red')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        img_bytes.seek(0)

        response = app_client.post(
            "/v1/ocr_recapture",
            files={"file": ("test.jpg", img_bytes, "image/jpeg")}
        )

        assert response.status_code == 200
        body = response.json()
        data = body["data"]
        assert data["prediction"] == "RECAPTURED"
        assert data["confidence"] == 0.85
        assert data["probability_recaptured"] == 0.85

    @patch('src.api.routes.insert_log', new_callable=AsyncMock)
    @patch('src.api.routes.get_model')
    def test_predict_endpoint_model_not_loaded(self, mock_get_model, mock_insert_log, app_client):
        """Test prediction when model is not loaded."""
        mock_get_model.return_value = None

        img = Image.new('RGB', (100, 100))
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        img_bytes.seek(0)

        response = app_client.post(
            "/v1/ocr_recapture",
            files={"file": ("test.jpg", img_bytes, "image/jpeg")}
        )

        assert response.status_code == 503
        body = response.json()
        assert body["error_code"] == "MODEL_NOT_LOADED"

    @patch('src.api.routes.insert_log', new_callable=AsyncMock)
    @patch('src.api.routes.get_model')
    def test_predict_endpoint_invalid_file_type(self, mock_get_model, mock_insert_log, app_client):
        """Test prediction with invalid file type."""
        mock_get_model.return_value = MagicMock()

        text_file = io.BytesIO(b"This is not an image")

        response = app_client.post(
            "/v1/ocr_recapture",
            files={"file": ("test.txt", text_file, "text/plain")}
        )

        assert response.status_code == 400
        body = response.json()
        assert body["error_code"] == "INVALID_FILE_TYPE"

    @patch('src.api.routes.insert_log', new_callable=AsyncMock)
    @patch('src.api.routes.get_model')
    @patch('src.api.routes.process_and_predict_sync')
    def test_predict_endpoint_processing_error(self, mock_predict, mock_get_model, mock_insert_log, app_client):
        """Test prediction when image processing fails."""
        mock_get_model.return_value = MagicMock()
        mock_predict.side_effect = Exception("Failed to process")

        img = Image.new('RGB', (100, 100))
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        img_bytes.seek(0)

        response = app_client.post(
            "/v1/ocr_recapture",
            files={"file": ("test.jpg", img_bytes, "image/jpeg")}
        )

        assert response.status_code == 400
        body = response.json()
        assert body["error_code"] == "IMAGE_PROCESSING_FAILED"

    @patch('src.api.routes.insert_log', new_callable=AsyncMock)
    @patch('src.api.routes.get_model')
    @patch('src.api.routes.process_and_predict_sync')
    def test_predict_endpoint_high_confidence(self, mock_predict, mock_get_model, mock_insert_log, app_client):
        """Test prediction with high confidence score."""
        mock_get_model.return_value = MagicMock()
        mock_predict.return_value = (1, 0.05, 0.95, (1024, 768))

        img = Image.new('RGB', (100, 100))
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        img_bytes.seek(0)

        response = app_client.post(
            "/v1/ocr_recapture",
            files={"file": ("test.jpg", img_bytes, "image/jpeg")}
        )

        assert response.status_code == 200
        body = response.json()
        data = body["data"]
        assert data["confidence"] == 0.95
        assert data["prediction"] == "ORIGINAL"

    @patch('src.api.routes.insert_log', new_callable=AsyncMock)
    @patch('src.api.routes.get_model')
    @patch('src.api.routes.process_and_predict_sync')
    def test_predict_endpoint_with_png_image(self, mock_predict, mock_get_model, mock_insert_log, app_client):
        """Test prediction with PNG image."""
        mock_get_model.return_value = MagicMock()
        mock_predict.return_value = (0, 0.70, 0.70, (500, 500))

        img = Image.new('RGB', (100, 100), color='green')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)

        response = app_client.post(
            "/v1/ocr_recapture",
            files={"file": ("test.png", img_bytes, "image/png")}
        )

        assert response.status_code == 200
        body = response.json()
        data = body["data"]
        assert data["filename"] == "test.png"
        assert "timestamp" in data

    @patch('src.api.routes.insert_log', new_callable=AsyncMock)
    @patch('src.api.routes.get_model')
    @patch('src.api.routes.process_and_predict_sync')
    def test_predict_endpoint_zero_byte_file(self, mock_predict, mock_get_model, mock_insert_log, app_client):
        """Zero-byte upload should be surfaced as image processing failure (not 500)."""
        mock_get_model.return_value = MagicMock()
        mock_predict.side_effect = Exception("cannot identify image file")

        response = app_client.post(
            "/v1/ocr_recapture",
            files={"file": ("empty.jpg", io.BytesIO(b""), "image/jpeg")},
        )

        assert response.status_code == 400
        assert response.json()["error_code"] == "IMAGE_PROCESSING_FAILED"
        mock_insert_log.assert_awaited_once()

    @patch('src.api.routes.insert_log', new_callable=AsyncMock)
    @patch('src.api.routes.get_model')
    @patch('src.api.routes.process_and_predict_sync')
    def test_predict_endpoint_logs_on_success(self, mock_predict, mock_get_model, mock_insert_log, app_client):
        """Finally-block logging must run on 200 with result payload populated."""
        mock_get_model.return_value = MagicMock()
        mock_predict.return_value = (1, 0.1, 0.9, (640, 480))

        img = Image.new('RGB', (100, 100))
        buf = io.BytesIO()
        img.save(buf, format='JPEG')
        buf.seek(0)

        response = app_client.post(
            "/v1/ocr_recapture",
            files={"file": ("x.jpg", buf, "image/jpeg")},
        )

        assert response.status_code == 200
        mock_insert_log.assert_awaited_once()
        kwargs = mock_insert_log.await_args.kwargs
        assert kwargs["response_code"] == 200
        assert kwargs["error_message"] == ""
        assert kwargs["result"] is not None
        assert kwargs["payload"] == {"file_name": "x.jpg"}

    @patch('src.api.routes.insert_log', new_callable=AsyncMock)
    @patch('src.api.routes.get_model')
    @patch('src.api.routes.process_and_predict_sync')
    def test_predict_endpoint_logs_on_processing_error(self, mock_predict, mock_get_model, mock_insert_log, app_client):
        """Finally-block must log with response_code=400 and error_message set when processing fails."""
        mock_get_model.return_value = MagicMock()
        mock_predict.side_effect = RuntimeError("bad pixels")

        img = Image.new('RGB', (10, 10))
        buf = io.BytesIO()
        img.save(buf, format='JPEG')
        buf.seek(0)

        response = app_client.post(
            "/v1/ocr_recapture",
            files={"file": ("err.jpg", buf, "image/jpeg")},
        )

        assert response.status_code == 400
        mock_insert_log.assert_awaited_once()
        kwargs = mock_insert_log.await_args.kwargs
        assert kwargs["response_code"] == 400
        assert "bad pixels" in kwargs["error_message"]
        assert kwargs["result"] is None

    @patch('src.api.routes.insert_log', new_callable=AsyncMock)
    @patch('src.api.routes.get_model')
    def test_predict_endpoint_logs_on_model_not_loaded(self, mock_get_model, mock_insert_log, app_client):
        """Finally-block must still log when model is missing."""
        mock_get_model.return_value = None

        img = Image.new('RGB', (10, 10))
        buf = io.BytesIO()
        img.save(buf, format='JPEG')
        buf.seek(0)

        response = app_client.post(
            "/v1/ocr_recapture",
            files={"file": ("nomodel.jpg", buf, "image/jpeg")},
        )

        assert response.status_code == 503
        mock_insert_log.assert_awaited_once()
        kwargs = mock_insert_log.await_args.kwargs
        assert kwargs["response_code"] == 503
        assert kwargs["error_message"] == "Model not loaded"

    @patch('src.api.routes.insert_log', new_callable=AsyncMock)
    @patch('src.api.routes.get_model')
    @patch('src.api.routes.process_and_predict_sync')
    def test_predict_endpoint_internal_error_logs(self, mock_predict, mock_get_model, mock_insert_log, app_client):
        """Non-processing exceptions (outer except) should produce 500 and still log."""
        mock_get_model.return_value = MagicMock()
        mock_predict.return_value = (1, 0.2, 0.8, (100, 100))
        # Force outer (non-processing) exception by making PredictionResponse
        # construction fail after a successful inference.
        with patch("src.api.routes.PredictionResponse", side_effect=RuntimeError("kaboom")):
            img = Image.new('RGB', (10, 10))
            buf = io.BytesIO()
            img.save(buf, format='JPEG')
            buf.seek(0)
            response = app_client.post(
                "/v1/ocr_recapture",
                files={"file": ("bad.jpg", buf, "image/jpeg")},
            )

        assert response.status_code == 500
        assert response.json()["error_code"] == "INTERNAL_ERROR"
        mock_insert_log.assert_awaited()
        kwargs = mock_insert_log.await_args.kwargs
        assert kwargs["response_code"] == 500
        assert "kaboom" in kwargs["error_message"]

    @patch('src.api.routes.insert_log', new_callable=AsyncMock)
    @patch('src.api.routes.get_model')
    @patch('src.api.routes.process_and_predict_sync')
    def test_predict_endpoint_handles_missing_filename(self, mock_predict, mock_get_model, mock_insert_log, app_client):
        """`file.filename or ""` fallback should produce empty-string filename without crashing."""
        mock_get_model.return_value = MagicMock()
        mock_predict.return_value = (1, 0.2, 0.8, (100, 100))

        img = Image.new('RGB', (10, 10))
        buf = io.BytesIO()
        img.save(buf, format='JPEG')
        buf.seek(0)

        # httpx/Starlette requires a filename tuple element; sending empty string
        # exercises the `or ""` branch.
        response = app_client.post(
            "/v1/ocr_recapture",
            files={"file": ("", buf, "image/jpeg")},
        )

        # Some servers reject empty filename with 422; accept either the
        # 200 path (filename coerced to "") or 422 validation.
        assert response.status_code in (200, 422)


class TestEnvelope:
    """Direct tests for the `_envelope` helper (covers error_code / errors shape)."""

    def test_envelope_data_defaults_to_empty_string(self):
        from src.api.routes import _envelope
        resp = _envelope(200, "rid-1")
        import json
        body = json.loads(resp.body)
        assert body["data"] == ""
        assert body["error_code"] is None
        assert body["errors"] is None
        assert body["status_desc"] == "OK"

    def test_envelope_unknown_status_code_desc(self):
        from src.api.routes import _envelope
        resp = _envelope(418, "rid-2", message="teapot", error_code="TEAPOT", errors=["a"])
        import json
        body = json.loads(resp.body)
        assert body["status_desc"] == "Unknown"
        assert body["error_code"] == "TEAPOT"
        assert body["errors"] == ["a"]


class TestCheckDatabase:
    """Direct tests for `_check_database` helper."""

    @pytest.mark.asyncio
    async def test_check_database_not_initialised(self):
        from src.api import routes as routes_mod
        from src.services import database_services

        original = database_services._async_session_factory
        try:
            database_services._async_session_factory = None
            result = await routes_mod._check_database()
            assert result == {"status": "down", "reason": "not initialised"}
        finally:
            database_services._async_session_factory = original

    @pytest.mark.asyncio
    async def test_check_database_generic_exception(self):
        """Any Exception (not just connection errors) should be captured into reason."""
        from src.api import routes as routes_mod
        from src.services import database_services

        original = database_services._async_session_factory
        try:
            factory = MagicMock()
            factory.return_value.__aenter__ = AsyncMock(side_effect=ValueError("weird"))
            factory.return_value.__aexit__ = AsyncMock(return_value=None)
            database_services._async_session_factory = factory

            result = await routes_mod._check_database()
            assert result["status"] == "down"
            assert "weird" in result["reason"]
        finally:
            database_services._async_session_factory = original

    @pytest.mark.asyncio
    async def test_check_database_up(self):
        from src.api import routes as routes_mod
        from src.services import database_services

        original = database_services._async_session_factory
        try:
            session = AsyncMock()
            session.execute = AsyncMock(return_value=None)
            factory = MagicMock()
            factory.return_value.__aenter__ = AsyncMock(return_value=session)
            factory.return_value.__aexit__ = AsyncMock(return_value=None)
            database_services._async_session_factory = factory

            result = await routes_mod._check_database()
            assert result == {"status": "up"}
        finally:
            database_services._async_session_factory = original


class TestCheckModel:
    """Direct tests for `_check_model`."""

    def test_check_model_down_when_none(self):
        from src.api import routes as routes_mod

        with patch.object(routes_mod, "get_model", return_value=None):
            assert routes_mod._check_model() == {"status": "down", "reason": "model not loaded"}

    def test_check_model_down_on_exception(self):
        from src.api import routes as routes_mod

        with patch.object(routes_mod, "get_model", side_effect=RuntimeError("boom")):
            result = routes_mod._check_model()
            assert result["status"] == "down"
            assert "boom" in result["reason"]

    def test_check_model_up_with_none_device(self):
        """When device helper returns None, _check_model should fall back to 'cpu'."""
        from src.api import routes as routes_mod

        with patch.object(routes_mod, "get_model", return_value=MagicMock()), \
             patch.object(routes_mod, "get_service_device", return_value=None):
            result = routes_mod._check_model()
            assert result == {"status": "up", "device": "cpu"}


class TestRootEndpointDefaults:
    """Root endpoint should still respond when api.version config is missing."""

    @pytest.fixture
    def app_client(self):
        from fastapi import FastAPI
        from src.api.routes import router
        app = FastAPI()
        app.include_router(router)
        return TestClient(app)

    def test_root_uses_default_version_when_missing(self, app_client):
        from src.core.config import config
        original_get = config.get

        def patched_get(key, default=None):
            if key == "api.version":
                return default  # simulate missing config → default kicks in
            return original_get(key, default)

        with patch.object(config, "get", side_effect=patched_get):
            response = app_client.get("/")

        assert response.status_code == 200
        assert response.json()["version"] == "1.0.0"


class TestEarlyRejectionEnvelope:
    """401/403/422 use the unified envelope shape and are logged to OcrKtpLog."""

    _ENVELOPE_KEYS = {"status_code", "status_desc", "message", "data", "error_code", "errors", "request_id"}

    @pytest.fixture
    def app_client(self):
        from fastapi import FastAPI
        from src.api.routes import router, register_exception_handlers
        from src.middleware.add_requestid import RequestIdMiddleware
        app = FastAPI()
        app.add_middleware(RequestIdMiddleware)
        app.include_router(router)
        register_exception_handlers(app)
        return TestClient(app)

    def _jpeg(self):
        img = Image.new("RGB", (10, 10))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        buf.seek(0)
        return buf

    @patch("src.api.routes.insert_log", new_callable=AsyncMock)
    def test_wrong_api_key_uses_envelope_and_logs(self, mock_log, app_client):
        response = app_client.post(
            "/v1/ocr_recapture",
            files={"file": ("test.jpg", self._jpeg(), "image/jpeg")},
            headers={"X-API-Key": "wrong"},
        )
        assert response.status_code == 401
        body = response.json()
        assert self._ENVELOPE_KEYS == set(body.keys())
        assert body["error_code"] == "UNAUTHORIZED"
        mock_log.assert_awaited_once()
        assert mock_log.call_args.kwargs["response_code"] == 401

    @patch("src.api.routes.insert_log", new_callable=AsyncMock)
    def test_missing_file_uses_envelope_and_logs(self, mock_log, app_client):
        response = app_client.post(
            "/v1/ocr_recapture",
            headers={"X-API-Key": "test"},  # file field omitted
        )
        assert response.status_code == 422
        body = response.json()
        assert self._ENVELOPE_KEYS == set(body.keys())
        assert body["error_code"] == "VALIDATION_ERROR"
        assert body["errors"]
        mock_log.assert_awaited_once()
        assert mock_log.call_args.kwargs["response_code"] == 422
