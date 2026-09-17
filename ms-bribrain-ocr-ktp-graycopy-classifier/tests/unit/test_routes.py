"""Unit tests for src.api.routes module"""

import asyncio
import io
import json
import os
from unittest.mock import patch, MagicMock, AsyncMock

import httpx
import pytest
from PIL import Image
from fastapi import FastAPI, UploadFile

from src.middleware.add_requestid import RequestIdMiddleware

from src.api.routes import (
    router,
    _envelope,
    _process_image_sync,
    _check_database,
    _check_model,
    get_executor,
    shutdown_executor,
    set_detection_service,
    register_exception_handlers,
    root,
    liveness,
    health_check,
    readiness_check,
    predict,
)
from src.core.logging import request_id_ctx


# --- Helper tests ---

class TestEnvelope:
    def test_success_envelope(self):
        resp = _envelope(200, "req-1", data={"key": "val"})
        body = json.loads(resp.body.decode())
        assert body["status_code"] == 200
        assert body["status_desc"] == "OK"
        assert body["message"] == "Success"
        assert body["data"] == {"key": "val"}
        assert body["request_id"] == "req-1"
        assert body["error_code"] is None

    def test_error_envelope(self):
        resp = _envelope(400, "req-2", message="Bad", error_code="INVALID")
        body = json.loads(resp.body.decode())
        assert body["status_code"] == 400
        assert body["status_desc"] == "Bad Request"
        assert body["error_code"] == "INVALID"

    def test_unknown_status(self):
        resp = _envelope(418, "req-3")
        body = json.loads(resp.body.decode())
        assert body["status_desc"] == "Unknown"

    def test_empty_data_default(self):
        resp = _envelope(200, "req-4")
        body = json.loads(resp.body.decode())
        assert body["data"] == ""


class TestProcessImageSync:
    def test_valid_jpeg(self):
        img = Image.new("RGB", (100, 100), "red")
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        result = _process_image_sync(buf.getvalue())
        assert isinstance(result, Image.Image)
        assert result.mode == "RGB"

    def test_valid_png(self):
        img = Image.new("RGBA", (50, 50), "blue")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        result = _process_image_sync(buf.getvalue())
        assert result.mode == "RGB"

    def test_invalid_bytes(self):
        with pytest.raises(ValueError, match="Invalid image data"):
            _process_image_sync(b"not an image")


class TestExecutor:
    def test_get_executor_creates_and_caches(self):
        import src.api.routes as routes_mod
        routes_mod._executor = None
        executor = get_executor()
        assert executor is not None
        assert get_executor() is executor
        shutdown_executor()

    def test_shutdown_executor(self):
        import src.api.routes as routes_mod
        routes_mod._executor = None
        get_executor()
        shutdown_executor()
        assert routes_mod._executor is None

    def test_shutdown_executor_when_none(self):
        import src.api.routes as routes_mod
        routes_mod._executor = None
        shutdown_executor()  # should not raise


class TestDetectionServiceDI:
    def test_set_and_get(self):
        import src.api.routes as routes_mod
        mock_service = MagicMock()
        mock_service.is_ready.return_value = True
        set_detection_service(mock_service)
        assert routes_mod._detection_service is mock_service
        # cleanup
        set_detection_service(None)


# --- Endpoint tests ---

@pytest.mark.asyncio
class TestRootEndpoint:
    async def test_root(self):
        result = await root()
        assert result["message"] == "Graycopy Detection API"
        assert "predict" in result["endpoints"]


@pytest.mark.asyncio
class TestLivenessEndpoint:
    async def test_alive(self):
        resp = await liveness()
        assert resp.status == "alive"
        assert resp.version == "1.0.0"


@pytest.mark.asyncio
class TestHealthEndpoint:
    async def test_healthy(self):
        import src.api.routes as routes_mod
        mock_svc = MagicMock()
        mock_svc.is_ready.return_value = True
        mock_svc.get_device_info.return_value = "cpu"
        routes_mod._detection_service = mock_svc
        with patch("src.api.routes._check_database", return_value={"status": "up"}):
            resp = await health_check()
            assert resp.status == "healthy"
            assert resp.model_loaded is True
            assert resp.checks is not None
            assert resp.checks["model"]["status"] == "up"
            assert resp.checks["database"]["status"] == "up"
        routes_mod._detection_service = None

    async def test_degraded(self):
        import src.api.routes as routes_mod
        mock_svc = MagicMock()
        mock_svc.is_ready.return_value = True
        mock_svc.get_device_info.return_value = "cpu"
        routes_mod._detection_service = mock_svc
        with patch("src.api.routes._check_database", return_value={"status": "down", "reason": "not initialised"}):
            resp = await health_check()
            assert resp.status == "degraded"
            assert resp.model_loaded is True
        routes_mod._detection_service = None

    async def test_unhealthy_no_model(self):
        import src.api.routes as routes_mod
        routes_mod._detection_service = None
        with patch("src.api.routes._check_database", return_value={"status": "up"}):
            resp = await health_check()
            assert resp.status == "unhealthy"
            assert resp.model_loaded is False

    async def test_unhealthy_not_ready(self):
        import src.api.routes as routes_mod
        mock_svc = MagicMock()
        mock_svc.is_ready.return_value = False
        mock_svc.get_device_info.return_value = "cpu"
        routes_mod._detection_service = mock_svc
        with patch("src.api.routes._check_database", return_value={"status": "up"}):
            resp = await health_check()
            assert resp.status == "unhealthy"
            assert resp.model_loaded is False
        routes_mod._detection_service = None


@pytest.mark.asyncio
class TestReadinessEndpoint:
    async def test_ready(self):
        import src.api.routes as routes_mod
        mock_svc = MagicMock()
        mock_svc.is_ready.return_value = True
        mock_svc.get_device_info.return_value = "cpu"
        routes_mod._detection_service = mock_svc
        with patch("src.api.routes._check_database", return_value={"status": "up"}):
            resp = await readiness_check()
            import json
            body = json.loads(resp.body.decode())
            assert body["status"] == "ready"
            assert resp.status_code == 200
        routes_mod._detection_service = None

    async def test_not_ready(self):
        import src.api.routes as routes_mod
        routes_mod._detection_service = None
        with patch("src.api.routes._check_database", return_value={"status": "up"}):
            resp = await readiness_check()
            assert resp.status_code == 503


@pytest.mark.asyncio
class TestPredictEndpoint:
    def _make_file(self, image_bytes: bytes, filename="test.jpg"):
        mock_file = MagicMock(spec=UploadFile)
        mock_file.filename = filename
        mock_file.read = AsyncMock(return_value=image_bytes)
        return mock_file

    def _make_image_bytes(self):
        img = Image.new("RGB", (100, 100), "green")
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        return buf.getvalue()

    async def test_model_not_loaded(self):
        import src.api.routes as routes_mod
        routes_mod._detection_service = None
        token = request_id_ctx.set("test-req")
        try:
            with patch("src.api.routes.insert_log", new_callable=AsyncMock):
                resp = await predict(self._make_file(b""))
                body = json.loads(resp.body.decode())
                assert resp.status_code == 503
                assert body["error_code"] == "MODEL_NOT_LOADED"
        finally:
            request_id_ctx.reset(token)

    async def test_success(self):
        import src.api.routes as routes_mod
        mock_svc = MagicMock()
        mock_svc.is_ready.return_value = True
        mock_svc.predict.return_value = {
            "prediction": "ORIGINAL",
            "confidence": 0.95,
            "probability_original": 0.95,
            "probability_graycopy": 0.05,
        }
        routes_mod._detection_service = mock_svc

        image_bytes = self._make_image_bytes()
        token = request_id_ctx.set("test-req")
        try:
            with patch("src.api.routes.insert_log", new_callable=AsyncMock), \
                 patch("src.api.routes.get_provider") as mock_provider:
                mock_provider.return_value.get.return_value = 0.5
                resp = await predict(self._make_file(image_bytes))
                body = json.loads(resp.body.decode())
                assert resp.status_code == 200
                assert body["data"]["prediction"] == "ORIGINAL"
        finally:
            request_id_ctx.reset(token)
            routes_mod._detection_service = None

    async def test_image_processing_error(self):
        import src.api.routes as routes_mod
        mock_svc = MagicMock()
        mock_svc.is_ready.return_value = True
        routes_mod._detection_service = mock_svc

        token = request_id_ctx.set("test-req")
        try:
            with patch("src.api.routes.insert_log", new_callable=AsyncMock):
                resp = await predict(self._make_file(b"bad data"))
                body = json.loads(resp.body.decode())
                assert resp.status_code == 400
                assert body["error_code"] == "IMAGE_PROCESSING_FAILED"
        finally:
            request_id_ctx.reset(token)
            routes_mod._detection_service = None

    async def test_prediction_error(self):
        import src.api.routes as routes_mod
        mock_svc = MagicMock()
        mock_svc.is_ready.return_value = True
        mock_svc.predict.side_effect = RuntimeError("Model error")
        routes_mod._detection_service = mock_svc

        image_bytes = self._make_image_bytes()
        token = request_id_ctx.set("test-req")
        try:
            with patch("src.api.routes.insert_log", new_callable=AsyncMock), \
                 patch("src.api.routes.get_provider") as mock_provider:
                mock_provider.return_value.get.return_value = 0.5
                resp = await predict(self._make_file(image_bytes))
                body = json.loads(resp.body.decode())
                assert resp.status_code == 500
                assert body["error_code"] == "PREDICTION_FAILED"
        finally:
            request_id_ctx.reset(token)
            routes_mod._detection_service = None

    async def test_unexpected_error(self):
        """Unexpected error outside specific handlers routes to INTERNAL_ERROR (covers routes.py:357-362)."""
        import src.api.routes as routes_mod
        mock_svc = MagicMock()
        mock_svc.is_ready.return_value = True
        routes_mod._detection_service = mock_svc

        token = request_id_ctx.set("test-req")
        try:
            mock_file = MagicMock(spec=UploadFile)
            mock_file.filename = "test.jpg"
            mock_file.read = AsyncMock(side_effect=RuntimeError("unexpected IO"))
            with patch("src.api.routes.insert_log", new_callable=AsyncMock):
                resp = await predict(mock_file)
                body = json.loads(resp.body.decode())
                assert resp.status_code == 500
                assert body["error_code"] == "INTERNAL_ERROR"
                assert body["message"] == "Internal server error"
        finally:
            request_id_ctx.reset(token)
            routes_mod._detection_service = None


@pytest.mark.asyncio
class TestCheckDatabase:
    """Direct tests for _check_database (covers routes.py:148-158)."""

    async def test_not_initialized(self):
        import src.services.database_service as db_mod
        original = db_mod._async_session_factory
        db_mod._async_session_factory = None
        try:
            result = await _check_database()
            assert result["status"] == "down"
            assert result["reason"] == "not initialised"
        finally:
            db_mod._async_session_factory = original

    async def test_session_query_success(self):
        import src.services.database_service as db_mod
        original = db_mod._async_session_factory

        mock_session = MagicMock()
        mock_session.execute = AsyncMock(return_value=MagicMock())
        mock_factory = MagicMock()
        mock_factory.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_factory.return_value.__aexit__ = AsyncMock(return_value=False)
        db_mod._async_session_factory = mock_factory
        try:
            result = await _check_database()
            assert result["status"] == "up"
            mock_session.execute.assert_awaited_once()
        finally:
            db_mod._async_session_factory = original

    async def test_session_query_failure(self):
        import src.services.database_service as db_mod
        original = db_mod._async_session_factory

        mock_session = MagicMock()
        mock_session.execute = AsyncMock(side_effect=Exception("db offline"))
        mock_factory = MagicMock()
        mock_factory.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_factory.return_value.__aexit__ = AsyncMock(return_value=False)
        db_mod._async_session_factory = mock_factory
        try:
            result = await _check_database()
            assert result["status"] == "down"
            assert "db offline" in result["reason"]
        finally:
            db_mod._async_session_factory = original


class TestCheckModel:
    """Direct tests for _check_model (covers routes.py:167-168 exception branch)."""

    def test_no_service(self):
        import src.api.routes as routes_mod
        routes_mod._detection_service = None
        result = _check_model()
        assert result["status"] == "down"
        assert "model not loaded" in result["reason"]

    def test_service_not_ready(self):
        import src.api.routes as routes_mod
        mock_svc = MagicMock()
        mock_svc.is_ready.return_value = False
        routes_mod._detection_service = mock_svc
        try:
            result = _check_model()
            assert result["status"] == "down"
        finally:
            routes_mod._detection_service = None

    def test_service_raises(self):
        import src.api.routes as routes_mod
        mock_svc = MagicMock()
        mock_svc.is_ready.side_effect = RuntimeError("broken")
        routes_mod._detection_service = mock_svc
        try:
            result = _check_model()
            assert result["status"] == "down"
            assert "broken" in result["reason"]
        finally:
            routes_mod._detection_service = None


# --- Unified error envelope for early rejections (auth / validation) ---

_ENVELOPE_KEYS = {"status_code", "status_desc", "message", "data", "error_code", "errors", "request_id"}


def _app_with_handlers():
    app = FastAPI()
    app.add_middleware(RequestIdMiddleware)
    app.include_router(router)
    register_exception_handlers(app)
    return app


@pytest.mark.asyncio
class TestEarlyRejectionEnvelope:
    """401/403/422 use the unified envelope shape and are logged to OcrKtpLog."""

    async def test_wrong_api_key_uses_envelope_and_logs(self):
        app = _app_with_handlers()
        with patch.dict(os.environ, {"API_KEY": "secret"}):
            with patch("src.api.routes.insert_log", new_callable=AsyncMock) as mock_log:
                transport = httpx.ASGITransport(app=app)
                async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                    resp = await client.post(
                        "/v1/ocr_graycopy",
                        files={"file": ("t.jpg", b"\xff\xd8\xff\xe0", "image/jpeg")},
                        headers={"X-API-Key": "wrong"},
                    )
        assert resp.status_code == 401
        body = resp.json()
        assert _ENVELOPE_KEYS == set(body.keys())
        assert body["error_code"] == "UNAUTHORIZED"
        mock_log.assert_awaited_once()
        assert mock_log.call_args.kwargs["response_code"] == 401

    async def test_missing_file_uses_envelope_and_logs(self):
        app = _app_with_handlers()
        with patch.dict(os.environ, {"API_KEY": "secret"}):
            with patch("src.api.routes.insert_log", new_callable=AsyncMock) as mock_log:
                transport = httpx.ASGITransport(app=app)
                async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                    resp = await client.post(
                        "/v1/ocr_graycopy",
                        headers={"X-API-Key": "secret"},  # file field omitted
                    )
        assert resp.status_code == 422
        body = resp.json()
        assert _ENVELOPE_KEYS == set(body.keys())
        assert body["error_code"] == "VALIDATION_ERROR"
        assert body["errors"]
        mock_log.assert_awaited_once()
        assert mock_log.call_args.kwargs["response_code"] == 422
