"""Unit tests for src.api.routes module."""

import json
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch, MagicMock, AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from src.api.routes import _envelope, root, liveness, health_check, readiness_check, set_executor, router


@pytest.fixture(autouse=True)
def _set_api_key():
    with patch.dict("os.environ", {"API_KEY": "test"}):
        yield


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
        assert content["message"] == "Bad input"

    def test_500_envelope(self):
        resp = _envelope(500, "req-3", message="fail", error_code="ERR")
        content = json.loads(bytes(resp.body))
        assert content["status_code"] == 500
        assert content["status_desc"] == "Internal Server Error"

    def test_503_envelope(self):
        resp = _envelope(503, "req-4")
        content = json.loads(bytes(resp.body))
        assert content["status_desc"] == "Service Unavailable"

    def test_unknown_status_desc(self):
        resp = _envelope(418, "req-5")
        content = json.loads(bytes(resp.body))
        assert content["status_desc"] == "Unknown"

    def test_empty_data_defaults_to_empty_string(self):
        resp = _envelope(200, "req-6")
        content = json.loads(bytes(resp.body))
        assert content["data"] == ""

    def test_errors_field(self):
        resp = _envelope(400, "req-7", errors=["e1", "e2"])
        content = json.loads(bytes(resp.body))
        assert content["errors"] == ["e1", "e2"]


class TestSetExecutor:
    def test_sets_executor(self):
        from src.api import routes
        original = routes._executor
        mock_exec = MagicMock()
        set_executor(mock_exec)
        assert routes._executor is mock_exec
        routes._executor = original


class TestRootEndpoint:
    async def test_root_returns_info(self):
        result = await root()
        assert "name" in result
        assert "endpoints" in result
        assert result["endpoints"]["health"] == "/health"


class TestLivenessEndpoint:
    async def test_alive(self):
        resp = await liveness()
        assert resp.status == "alive"
        assert resp.version == "1.0.0"


class TestHealthEndpoint:
    @patch("src.api.routes._check_database", return_value={"status": "up"})
    @patch("src.api.routes.get_device")
    @patch("src.api.routes.get_model")
    async def test_healthy(self, mock_get_model, mock_get_device, _db):
        mock_get_model.return_value = MagicMock()
        mock_get_device.return_value = MagicMock(__str__=lambda s: "cpu")
        result = await health_check()
        assert result.status == "healthy"
        assert result.model_loaded is True
        assert result.checks is not None
        assert result.checks["model"]["status"] == "up"
        assert result.checks["database"]["status"] == "up"

    @patch("src.api.routes._check_database", return_value={"status": "down", "reason": "not initialised"})
    @patch("src.api.routes.get_device")
    @patch("src.api.routes.get_model")
    async def test_degraded(self, mock_get_model, mock_get_device, _db):
        mock_get_model.return_value = MagicMock()
        mock_get_device.return_value = MagicMock(__str__=lambda s: "cpu")
        result = await health_check()
        assert result.status == "degraded"
        assert result.model_loaded is True

    @patch("src.api.routes._check_database", return_value={"status": "up"})
    @patch("src.api.routes.get_device")
    @patch("src.api.routes.get_model")
    async def test_unhealthy_when_no_model(self, mock_get_model, mock_get_device, _db):
        mock_get_model.return_value = None
        mock_get_device.return_value = None
        result = await health_check()
        assert result.status == "unhealthy"
        assert result.model_loaded is False


class TestReadinessEndpoint:
    @patch("src.api.routes._check_database", return_value={"status": "up"})
    @patch("src.api.routes.get_device")
    @patch("src.api.routes.get_model")
    async def test_ready(self, mock_get_model, mock_get_device, _db):
        mock_get_model.return_value = MagicMock()
        mock_get_device.return_value = MagicMock(__str__=lambda s: "cpu")
        resp = await readiness_check()
        body = json.loads(resp.body.decode())
        assert body["status"] == "ready"
        assert resp.status_code == 200

    @patch("src.api.routes._check_database", return_value={"status": "up"})
    @patch("src.api.routes.get_model")
    async def test_not_ready(self, mock_get_model, _db):
        mock_get_model.return_value = None
        resp = await readiness_check()
        assert resp.status_code == 503


class TestFilterQualityEndpoint:
    """Tests for the /filter endpoint using httpx AsyncClient."""

    def _create_app(self):
        app = FastAPI()
        app.include_router(router)
        return app

    @patch("src.api.routes.get_model", return_value=None)
    async def test_model_not_loaded(self, _mock):
        import httpx
        app = self._create_app()
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            crops_json = json.dumps([])
            resp = await client.post(
                "/v1/ocr_qualitydl",
                files={"file": ("test.jpg", b"\xff\xd8\xff\xe0", "image/jpeg")},
                data={"crops": crops_json},
                headers={"X-API-Key": "test"},
            )
        content = resp.json()
        assert content["status_code"] == 503
        assert content["error_code"] == "MODEL_NOT_LOADED"

    @patch("src.api.routes.insert_log", new_callable=AsyncMock)
    @patch("src.api.routes.get_model", return_value=MagicMock())
    async def test_invalid_file_type(self, _model, _log):
        import httpx
        app = self._create_app()
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/v1/ocr_qualitydl",
                files={"file": ("test.txt", b"hello", "text/plain")},
                data={"crops": "[]"},
                headers={"X-API-Key": "test"},
            )
        content = resp.json()
        assert content["status_code"] == 400
        assert content["error_code"] == "INVALID_FILE_TYPE"

    @patch("src.api.routes.insert_log", new_callable=AsyncMock)
    @patch("src.api.routes.get_model", return_value=MagicMock())
    async def test_invalid_crops_json(self, _model, _log):
        import httpx
        app = self._create_app()
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/v1/ocr_qualitydl",
                files={"file": ("test.jpg", b"\xff\xd8\xff\xe0", "image/jpeg")},
                data={"crops": "not json"},
                headers={"X-API-Key": "test"},
            )
        content = resp.json()
        assert content["status_code"] == 400
        assert content["error_code"] == "INVALID_INPUT"

    @patch("src.api.routes.insert_log", new_callable=AsyncMock)
    @patch("src.api.routes.get_model", return_value=MagicMock())
    async def test_crops_not_a_list(self, _model, _log):
        import httpx
        app = self._create_app()
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/v1/ocr_qualitydl",
                files={"file": ("test.jpg", b"\xff\xd8\xff\xe0", "image/jpeg")},
                data={"crops": json.dumps({"not": "a list"})},
                headers={"X-API-Key": "test"},
            )
        content = resp.json()
        assert content["status_code"] == 400
        assert content["error_code"] == "INVALID_INPUT"

    @patch("src.api.routes.insert_log", new_callable=AsyncMock)
    @patch("src.api.routes.get_model", return_value=MagicMock())
    async def test_invalid_crop_schema_returns_400(self, _model, _log):
        """Crop dict missing required bbox field → 400 INVALID_INPUT."""
        import httpx
        app = self._create_app()
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/v1/ocr_qualitydl",
                files={"file": ("test.jpg", b"\xff\xd8\xff\xe0", "image/jpeg")},
                data={"crops": json.dumps([{"text": "no bbox here"}])},
                headers={"X-API-Key": "test"},
            )
        content = resp.json()
        assert content["status_code"] == 400
        assert content["error_code"] == "INVALID_INPUT"

    async def test_missing_api_key_rejected(self):
        """No X-API-Key header → 401 or 403 (security raises before business logic)."""
        import httpx
        app = self._create_app()
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/v1/ocr_qualitydl",
                files={"file": ("test.jpg", b"\xff\xd8\xff\xe0", "image/jpeg")},
                data={"crops": "[]"},
                # No X-API-Key header
            )
        assert resp.status_code in (401, 403)

    @patch("src.api.routes.insert_log", new_callable=AsyncMock)
    @patch("src.api.routes.process_and_classify_sync", side_effect=RuntimeError("model crash"))
    @patch("src.api.routes.get_model", return_value=MagicMock())
    @patch("src.api.routes.config")
    async def test_classification_exception_returns_500(self, mock_cfg, _model, _classify, _log):
        """Unexpected exception in classify → 500 PREDICTION_FAILED."""
        import httpx
        mock_cfg.log_to_database = True
        mock_cfg.get.return_value = "1.0.0"
        executor = ThreadPoolExecutor(max_workers=1)
        set_executor(executor)
        try:
            app = self._create_app()
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post(
                    "/v1/ocr_qualitydl",
                    files={"file": ("test.jpg", b"\xff\xd8\xff\xe0", "image/jpeg")},
                    data={"crops": "[]"},
                    headers={"X-API-Key": "test"},
                )
            content = resp.json()
            assert content["status_code"] == 500
            assert content["error_code"] == "PREDICTION_FAILED"
        finally:
            executor.shutdown(wait=False)
            set_executor(None)  # type: ignore[arg-type]

    @patch("src.api.routes.insert_log", new_callable=AsyncMock)
    @patch("src.api.routes.process_and_classify_sync")
    @patch("src.api.routes.get_model", return_value=MagicMock())
    @patch("src.api.routes.config")
    async def test_log_to_database_false_skips_insert(self, mock_cfg, _model, _classify, mock_log):
        """When log_to_database=False, insert_log is never awaited."""
        import httpx
        from src.schemas.api_schema import ClassificationResponse
        mock_cfg.log_to_database = False
        mock_cfg.get.return_value = "1.0.0"
        mock_cfg.max_image_size_mb = 1
        _classify.return_value = ClassificationResponse(
            label="good", num_bad=0, num_filtered=0,
            total_crops=0, num_failed_crops=0, bad_crop_threshold=4, predictions=[],
        )
        executor = ThreadPoolExecutor(max_workers=1)
        set_executor(executor)
        try:
            app = self._create_app()
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post(
                    "/v1/ocr_qualitydl",
                    files={"file": ("test.jpg", b"\xff\xd8\xff\xe0", "image/jpeg")},
                    data={"crops": "[]"},
                    headers={"X-API-Key": "test"},
                )
            assert resp.json()["status_code"] == 200
            mock_log.assert_not_awaited()
        finally:
            executor.shutdown(wait=False)
            set_executor(None)  # type: ignore[arg-type]


class TestCheckModelHelper:
    @patch("src.api.routes.get_model", side_effect=RuntimeError("model broken"))
    def test_exception_returns_down(self, _mock):
        from src.api.routes import _check_model
        result = _check_model()
        assert result["status"] == "down"
        assert "model broken" in result["reason"]

    @patch("src.api.routes.get_device", return_value=None)
    @patch("src.api.routes.get_model", return_value=MagicMock())
    def test_model_up_unknown_device(self, _model, _device):
        from src.api.routes import _check_model
        result = _check_model()
        assert result["status"] == "up"
        assert result["device"] == "unknown"
