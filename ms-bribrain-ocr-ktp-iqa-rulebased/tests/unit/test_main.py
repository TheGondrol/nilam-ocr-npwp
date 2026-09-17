"""Unit tests for src.main module."""

import json
from unittest.mock import patch, AsyncMock, MagicMock

import pytest

from src.main import app, root, health_check, liveness, readiness_check, _check_database


class TestAppCreation:
    def test_app_exists(self):
        assert app is not None
        assert app.title is not None

    def test_app_has_routes(self):
        routes = [getattr(r, "path", None) for r in app.routes]
        assert "/" in routes
        assert "/health" in routes


class TestRootEndpoint:
    async def test_root_returns_info(self):
        result = await root()
        assert "service" in result
        assert "version" in result
        assert result["status"] == "running"


class TestHealthEndpoint:
    async def test_health_returns_healthy(self):
        with patch("src.main._check_database", new_callable=AsyncMock, return_value={"status": "up"}):
            result = await health_check()
        assert result.status == "healthy"

    async def test_health_returns_unhealthy_when_db_down(self):
        with patch("src.main._check_database", new_callable=AsyncMock, return_value={"status": "down", "reason": "x"}):
            result = await health_check()
        assert result.status == "unhealthy"


class TestLivenessEndpoint:
    async def test_liveness_returns_alive(self):
        result = await liveness()
        assert result.status == "alive"
        assert result.version is not None


class TestReadinessEndpoint:
    async def test_readiness_ready_when_db_up(self):
        with patch("src.main._check_database", new_callable=AsyncMock, return_value={"status": "up"}):
            resp = await readiness_check()
        assert resp.status_code == 200
        body = json.loads(bytes(resp.body))
        assert body["status"] == "ready"

    async def test_readiness_not_ready_when_db_down(self):
        with patch("src.main._check_database", new_callable=AsyncMock, return_value={"status": "down", "reason": "oops"}):
            resp = await readiness_check()
        assert resp.status_code == 503
        body = json.loads(bytes(resp.body))
        assert body["status"] == "not_ready"


class TestCheckDatabase:
    async def test_returns_down_when_factory_missing(self):
        with patch("src.services.database_service._async_session_factory", None):
            result = await _check_database()
        assert result["status"] == "down"
        assert "not initialised" in result["reason"]

    async def test_returns_up_when_query_succeeds(self):
        session = AsyncMock()
        session.execute = AsyncMock()
        factory = MagicMock()
        factory.return_value.__aenter__.return_value = session
        factory.return_value.__aexit__.return_value = None
        with patch("src.services.database_service._async_session_factory", factory):
            result = await _check_database()
        assert result["status"] == "up"

    async def test_returns_down_on_exception(self):
        factory = MagicMock(side_effect=RuntimeError("boom"))
        with patch("src.services.database_service._async_session_factory", factory):
            result = await _check_database()
        assert result["status"] == "down"
        assert "boom" in result["reason"]


class TestLifespan:
    @patch("src.main.dispose_engine", new_callable=AsyncMock)
    @patch("src.main.init_engine", new_callable=AsyncMock)
    @patch("src.core.validation.validate_startup_configuration")
    @patch("src.main.get_device_info", return_value={"selected_device": "cpu"})
    async def test_lifespan_startup_shutdown(self, _device, _validate, mock_init, mock_dispose):
        from src.main import lifespan
        from fastapi import FastAPI

        test_app = FastAPI()
        async with lifespan(test_app):
            mock_init.assert_awaited_once()

        mock_dispose.assert_awaited_once()

    @patch("src.main.dispose_engine", new_callable=AsyncMock)
    @patch("src.main.init_engine", new_callable=AsyncMock)
    @patch("src.core.validation.validate_startup_configuration", side_effect=SystemExit(1))
    async def test_lifespan_validation_failure(self, _validate, _init, _dispose):
        from src.main import lifespan
        from fastapi import FastAPI

        test_app = FastAPI()
        with pytest.raises(SystemExit):
            async with lifespan(test_app):
                pass
