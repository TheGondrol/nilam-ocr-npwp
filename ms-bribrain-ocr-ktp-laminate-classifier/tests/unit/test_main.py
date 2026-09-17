"""Unit tests for src.main module."""

from unittest.mock import patch, AsyncMock, MagicMock

import pytest

from src.main import app, _configure_cpu_threading


class TestAppCreation:
    def test_app_exists(self):
        assert app is not None
        assert app.title is not None

    def test_app_has_routes(self):
        routes = [getattr(r, "path", None) for r in app.routes]
        assert "/health" in routes
        assert "/" in routes


class TestConfigureCpuThreading:
    def test_sets_env_vars(self):
        import os
        # Clear any existing values
        for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ.pop(key, None)
        _configure_cpu_threading()
        assert "OMP_NUM_THREADS" in os.environ

    def test_uses_config_value(self):
        import os
        for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ.pop(key, None)
        with patch("src.main.config") as mock_cfg:
            mock_cfg.get.return_value = 2
            _configure_cpu_threading()
        assert os.environ.get("OMP_NUM_THREADS") == "2"


class TestLifespan:
    @patch("src.main.dispose_engine", new_callable=AsyncMock)
    @patch("src.main.init_engine", new_callable=AsyncMock)
    @patch("src.main.predictor")
    @patch("src.main.download_model_minio")
    @patch("src.core.validation.validate_startup_configuration")
    async def test_lifespan_startup_shutdown(self, _validate, _download, mock_pred, mock_init, mock_dispose):
        from src.main import lifespan
        from fastapi import FastAPI

        mock_pred.load_model = MagicMock()
        mock_pred.cleanup = MagicMock()

        test_app = FastAPI()
        async with lifespan(test_app):
            mock_init.assert_awaited_once()
            mock_pred.load_model.assert_called_once()

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
