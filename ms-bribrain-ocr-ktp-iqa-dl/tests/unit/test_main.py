"""Unit tests for src.main module."""

from unittest.mock import patch, MagicMock, AsyncMock

import pytest
import torch


class TestConfigureCpuThreading:
    @patch("src.main.config")
    def test_sets_env_vars(self, mock_config):
        mock_config.thread_pool_workers = 8
        from src.main import _configure_cpu_threading
        import os

        _configure_cpu_threading()
        assert os.environ["OMP_NUM_THREADS"] == "8"
        assert os.environ["MKL_NUM_THREADS"] == "8"


class TestLifespan:
    @pytest.fixture(autouse=True)
    def _mock_threshold_provider(self):
        """Stub the threshold provider so lifespan startup/shutdown does no real DB work."""
        with patch("src.main.init_provider") as mock_init, \
             patch("src.main.get_provider") as mock_get:
            mock_init.return_value.initialize = AsyncMock()
            mock_get.return_value.shutdown = AsyncMock()
            yield

    @patch("src.main.dispose_engine", new_callable=AsyncMock)
    @patch("src.main.init_engine", new_callable=AsyncMock)
    @patch("src.main.set_model_globals")
    @patch("src.main.warmup_model")
    @patch("src.main.get_transform")
    @patch("src.main.load_mobilenet_model")
    @patch("src.main.download_model_minio")
    @patch("src.core.validation.validate_startup_configuration")
    @patch("src.main._configure_cpu_threading")
    @patch("src.main.get_device")
    @patch("src.main.config")
    @patch("src.main.APP_ENVIRO", "onprem")
    async def test_lifespan_onprem(
        self, mock_config, mock_get_device, mock_cpu_threading,
        mock_validate, mock_download_minio, mock_load_model,
        mock_get_transform, mock_warmup, mock_set_globals,
        mock_init_engine, mock_dispose_engine,
    ):
        mock_config.get.return_value = 0.67
        mock_config.model_path = "model.pth"
        mock_config.num_classes = 2
        mock_config.thread_pool_workers = 2
        mock_config.bad_crop_threshold = 4
        mock_get_device.return_value = torch.device("cpu")
        mock_model = MagicMock()
        mock_load_model.return_value = mock_model
        mock_transform = MagicMock()
        mock_get_transform.return_value = mock_transform

        from src.main import lifespan
        from fastapi import FastAPI
        app = FastAPI()

        async with lifespan(app):
            mock_validate.assert_called_once()
            mock_cpu_threading.assert_called_once()
            mock_download_minio.assert_called_once()
            mock_load_model.assert_called_once()
            mock_set_globals.assert_called_once()
            mock_warmup.assert_called_once()
            mock_init_engine.assert_awaited_once()

        mock_dispose_engine.assert_awaited_once()

    @patch("src.main.dispose_engine", new_callable=AsyncMock)
    @patch("src.main.init_engine", new_callable=AsyncMock)
    @patch("src.main.set_model_globals")
    @patch("src.main.warmup_model")
    @patch("src.main.get_transform")
    @patch("src.main.load_mobilenet_model")
    @patch("src.main.download_model_gcs")
    @patch("src.core.validation.validate_startup_configuration")
    @patch("src.main._configure_cpu_threading")
    @patch("src.main.get_device")
    @patch("src.main.config")
    @patch("src.main.APP_ENVIRO", "cloud")
    async def test_lifespan_cloud(
        self, mock_config, mock_get_device, mock_cpu_threading,
        mock_validate, mock_download_gcs, mock_load_model,
        mock_get_transform, mock_warmup, mock_set_globals,
        mock_init_engine, mock_dispose_engine,
    ):
        mock_config.get.return_value = 0.67
        mock_config.model_path = "model.pth"
        mock_config.num_classes = 2
        mock_config.thread_pool_workers = 2
        mock_config.bad_crop_threshold = 4
        mock_get_device.return_value = torch.device("cpu")
        mock_load_model.return_value = MagicMock()
        mock_get_transform.return_value = MagicMock()

        from src.main import lifespan
        from fastapi import FastAPI
        app = FastAPI()

        async with lifespan(app):
            mock_download_gcs.assert_called_once()

        mock_dispose_engine.assert_awaited_once()

    @patch("src.main.dispose_engine", new_callable=AsyncMock)
    @patch("src.main.init_engine", new_callable=AsyncMock)
    @patch("src.main.set_model_globals")
    @patch("src.main.warmup_model")
    @patch("src.main.get_transform")
    @patch("src.main.load_mobilenet_model", side_effect=Exception("model fail"))
    @patch("src.main.download_model_minio")
    @patch("src.core.validation.validate_startup_configuration")
    @patch("src.main._configure_cpu_threading")
    @patch("src.main.get_device")
    @patch("src.main.config")
    @patch("src.main.APP_ENVIRO", "onprem")
    async def test_lifespan_model_load_failure(
        self, mock_config, mock_get_device, mock_cpu_threading,
        mock_validate, mock_download, mock_load_model,
        mock_get_transform, mock_warmup, mock_set_globals,
        mock_init_engine, mock_dispose_engine,
    ):
        mock_config.get.return_value = 0.67
        mock_config.model_path = "model.pth"
        mock_config.num_classes = 2
        mock_config.thread_pool_workers = 2
        mock_config.bad_crop_threshold = 4
        mock_get_device.return_value = torch.device("cpu")

        from src.main import lifespan
        from fastapi import FastAPI
        app = FastAPI()

        with pytest.raises(Exception, match="model fail"):
            async with lifespan(app):
                pass

    @patch("src.main.dispose_engine", new_callable=AsyncMock)
    @patch("src.main.init_engine", new_callable=AsyncMock)
    @patch("src.main.set_model_globals")
    @patch("src.main.warmup_model")
    @patch("src.main.get_transform")
    @patch("src.main.load_mobilenet_model")
    @patch("src.main.download_model_minio", side_effect=Exception("download fail"))
    @patch("src.core.validation.validate_startup_configuration")
    @patch("src.main._configure_cpu_threading")
    @patch("src.main.get_device")
    @patch("src.main.config")
    @patch("src.main.APP_ENVIRO", "onprem")
    async def test_lifespan_download_failure_continues(
        self, mock_config, mock_get_device, mock_cpu_threading,
        mock_validate, mock_download, mock_load_model,
        mock_get_transform, mock_warmup, mock_set_globals,
        mock_init_engine, mock_dispose_engine,
    ):
        mock_config.get.return_value = 0.67
        mock_config.model_path = "model.pth"
        mock_config.num_classes = 2
        mock_config.thread_pool_workers = 2
        mock_config.bad_crop_threshold = 4
        mock_get_device.return_value = torch.device("cpu")
        mock_load_model.return_value = MagicMock()
        mock_get_transform.return_value = MagicMock()

        from src.main import lifespan
        from fastapi import FastAPI
        app = FastAPI()

        # Download failure should be caught, app should still start
        async with lifespan(app):
            mock_load_model.assert_called_once()

    @patch("src.main.dispose_engine", new_callable=AsyncMock)
    @patch("src.main.init_engine", new_callable=AsyncMock, side_effect=Exception("db fail"))
    @patch("src.main.set_model_globals")
    @patch("src.main.warmup_model")
    @patch("src.main.get_transform")
    @patch("src.main.load_mobilenet_model")
    @patch("src.main.download_model_minio")
    @patch("src.core.validation.validate_startup_configuration")
    @patch("src.main._configure_cpu_threading")
    @patch("src.main.get_device")
    @patch("src.main.config")
    @patch("src.main.APP_ENVIRO", "onprem")
    async def test_lifespan_db_failure_continues(
        self, mock_config, mock_get_device, mock_cpu_threading,
        mock_validate, mock_download, mock_load_model,
        mock_get_transform, mock_warmup, mock_set_globals,
        mock_init_engine, mock_dispose_engine,
    ):
        mock_config.get.return_value = 0.67
        mock_config.model_path = "model.pth"
        mock_config.num_classes = 2
        mock_config.thread_pool_workers = 2
        mock_config.bad_crop_threshold = 4
        mock_get_device.return_value = torch.device("cpu")
        mock_load_model.return_value = MagicMock()
        mock_get_transform.return_value = MagicMock()

        from src.main import lifespan
        from fastapi import FastAPI
        app = FastAPI()

        # DB init failure should be caught, app should still start
        async with lifespan(app):
            pass


class TestAppCreation:
    def test_app_exists(self):
        from src.main import app
        assert app is not None
        assert app.title is not None
