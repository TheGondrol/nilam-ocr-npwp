"""Unit tests for src.core.validation module."""

from unittest.mock import patch

import pytest

from src.core.validation import (
    validate_environment_variables,
    validate_config_values,
    validate_startup_configuration,
)


class TestValidateEnvironmentVariables:
    def test_all_set_onprem(self):
        env = {
            "APP_ENVIRO": "onprem",
            "DATABASE_URL": "postgresql://x",
            "API_KEY": "key",
            "MINIO_ENDPOINT": "ep",
            "MINIO_ACCESS_KEY": "ak",
            "MINIO_SECRET_KEY": "sk",
        }
        with patch.dict("os.environ", env, clear=True):
            valid, missing = validate_environment_variables()
        assert valid is True
        assert missing == []

    def test_missing_minio_onprem(self):
        env = {
            "APP_ENVIRO": "onprem",
            "DATABASE_URL": "postgresql://x",
            "API_KEY": "key",
        }
        with patch.dict("os.environ", env, clear=True):
            valid, missing = validate_environment_variables()
        assert valid is False
        assert len(missing) == 3

    def test_cloud_env_vars(self):
        env = {
            "APP_ENVIRO": "cloud",
            "DATABASE_URL": "postgresql://x",
            "API_KEY": "key",
            "SA_TYPE": "t",
            "SA_PROJECT_ID": "p",
            "SA_PRIVATE_KEY": "pk",
            "SA_CLIENT_MAIL": "m",
            "SA_CLIENT_ID": "id",
            "SA_AUTH_URI": "au",
            "SA_TOKEN_URI": "tu",
            "SA_AUTH_PROVIDER": "ap",
            "SA_CERT_URL": "cu",
        }
        with patch.dict("os.environ", env, clear=True):
            valid, missing = validate_environment_variables()
        assert valid is True

    def test_missing_database_url(self):
        env = {"APP_ENVIRO": "onprem", "API_KEY": "key"}
        with patch.dict("os.environ", env, clear=True):
            valid, missing = validate_environment_variables()
        assert valid is False


class TestValidateConfigValues:
    def test_valid_config(self):
        valid, missing = validate_config_values()
        assert valid is True
        assert missing == []

    def test_threshold_above_one_rejected(self):
        # Guard against silent misconfiguration: threshold outside [0, 1]
        # would classify every input as one class regardless of model output.
        with patch("src.core.validation.config") as mock_cfg:
            mock_cfg.model_path = "src/models/best_cnn_detector_correctedv2.pth"
            mock_cfg.server_host = "0.0.0.0"
            mock_cfg.server_port = 8000
            mock_cfg.image_size = 76
            mock_cfg.threshold = 1.5
            mock_cfg.get.return_value = "some-value"
            with patch.dict("os.environ", {"APP_ENVIRO": "cloud"}, clear=True):
                valid, missing = validate_config_values()
        assert valid is False
        assert any("threshold" in m for m in missing)

    def test_threshold_negative_rejected(self):
        with patch("src.core.validation.config") as mock_cfg:
            mock_cfg.model_path = "src/models/best_cnn_detector_correctedv2.pth"
            mock_cfg.server_host = "0.0.0.0"
            mock_cfg.server_port = 8000
            mock_cfg.image_size = 76
            mock_cfg.threshold = -0.1
            mock_cfg.get.return_value = "some-value"
            with patch.dict("os.environ", {"APP_ENVIRO": "cloud"}, clear=True):
                valid, missing = validate_config_values()
        assert valid is False
        assert any("threshold" in m for m in missing)

    def test_image_size_zero_rejected(self):
        with patch("src.core.validation.config") as mock_cfg:
            mock_cfg.model_path = "src/models/best_cnn_detector_correctedv2.pth"
            mock_cfg.server_host = "0.0.0.0"
            mock_cfg.server_port = 8000
            mock_cfg.image_size = 0
            mock_cfg.threshold = 0.5
            mock_cfg.get.return_value = "some-value"
            with patch.dict("os.environ", {"APP_ENVIRO": "cloud"}, clear=True):
                valid, missing = validate_config_values()
        assert valid is False
        assert any("image_size" in m for m in missing)


class TestValidateStartupConfiguration:
    def test_passes_when_valid(self):
        env = {
            "APP_ENVIRO": "onprem",
            "DATABASE_URL": "postgresql://x",
            "API_KEY": "key",
            "MINIO_ENDPOINT": "ep",
            "MINIO_ACCESS_KEY": "ak",
            "MINIO_SECRET_KEY": "sk",
        }
        with patch.dict("os.environ", env, clear=True):
            validate_startup_configuration()  # Should not raise

    def test_exits_on_failure(self):
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(SystemExit):
                validate_startup_configuration()
