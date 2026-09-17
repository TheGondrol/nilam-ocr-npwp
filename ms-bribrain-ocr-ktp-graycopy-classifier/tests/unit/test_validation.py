"""Unit tests for src.core.validation module"""

import os
from unittest.mock import patch, MagicMock

import pytest


class TestValidateEnvironmentVariables:
    def test_all_onprem_vars_set(self):
        env = {
            "DATABASE_URL": "postgresql://...",
            "API_KEY": "key",
            "MINIO_ENDPOINT": "minio:9000",
            "MINIO_ACCESS_KEY": "access",
            "MINIO_SECRET_KEY": "secret",
            "APP_ENVIRO": "onprem",
        }
        with patch.dict(os.environ, env, clear=True):
            from src.core.validation import validate_environment_variables
            valid, missing = validate_environment_variables()
            assert valid is True
            assert missing == []

    def test_missing_onprem_vars(self):
        env = {"APP_ENVIRO": "onprem"}
        with patch.dict(os.environ, env, clear=True):
            from src.core.validation import validate_environment_variables
            valid, missing = validate_environment_variables()
            assert valid is False
            assert len(missing) > 0

    def test_cloud_vars_required(self):
        env = {
            "DATABASE_URL": "postgresql://...",
            "API_KEY": "key",
            "APP_ENVIRO": "cloud",
        }
        with patch.dict(os.environ, env, clear=True):
            from src.core.validation import validate_environment_variables
            valid, missing = validate_environment_variables()
            assert valid is False
            # Should require SA_* vars
            assert any("SA_TYPE" in m for m in missing)

    def test_all_cloud_vars_set(self):
        env = {
            "DATABASE_URL": "postgresql://...",
            "API_KEY": "key",
            "APP_ENVIRO": "cloud",
            "SA_TYPE": "service_account",
            "SA_PROJECT_ID": "proj",
            "SA_PRIVATE_KEY": "key",
            "SA_CLIENT_MAIL": "mail",
            "SA_CLIENT_ID": "id",
            "SA_AUTH_URI": "uri",
            "SA_TOKEN_URI": "token",
            "SA_AUTH_PROVIDER": "prov",
            "SA_CERT_URL": "cert",
        }
        with patch.dict(os.environ, env, clear=True):
            from src.core.validation import validate_environment_variables
            valid, missing = validate_environment_variables()
            assert valid is True


class TestValidateConfigValues:
    def test_valid_config(self):
        mock_config = MagicMock()
        mock_config.model.path = "/path/to/model.pth"
        mock_config.server.host = "0.0.0.0"
        mock_config.server.port = 8020
        mock_config.minio.bucket = "models"
        mock_config.minio.object = "model.pth"
        mock_config.device.compile_model = False

        with patch("src.core.validation.get_config", return_value=mock_config):
            with patch.dict(os.environ, {"APP_ENVIRO": "onprem"}):
                from src.core.validation import validate_config_values
                valid, missing = validate_config_values()
                assert valid is True

    def test_missing_model_path(self):
        mock_config = MagicMock()
        mock_config.model.path = ""
        mock_config.server.host = "0.0.0.0"
        mock_config.server.port = 8020
        mock_config.minio.bucket = "models"
        mock_config.minio.object = "model.pth"
        mock_config.device.compile_model = False

        with patch("src.core.validation.get_config", return_value=mock_config):
            with patch.dict(os.environ, {"APP_ENVIRO": "onprem"}):
                from src.core.validation import validate_config_values
                valid, missing = validate_config_values()
                assert valid is False
                assert any("model.path" in m for m in missing)

    def test_missing_server_host(self):
        mock_config = MagicMock()
        mock_config.model.path = "/path/model.pth"
        mock_config.server.host = ""
        mock_config.server.port = 8020
        mock_config.minio.bucket = "models"
        mock_config.minio.object = "model.pth"
        mock_config.device.compile_model = False

        with patch("src.core.validation.get_config", return_value=mock_config):
            with patch.dict(os.environ, {"APP_ENVIRO": "onprem"}):
                from src.core.validation import validate_config_values
                valid, missing = validate_config_values()
                assert valid is False
                assert any("server.host" in m for m in missing)

    def test_missing_server_port(self):
        mock_config = MagicMock()
        mock_config.model.path = "/path/model.pth"
        mock_config.server.host = "0.0.0.0"
        mock_config.server.port = 0
        mock_config.minio.bucket = "models"
        mock_config.minio.object = "model.pth"
        mock_config.device.compile_model = False

        with patch("src.core.validation.get_config", return_value=mock_config):
            with patch.dict(os.environ, {"APP_ENVIRO": "onprem"}):
                from src.core.validation import validate_config_values
                valid, missing = validate_config_values()
                assert valid is False
                assert any("server.port" in m for m in missing)

    def test_missing_minio_bucket_onprem(self):
        mock_config = MagicMock()
        mock_config.model.path = "/path/model.pth"
        mock_config.server.host = "0.0.0.0"
        mock_config.server.port = 8020
        mock_config.minio.bucket = ""
        mock_config.minio.object = "model.pth"
        mock_config.device.compile_model = False

        with patch("src.core.validation.get_config", return_value=mock_config):
            with patch.dict(os.environ, {"APP_ENVIRO": "onprem"}):
                from src.core.validation import validate_config_values
                valid, missing = validate_config_values()
                assert valid is False
                assert any("minio.bucket" in m for m in missing)

    def test_missing_minio_object_onprem(self):
        mock_config = MagicMock()
        mock_config.model.path = "/path/model.pth"
        mock_config.server.host = "0.0.0.0"
        mock_config.server.port = 8020
        mock_config.minio.bucket = "models"
        mock_config.minio.object = ""
        mock_config.device.compile_model = False

        with patch("src.core.validation.get_config", return_value=mock_config):
            with patch.dict(os.environ, {"APP_ENVIRO": "onprem"}):
                from src.core.validation import validate_config_values
                valid, missing = validate_config_values()
                assert valid is False
                assert any("minio.object" in m for m in missing)

    def test_cloud_env_skips_minio_checks(self):
        mock_config = MagicMock()
        mock_config.model.path = "/path/model.pth"
        mock_config.server.host = "0.0.0.0"
        mock_config.server.port = 8020
        mock_config.minio.bucket = ""
        mock_config.minio.object = ""
        mock_config.device.compile_model = False

        with patch("src.core.validation.get_config", return_value=mock_config):
            with patch.dict(os.environ, {"APP_ENVIRO": "cloud"}):
                from src.core.validation import validate_config_values
                valid, missing = validate_config_values()
                # MinIO values missing but APP_ENVIRO=cloud skips those checks
                assert valid is True
                assert not any("minio" in m for m in missing)

    def test_invalid_compile_mode(self):
        mock_config = MagicMock()
        mock_config.model.path = "/path/model.pth"
        mock_config.server.host = "0.0.0.0"
        mock_config.server.port = 8020
        mock_config.minio.bucket = "models"
        mock_config.minio.object = "model.pth"
        mock_config.device.compile_model = True
        mock_config.device.is_valid_compile_mode = False
        mock_config.device.compile_mode = "invalid_mode"

        with patch("src.core.validation.get_config", return_value=mock_config):
            with patch.dict(os.environ, {"APP_ENVIRO": "onprem"}):
                from src.core.validation import validate_config_values
                valid, missing = validate_config_values()
                assert valid is False
                assert any("compile_mode" in m for m in missing)


class TestValidateStartupConfiguration:
    def test_passes_when_valid(self):
        with patch("src.core.validation.validate_environment_variables", return_value=(True, [])):
            with patch("src.core.validation.validate_config_values", return_value=(True, [])):
                from src.core.validation import validate_startup_configuration
                validate_startup_configuration()  # should not exit

    def test_exits_when_env_invalid(self):
        with patch("src.core.validation.validate_environment_variables", return_value=(False, ["  - API_KEY"])):
            with patch("src.core.validation.validate_config_values", return_value=(True, [])):
                from src.core.validation import validate_startup_configuration
                with pytest.raises(SystemExit):
                    validate_startup_configuration()

    def test_exits_when_config_invalid(self):
        with patch("src.core.validation.validate_environment_variables", return_value=(True, [])):
            with patch("src.core.validation.validate_config_values", return_value=(False, ["  - model.path"])):
                from src.core.validation import validate_startup_configuration
                with pytest.raises(SystemExit):
                    validate_startup_configuration()
