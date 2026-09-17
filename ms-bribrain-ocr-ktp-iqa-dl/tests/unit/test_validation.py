"""Unit tests for src.core.validation module."""

from unittest.mock import patch

import pytest

from src.core.validation import (
    validate_config_values,
    validate_environment_variables,
    validate_startup_configuration,
)


class TestValidateEnvironmentVariables:
    def test_onprem_all_set(self):
        env = {
            "APP_ENVIRO": "onprem",
            "DATABASE_URL": "postgresql://localhost/db",
            "API_KEY": "secret",
            "MINIO_ENDPOINT": "localhost:9000",
            "MINIO_ACCESS_KEY": "access",
            "MINIO_SECRET_KEY": "secret",
        }
        with patch.dict("os.environ", env, clear=True):
            valid, missing = validate_environment_variables()
            assert valid is True
            assert missing == []

    def test_onprem_missing_minio(self):
        env = {
            "APP_ENVIRO": "onprem",
            "DATABASE_URL": "postgresql://localhost/db",
            "API_KEY": "secret",
        }
        with patch.dict("os.environ", env, clear=True):
            valid, missing = validate_environment_variables()
            assert valid is False
            assert len(missing) == 3  # MINIO_ENDPOINT, ACCESS_KEY, SECRET_KEY

    def test_cloud_all_set(self):
        env = {
            "APP_ENVIRO": "cloud",
            "DATABASE_URL": "postgresql://localhost/db",
            "API_KEY": "secret",
            "SA_TYPE": "service_account",
            "SA_PROJECT_ID": "proj",
            "SA_PRIVATE_KEY": "key",
            "SA_CLIENT_MAIL": "mail",
            "SA_CLIENT_ID": "id",
            "SA_AUTH_URI": "uri",
            "SA_TOKEN_URI": "uri",
            "SA_AUTH_PROVIDER": "prov",
            "SA_CERT_URL": "url",
        }
        with patch.dict("os.environ", env, clear=True):
            valid, missing = validate_environment_variables()
            assert valid is True

    def test_cloud_missing_sa_vars(self):
        env = {
            "APP_ENVIRO": "cloud",
            "DATABASE_URL": "postgresql://localhost/db",
            "API_KEY": "secret",
        }
        with patch.dict("os.environ", env, clear=True):
            valid, missing = validate_environment_variables()
            assert valid is False
            assert len(missing) == 9

    def test_missing_database_url(self):
        env = {
            "APP_ENVIRO": "onprem",
            "API_KEY": "secret",
            "MINIO_ENDPOINT": "localhost",
            "MINIO_ACCESS_KEY": "a",
            "MINIO_SECRET_KEY": "s",
        }
        with patch.dict("os.environ", env, clear=True):
            valid, missing = validate_environment_variables()
            assert valid is False
            assert any("DATABASE_URL" in m for m in missing)

    def test_defaults_to_onprem(self):
        env = {
            "DATABASE_URL": "pg://",
            "API_KEY": "k",
            "MINIO_ENDPOINT": "e",
            "MINIO_ACCESS_KEY": "a",
            "MINIO_SECRET_KEY": "s",
        }
        with patch.dict("os.environ", env, clear=True):
            valid, _ = validate_environment_variables()
            assert valid is True

    def test_empty_string_env_var_treated_as_missing(self):
        """Empty string values fail `if not value` check, so they count as missing."""
        env = {
            "APP_ENVIRO": "onprem",
            "DATABASE_URL": "",  # empty → missing
            "API_KEY": "secret",
            "MINIO_ENDPOINT": "localhost:9000",
            "MINIO_ACCESS_KEY": "access",
            "MINIO_SECRET_KEY": "secret",
        }
        with patch.dict("os.environ", env, clear=True):
            valid, missing = validate_environment_variables()
            assert valid is False
            assert any("DATABASE_URL" in m for m in missing)

    def test_non_onprem_app_enviro_requires_cloud_vars(self):
        """Any APP_ENVIRO value other than 'onprem' falls into the else (cloud) branch."""
        env = {
            "APP_ENVIRO": "staging",
            "DATABASE_URL": "pg://localhost/db",
            "API_KEY": "secret",
            # SA vars missing
        }
        with patch.dict("os.environ", env, clear=True):
            valid, missing = validate_environment_variables()
            assert valid is False
            assert len(missing) == 9  # all SA vars required


class TestValidateConfigValues:
    def test_valid_config(self):
        valid, missing = validate_config_values()
        assert valid is True
        assert missing == []

    @patch("src.core.validation.config")
    def test_img_height_zero_is_invalid(self, mock_cfg):
        mock_cfg.model_path = "/some/model.pth"
        mock_cfg.server_host = "0.0.0.0"
        mock_cfg.server_port = 8080
        mock_cfg.img_height = 0
        mock_cfg.img_width = 64
        mock_cfg.use_compile = False
        mock_cfg.get.side_effect = lambda k, d=None: {"minio.bucket": "b", "minio.object": "o"}.get(k, d)
        with patch.dict("os.environ", {"APP_ENVIRO": "onprem"}):
            valid, missing = validate_config_values()
        assert valid is False
        assert any("img_height" in m for m in missing)

    @patch("src.core.validation.config")
    def test_img_width_negative_is_invalid(self, mock_cfg):
        mock_cfg.model_path = "/some/model.pth"
        mock_cfg.server_host = "0.0.0.0"
        mock_cfg.server_port = 8080
        mock_cfg.img_height = 64
        mock_cfg.img_width = -1
        mock_cfg.use_compile = False
        mock_cfg.get.side_effect = lambda k, d=None: {"minio.bucket": "b", "minio.object": "o"}.get(k, d)
        with patch.dict("os.environ", {"APP_ENVIRO": "onprem"}):
            valid, missing = validate_config_values()
        assert valid is False
        assert any("img_width" in m for m in missing)

    @patch("src.core.validation.config")
    def test_invalid_compile_mode_when_compile_enabled(self, mock_cfg):
        mock_cfg.model_path = "/some/model.pth"
        mock_cfg.server_host = "0.0.0.0"
        mock_cfg.server_port = 8080
        mock_cfg.img_height = 64
        mock_cfg.img_width = 320
        mock_cfg.use_compile = True
        mock_cfg.compile_mode = "turbo"  # not in valid modes
        mock_cfg.get.side_effect = lambda k, d=None: {"minio.bucket": "b", "minio.object": "o"}.get(k, d)
        with patch.dict("os.environ", {"APP_ENVIRO": "onprem"}):
            valid, missing = validate_config_values()
        assert valid is False
        assert any("compile_mode" in m for m in missing)

    @patch("src.core.validation.config")
    def test_empty_model_path_is_invalid(self, mock_cfg):
        mock_cfg.model_path = "   "  # whitespace-only
        mock_cfg.server_host = "0.0.0.0"
        mock_cfg.server_port = 8080
        mock_cfg.img_height = 64
        mock_cfg.img_width = 320
        mock_cfg.use_compile = False
        mock_cfg.get.side_effect = lambda k, d=None: {"minio.bucket": "b", "minio.object": "o"}.get(k, d)
        with patch.dict("os.environ", {"APP_ENVIRO": "onprem"}):
            valid, missing = validate_config_values()
        assert valid is False
        assert any("model.path" in m for m in missing)


class TestValidateStartupConfiguration:
    @patch("src.core.validation.validate_config_values", return_value=(True, []))
    @patch("src.core.validation.validate_environment_variables", return_value=(True, []))
    def test_passes_when_all_valid(self, _env, _cfg):
        validate_startup_configuration()  # should not raise or exit

    @patch("src.core.validation.validate_config_values", return_value=(True, []))
    @patch("src.core.validation.validate_environment_variables", return_value=(False, ["  - DB_URL"]))
    def test_exits_when_env_invalid(self, _env, _cfg):
        with pytest.raises(SystemExit):
            validate_startup_configuration()

    @patch("src.core.validation.validate_config_values", return_value=(False, ["  - model.path"]))
    @patch("src.core.validation.validate_environment_variables", return_value=(True, []))
    def test_exits_when_config_invalid(self, _env, _cfg):
        with pytest.raises(SystemExit):
            validate_startup_configuration()

    @patch("src.core.validation.validate_config_values", return_value=(False, ["  - model.path"]))
    @patch("src.core.validation.validate_environment_variables", return_value=(False, ["  - API_KEY"]))
    def test_exits_when_both_invalid(self, _env, _cfg):
        """Both env and config failures still cause sys.exit(1)."""
        with pytest.raises(SystemExit):
            validate_startup_configuration()
