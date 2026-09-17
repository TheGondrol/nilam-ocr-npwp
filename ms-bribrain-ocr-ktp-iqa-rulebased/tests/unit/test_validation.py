"""Unit tests for src.core.validation module."""

from unittest.mock import MagicMock, patch

import pytest

from src.core.validation import (
    validate_config_values,
    validate_environment_variables,
    validate_startup_configuration,
)


class TestValidateEnvironmentVariables:
    def test_all_set(self):
        env = {
            "MINIO_ENDPOINT": "localhost:9000",
            "MINIO_ACCESS_KEY": "access",
            "MINIO_SECRET_KEY": "secret",
            "DATABASE_URL": "postgresql://localhost/db",
            "API_KEY": "key",
            "LOG_ENCRYPTION_KEY": "key",
        }
        with patch.dict("os.environ", env, clear=True):
            valid, missing = validate_environment_variables()
            assert valid is True
            assert missing == []

    def test_missing_all(self):
        with patch.dict("os.environ", {}, clear=True):
            valid, missing = validate_environment_variables()
            assert valid is False
            # 5 base vars + LOG_ENCRYPTION_KEY (required when log_to_database is on)
            assert len(missing) == 6

    def test_missing_database_url(self):
        env = {
            "MINIO_ENDPOINT": "e",
            "MINIO_ACCESS_KEY": "a",
            "MINIO_SECRET_KEY": "s",
            "API_KEY": "k",
        }
        with patch.dict("os.environ", env, clear=True):
            valid, missing = validate_environment_variables()
            assert valid is False
            assert any("DATABASE_URL" in m for m in missing)

    def test_missing_api_key(self):
        env = {
            "MINIO_ENDPOINT": "e",
            "MINIO_ACCESS_KEY": "a",
            "MINIO_SECRET_KEY": "s",
            "DATABASE_URL": "pg://",
        }
        with patch.dict("os.environ", env, clear=True):
            valid, missing = validate_environment_variables()
            assert valid is False
            assert any("API_KEY" in m for m in missing)


class TestValidateConfigValues:
    def test_valid_config(self):
        valid, missing = validate_config_values()
        assert valid is True
        assert missing == []

    @patch("src.core.validation.get_config")
    def test_empty_server_host(self, mock_get_config):
        cfg = MagicMock()
        cfg.server.host = ""
        cfg.server.port = 8000
        cfg.database.table_name = "t"
        cfg.database.db_schema = "s"
        cfg.quality.blur.threshold = 100.0
        cfg.quality.confidence.threshold_median = 0.5
        mock_get_config.return_value = cfg
        valid, missing = validate_config_values()
        assert valid is False
        assert any("server.host" in m for m in missing)

    @patch("src.core.validation.get_config")
    def test_zero_server_port(self, mock_get_config):
        cfg = MagicMock()
        cfg.server.host = "0.0.0.0"
        cfg.server.port = 0
        cfg.database.table_name = "t"
        cfg.database.db_schema = "s"
        cfg.quality.blur.threshold = 100.0
        cfg.quality.confidence.threshold_median = 0.5
        mock_get_config.return_value = cfg
        valid, missing = validate_config_values()
        assert valid is False
        assert any("server.port" in m for m in missing)

    @patch("src.core.validation.get_config")
    def test_empty_table_name(self, mock_get_config):
        cfg = MagicMock()
        cfg.server.host = "0.0.0.0"
        cfg.server.port = 8000
        cfg.database.table_name = "  "
        cfg.database.db_schema = "s"
        cfg.quality.blur.threshold = 100.0
        cfg.quality.confidence.threshold_median = 0.5
        mock_get_config.return_value = cfg
        valid, missing = validate_config_values()
        assert valid is False
        assert any("table_name" in m for m in missing)

    @patch("src.core.validation.get_config")
    def test_empty_db_schema(self, mock_get_config):
        cfg = MagicMock()
        cfg.server.host = "0.0.0.0"
        cfg.server.port = 8000
        cfg.database.table_name = "t"
        cfg.database.db_schema = ""
        cfg.quality.blur.threshold = 100.0
        cfg.quality.confidence.threshold_median = 0.5
        mock_get_config.return_value = cfg
        valid, missing = validate_config_values()
        assert valid is False
        assert any("schema" in m for m in missing)

    @patch("src.core.validation.get_config")
    def test_invalid_blur_threshold(self, mock_get_config):
        cfg = MagicMock()
        cfg.server.host = "0.0.0.0"
        cfg.server.port = 8000
        cfg.database.table_name = "t"
        cfg.database.db_schema = "s"
        cfg.quality.blur.threshold = 0
        cfg.quality.confidence.threshold_median = 0.5
        mock_get_config.return_value = cfg
        valid, missing = validate_config_values()
        assert valid is False
        assert any("blur.threshold" in m for m in missing)

    @patch("src.core.validation.get_config")
    def test_confidence_threshold_out_of_range(self, mock_get_config):
        cfg = MagicMock()
        cfg.server.host = "0.0.0.0"
        cfg.server.port = 8000
        cfg.database.table_name = "t"
        cfg.database.db_schema = "s"
        cfg.quality.blur.threshold = 100.0
        cfg.quality.confidence.threshold_median = 1.5
        mock_get_config.return_value = cfg
        valid, missing = validate_config_values()
        assert valid is False
        assert any("threshold_median" in m for m in missing)


class TestValidateStartupConfiguration:
    @patch("src.core.validation.validate_config_values", return_value=(True, []))
    @patch("src.core.validation.validate_environment_variables", return_value=(True, []))
    def test_passes_when_valid(self, _env, _cfg):
        validate_startup_configuration()  # should not raise

    @patch("src.core.validation.validate_config_values", return_value=(True, []))
    @patch("src.core.validation.validate_environment_variables", return_value=(False, ["  - DB_URL"]))
    def test_exits_when_env_invalid(self, _env, _cfg):
        with pytest.raises(SystemExit):
            validate_startup_configuration()

    @patch("src.core.validation.validate_config_values", return_value=(False, ["  - blur.threshold"]))
    @patch("src.core.validation.validate_environment_variables", return_value=(True, []))
    def test_exits_when_config_invalid(self, _env, _cfg):
        with pytest.raises(SystemExit):
            validate_startup_configuration()

    @patch("src.core.validation.validate_config_values", return_value=(False, ["  - x"]))
    @patch("src.core.validation.validate_environment_variables", return_value=(False, ["  - y"]))
    def test_exits_when_both_invalid(self, _env, _cfg):
        with pytest.raises(SystemExit):
            validate_startup_configuration()
