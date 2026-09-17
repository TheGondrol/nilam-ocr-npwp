"""Tests for src/core/validation.py."""

from unittest.mock import MagicMock, patch

import pytest

from src.core import validation


@pytest.fixture
def _clear_env(monkeypatch):
    for var in ("MINIO_ENDPOINT", "MINIO_ACCESS_KEY", "MINIO_SECRET_KEY", "DATABASE_URL", "API_KEY"):
        monkeypatch.delenv(var, raising=False)


class TestValidateEnvironmentVariables:
    def test_all_vars_set(self, monkeypatch, _clear_env):
        for var in ("MINIO_ENDPOINT", "MINIO_ACCESS_KEY", "MINIO_SECRET_KEY", "DATABASE_URL", "API_KEY"):
            monkeypatch.setenv(var, "x")
        is_valid, missing = validation.validate_environment_variables()
        assert is_valid is True
        assert missing == []

    def test_all_missing(self, _clear_env):
        is_valid, missing = validation.validate_environment_variables()
        assert is_valid is False
        assert len(missing) == 5

    def test_partial_missing(self, monkeypatch, _clear_env):
        monkeypatch.setenv("API_KEY", "x")
        monkeypatch.setenv("DATABASE_URL", "x")
        is_valid, missing = validation.validate_environment_variables()
        assert is_valid is False
        assert len(missing) == 3


def _make_config(api=None, thresholds=None, database=None):
    cfg = MagicMock()
    cfg.get_api_config.return_value = api if api is not None else {"host": "0.0.0.0", "port": 8000}
    cfg.get_thresholds.return_value = thresholds if thresholds is not None else {
        "partial": 75, "ratio": 80, "confidence": 0.6,
    }
    cfg.get_database_config.return_value = database if database is not None else {"table_name": "logs"}
    return cfg


class TestValidateConfigValues:
    def test_valid_config(self, monkeypatch):
        monkeypatch.setattr(validation, "get_config", lambda: _make_config())
        is_valid, missing = validation.validate_config_values()
        assert is_valid is True
        assert missing == []

    def test_missing_host(self, monkeypatch):
        cfg = _make_config(api={"port": 8000})
        monkeypatch.setattr(validation, "get_config", lambda: cfg)
        is_valid, missing = validation.validate_config_values()
        assert is_valid is False
        assert any("api.host" in m for m in missing)

    def test_whitespace_only_host(self, monkeypatch):
        cfg = _make_config(api={"host": "   ", "port": 8000})
        monkeypatch.setattr(validation, "get_config", lambda: cfg)
        is_valid, missing = validation.validate_config_values()
        assert is_valid is False
        assert any("api.host" in m for m in missing)

    def test_missing_port(self, monkeypatch):
        cfg = _make_config(api={"host": "0.0.0.0"})
        monkeypatch.setattr(validation, "get_config", lambda: cfg)
        is_valid, missing = validation.validate_config_values()
        assert is_valid is False
        assert any("api.port" in m for m in missing)

    def test_empty_thresholds(self, monkeypatch):
        cfg = _make_config(thresholds={})
        monkeypatch.setattr(validation, "get_config", lambda: cfg)
        is_valid, missing = validation.validate_config_values()
        assert is_valid is False
        assert any("thresholds" in m for m in missing)

    def test_partial_thresholds_missing_keys(self, monkeypatch):
        cfg = _make_config(thresholds={"partial": 75})  # ratio, confidence missing
        monkeypatch.setattr(validation, "get_config", lambda: cfg)
        is_valid, missing = validation.validate_config_values()
        assert is_valid is False
        keys = " ".join(missing)
        assert "thresholds.ratio" in keys
        assert "thresholds.confidence" in keys
        assert "thresholds.partial" not in keys

    def test_missing_table_name(self, monkeypatch):
        cfg = _make_config(database={})
        monkeypatch.setattr(validation, "get_config", lambda: cfg)
        is_valid, missing = validation.validate_config_values()
        assert is_valid is False
        assert any("database.table_name" in m for m in missing)

    def test_whitespace_only_table_name(self, monkeypatch):
        cfg = _make_config(database={"table_name": "   "})
        monkeypatch.setattr(validation, "get_config", lambda: cfg)
        is_valid, missing = validation.validate_config_values()
        assert is_valid is False
        assert any("database.table_name" in m for m in missing)


class TestValidateStartupConfiguration:
    def test_success_does_not_exit(self, monkeypatch, _clear_env):
        # LOG_ENCRYPTION_KEY is required when log_to_database is on (the mock config).
        for var in ("MINIO_ENDPOINT", "MINIO_ACCESS_KEY", "MINIO_SECRET_KEY", "DATABASE_URL", "API_KEY", "LOG_ENCRYPTION_KEY"):
            monkeypatch.setenv(var, "x")
        monkeypatch.setattr(validation, "get_config", lambda: _make_config())
        # should not raise SystemExit
        validation.validate_startup_configuration()

    def test_failure_calls_sys_exit(self, monkeypatch, _clear_env):
        # env missing + config missing
        cfg = _make_config(api={}, thresholds={}, database={})
        monkeypatch.setattr(validation, "get_config", lambda: cfg)
        with pytest.raises(SystemExit) as exc_info:
            validation.validate_startup_configuration()
        assert exc_info.value.code == 1

    def test_failure_only_env_missing(self, monkeypatch, _clear_env):
        monkeypatch.setattr(validation, "get_config", lambda: _make_config())
        with pytest.raises(SystemExit):
            validation.validate_startup_configuration()

    def test_failure_only_config_missing(self, monkeypatch, _clear_env):
        for var in ("MINIO_ENDPOINT", "MINIO_ACCESS_KEY", "MINIO_SECRET_KEY", "DATABASE_URL", "API_KEY"):
            monkeypatch.setenv(var, "x")
        cfg = _make_config(api={})
        monkeypatch.setattr(validation, "get_config", lambda: cfg)
        with pytest.raises(SystemExit):
            validation.validate_startup_configuration()
