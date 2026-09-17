"""
Unit tests for core.validation module
"""
from unittest.mock import patch, MagicMock
import pytest

from src.core.validation import (
    validate_environment_variables,
    validate_config_values,
    validate_startup_configuration,
)


@pytest.mark.unit
class TestValidateEnvironmentVariables:
    """Test validate_environment_variables function"""

    def test_all_vars_set(self):
        """Test validation passes when all required vars are set"""
        env = {
            "MINIO_ENDPOINT": "localhost:9000",
            "MINIO_ACCESS_KEY": "access",
            "MINIO_SECRET_KEY": "secret",
            "DATABASE_URL": "postgresql://localhost/db",
            "API_KEY": "some-key",
        }
        with patch.dict("os.environ", env, clear=True):
            is_valid, missing = validate_environment_variables()

            assert is_valid
            assert missing == []

    def test_missing_api_key(self):
        """Test validation fails when API_KEY is missing"""
        env = {
            "MINIO_ENDPOINT": "localhost:9000",
            "MINIO_ACCESS_KEY": "access",
            "MINIO_SECRET_KEY": "secret",
            "DATABASE_URL": "postgresql://localhost/db",
        }
        with patch.dict("os.environ", env, clear=True):
            is_valid, missing = validate_environment_variables()

            assert not is_valid
            assert "API_KEY" in missing

    def test_empty_api_key(self):
        """Test validation fails when API_KEY is empty"""
        env = {
            "MINIO_ENDPOINT": "localhost:9000",
            "MINIO_ACCESS_KEY": "access",
            "MINIO_SECRET_KEY": "secret",
            "DATABASE_URL": "postgresql://localhost/db",
            "API_KEY": "",
        }
        with patch.dict("os.environ", env, clear=True):
            is_valid, missing = validate_environment_variables()

            assert not is_valid
            assert "API_KEY" in missing


# ---------------------------------------------------------------------------
# Tests for validate_config_values
# ---------------------------------------------------------------------------

def _build_config_dict(**overrides):
    """Helper: build a full config dict with sensible defaults, then apply overrides."""
    defaults = {
        "model.path": "/models/yolo.pt",
        "model.confidence": 0.5,
        "model.iou_threshold": 0.45,
        "server.host": "0.0.0.0",
        "server.port": 8003,
        "minio.bucket": "models",
        "minio.object": "yolo/best.pt",
    }
    defaults.update(overrides)
    return defaults


@pytest.mark.unit
class TestValidateConfigValues:
    """Test validate_config_values function"""

    def _patch_config(self, cfg_dict):
        """Return a patch context manager that makes config.get look up *cfg_dict*."""
        mock_config = MagicMock()
        mock_config.get.side_effect = lambda key, default=None: cfg_dict.get(key, default)
        return patch("src.core.validation.config", mock_config)

    def test_all_configs_present(self):
        """All required configs present returns (True, [])"""
        cfg = _build_config_dict()
        with self._patch_config(cfg):
            is_valid, missing = validate_config_values()

            assert is_valid
            assert missing == []

    def test_one_config_missing_none(self):
        """A config that resolves to None is reported as missing"""
        cfg = _build_config_dict()
        cfg["model.path"] = None
        with self._patch_config(cfg):
            is_valid, missing = validate_config_values()

            assert not is_valid
            assert len(missing) == 1
            assert "model.path" in missing[0]
            assert "Model path" in missing[0]

    def test_config_empty_string_treated_as_missing(self):
        """A config whose value is an empty string is treated as missing"""
        cfg = _build_config_dict(**{"server.host": ""})
        with self._patch_config(cfg):
            is_valid, missing = validate_config_values()

            assert not is_valid
            assert any("server.host" in m for m in missing)

    def test_config_whitespace_only_treated_as_missing(self):
        """A config whose value is whitespace-only is treated as missing"""
        cfg = _build_config_dict(**{"minio.bucket": "   "})
        with self._patch_config(cfg):
            is_valid, missing = validate_config_values()

            assert not is_valid
            assert any("minio.bucket" in m for m in missing)

    def test_numeric_config_not_treated_as_missing(self):
        """Numeric values (e.g. server.port=8003) must not be treated as missing"""
        cfg = _build_config_dict(**{"server.port": 8003, "model.confidence": 0.5})
        with self._patch_config(cfg):
            is_valid, missing = validate_config_values()

            assert is_valid
            assert missing == []

    def test_multiple_configs_missing(self):
        """Several missing configs are all reported"""
        cfg = _build_config_dict()
        cfg["model.path"] = None
        cfg["server.host"] = ""
        cfg["minio.object"] = None
        with self._patch_config(cfg):
            is_valid, missing = validate_config_values()

            assert not is_valid
            assert len(missing) == 3
            keys_in_missing = " ".join(missing)
            assert "model.path" in keys_in_missing
            assert "server.host" in keys_in_missing
            assert "minio.object" in keys_in_missing

    def test_config_key_absent_from_dict_treated_as_missing(self):
        """If config.get returns None (key absent), it is treated as missing"""
        cfg = _build_config_dict()
        del cfg["model.iou_threshold"]
        with self._patch_config(cfg):
            is_valid, missing = validate_config_values()

            assert not is_valid
            assert any("model.iou_threshold" in m for m in missing)


# ---------------------------------------------------------------------------
# Tests for validate_startup_configuration
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestValidateStartupConfiguration:
    """Test validate_startup_configuration function"""

    @patch("src.core.validation.validate_config_values")
    @patch("src.core.validation.validate_environment_variables")
    def test_all_pass_no_exit(self, mock_env, mock_cfg):
        """When both validations pass, sys.exit is NOT called"""
        mock_env.return_value = (True, [])
        mock_cfg.return_value = (True, [])

        with patch("src.core.validation.sys.exit") as mock_exit:
            validate_startup_configuration()
            mock_exit.assert_not_called()

    @patch("src.core.validation.validate_config_values")
    @patch("src.core.validation.validate_environment_variables")
    def test_env_vars_fail_calls_exit(self, mock_env, mock_cfg):
        """When env var validation fails, sys.exit(1) is called"""
        mock_env.return_value = (False, ["API_KEY"])
        mock_cfg.return_value = (True, [])

        with patch("src.core.validation.sys.exit") as mock_exit:
            validate_startup_configuration()
            mock_exit.assert_called_once_with(1)

    @patch("src.core.validation.validate_config_values")
    @patch("src.core.validation.validate_environment_variables")
    def test_config_fail_calls_exit(self, mock_env, mock_cfg):
        """When config validation fails, sys.exit(1) is called"""
        mock_env.return_value = (True, [])
        mock_cfg.return_value = (False, ["model.path (Model path)"])

        with patch("src.core.validation.sys.exit") as mock_exit:
            validate_startup_configuration()
            mock_exit.assert_called_once_with(1)

    @patch("src.core.validation.validate_config_values")
    @patch("src.core.validation.validate_environment_variables")
    def test_both_fail_calls_exit(self, mock_env, mock_cfg):
        """When both validations fail, sys.exit(1) is still called exactly once"""
        mock_env.return_value = (False, ["MINIO_ENDPOINT"])
        mock_cfg.return_value = (False, ["server.port (Server port)"])

        with patch("src.core.validation.sys.exit") as mock_exit:
            validate_startup_configuration()
            mock_exit.assert_called_once_with(1)
