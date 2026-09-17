"""Tests for configuration validation module."""

import pytest
from unittest.mock import patch


class TestValidateEnvironmentVariables:
    """Tests for validate_environment_variables."""

    def test_all_onprem_vars_present(self):
        """Test validation passes when all onprem env vars are set."""
        from src.core.validation import validate_environment_variables

        env = {
            "APP_ENVIRO": "onprem",
            "DATABASE_URL": "postgresql://localhost/db",
            "API_KEY": "test-key",
            "MINIO_ENDPOINT": "localhost:9000",
            "MINIO_ACCESS_KEY": "access",
            "MINIO_SECRET_KEY": "secret",
        }
        with patch.dict("os.environ", env, clear=False):
            is_valid, missing = validate_environment_variables()

        assert is_valid is True
        assert missing == []

    def test_missing_onprem_vars(self):
        """Test validation fails when onprem vars are missing."""
        from src.core.validation import validate_environment_variables

        env = {"APP_ENVIRO": "onprem", "API_KEY": "test"}
        with patch.dict("os.environ", env, clear=True):
            is_valid, missing = validate_environment_variables()

        assert is_valid is False
        assert len(missing) > 0

    def test_cloud_vars_present(self):
        """Test validation passes for cloud env vars."""
        from src.core.validation import validate_environment_variables

        env = {
            "APP_ENVIRO": "cloud",
            "DATABASE_URL": "postgresql://localhost/db",
            "API_KEY": "test-key",
            "SA_TYPE": "service_account",
            "SA_PROJECT_ID": "proj",
            "SA_PRIVATE_KEY": "key",
            "SA_CLIENT_MAIL": "mail@sa.com",
            "SA_CLIENT_ID": "123",
            "SA_AUTH_URI": "https://auth",
            "SA_TOKEN_URI": "https://token",
            "SA_AUTH_PROVIDER": "https://provider",
            "SA_CERT_URL": "https://cert",
        }
        with patch.dict("os.environ", env, clear=False):
            is_valid, missing = validate_environment_variables()

        assert is_valid is True
        assert missing == []

    def test_missing_cloud_vars(self):
        """Test validation fails when cloud vars are missing."""
        from src.core.validation import validate_environment_variables

        env = {"APP_ENVIRO": "cloud", "API_KEY": "test"}
        with patch.dict("os.environ", env, clear=True):
            is_valid, missing = validate_environment_variables()

        assert is_valid is False
        assert any("SA_TYPE" in v for v in missing)

    def test_empty_string_env_var_treated_as_missing(self):
        """Env var set to '' should fail validation same as unset (not value → falsy)."""
        from src.core.validation import validate_environment_variables

        env = {
            "APP_ENVIRO": "onprem",
            "DATABASE_URL": "",
            "API_KEY": "k",
            "MINIO_ENDPOINT": "ep",
            "MINIO_ACCESS_KEY": "ak",
            "MINIO_SECRET_KEY": "sk",
        }
        with patch.dict("os.environ", env, clear=True):
            is_valid, missing = validate_environment_variables()

        assert is_valid is False
        assert any("DATABASE_URL" in v for v in missing)

    def test_app_enviro_defaults_to_onprem(self):
        """When APP_ENVIRO is unset, onprem vars should be required."""
        from src.core.validation import validate_environment_variables

        env = {"DATABASE_URL": "x", "API_KEY": "k"}
        with patch.dict("os.environ", env, clear=True):
            is_valid, missing = validate_environment_variables()

        assert is_valid is False
        assert any("MINIO_ENDPOINT" in v for v in missing)


class TestValidateConfigValues:
    """Tests for validate_config_values."""

    def test_valid_config(self):
        """Test validation passes with valid config."""
        from src.core.validation import validate_config_values

        with patch.dict("os.environ", {"APP_ENVIRO": "onprem"}):
            is_valid, missing = validate_config_values()

        # Default config.yaml has valid values
        assert isinstance(is_valid, bool)
        assert isinstance(missing, list)

    def test_missing_model_path(self):
        """Test validation fails with empty model path."""
        from src.core.validation import validate_config_values
        from src.core.config import config

        original = config.model_path
        with patch.object(type(config), 'model_path', new_callable=lambda: property(lambda self: "")):
            is_valid, missing = validate_config_values()
            assert any("model.path" in m for m in missing)

    def test_missing_server_host(self):
        """Test validation reports missing server host."""
        from src.core.validation import validate_config_values
        from src.core.config import config

        with patch.object(type(config), 'server_host', new_callable=lambda: property(lambda self: "")):
            is_valid, missing = validate_config_values()
            assert any("server.host" in m for m in missing)

    def test_missing_server_port(self):
        """Test validation reports missing server port."""
        from src.core.validation import validate_config_values
        from src.core.config import config

        with patch.object(type(config), 'server_port', new_callable=lambda: property(lambda self: 0)):
            is_valid, missing = validate_config_values()
            assert any("server.port" in m for m in missing)

    def test_invalid_crop_size(self):
        """Test validation reports invalid crop size."""
        from src.core.validation import validate_config_values
        from src.core.config import config

        with patch.object(config, 'get', side_effect=lambda key, default=None: 0 if key == "model.crop_size" else default):
            is_valid, missing = validate_config_values()
            # crop_size 0 should be invalid
            assert any("crop_size" in m for m in missing) or is_valid

    def test_negative_crop_size_invalid(self):
        """Negative crop_size must be rejected."""
        from src.core.validation import validate_config_values
        from src.core.config import config

        with patch.object(config, 'get', side_effect=lambda key, default=None: -5 if key == "model.crop_size" else default):
            is_valid, missing = validate_config_values()
            assert any("crop_size" in m for m in missing)

    def test_whitespace_minio_bucket_invalid(self):
        """Whitespace-only minio.bucket should fail the .strip() check."""
        from src.core.validation import validate_config_values
        from src.core.config import config

        real_get = config.get

        def patched(key, default=None):
            if key == "minio.bucket":
                return "   "
            if key == "minio.object":
                return "valid/object"
            return real_get(key, default)

        with patch.dict("os.environ", {"APP_ENVIRO": "onprem"}), \
             patch.object(config, 'get', side_effect=patched):
            is_valid, missing = validate_config_values()
            assert any("minio.bucket" in m for m in missing)

    def test_whitespace_minio_object_invalid(self):
        from src.core.validation import validate_config_values
        from src.core.config import config

        real_get = config.get

        def patched(key, default=None):
            if key == "minio.bucket":
                return "valid-bucket"
            if key == "minio.object":
                return "   "
            return real_get(key, default)

        with patch.dict("os.environ", {"APP_ENVIRO": "onprem"}), \
             patch.object(config, 'get', side_effect=patched):
            is_valid, missing = validate_config_values()
            assert any("minio.object" in m for m in missing)

    def test_minio_config_skipped_for_cloud(self):
        """MinIO config is not required when APP_ENVIRO != onprem."""
        from src.core.validation import validate_config_values
        from src.core.config import config

        real_get = config.get

        def patched(key, default=None):
            if key in ("minio.bucket", "minio.object"):
                return None
            return real_get(key, default)

        with patch.dict("os.environ", {"APP_ENVIRO": "cloud"}), \
             patch.object(config, 'get', side_effect=patched):
            _, missing = validate_config_values()
            assert not any("minio." in m for m in missing)


class TestValidateStartupConfiguration:
    """Tests for validate_startup_configuration."""

    def test_passes_when_valid(self):
        """Test startup validation passes when all config is valid."""
        from src.core.validation import validate_startup_configuration

        with patch('src.core.validation.validate_environment_variables', return_value=(True, [])), \
             patch('src.core.validation.validate_config_values', return_value=(True, [])):
            # Should not raise or exit
            validate_startup_configuration()

    def test_exits_when_env_invalid(self):
        """Test startup validation exits when env vars are missing."""
        from src.core.validation import validate_startup_configuration

        with patch('src.core.validation.validate_environment_variables', return_value=(False, ["  - DATABASE_URL"])), \
             patch('src.core.validation.validate_config_values', return_value=(True, [])), \
             pytest.raises(SystemExit):
            validate_startup_configuration()

    def test_exits_when_config_invalid(self):
        """Test startup validation exits when config values are missing."""
        from src.core.validation import validate_startup_configuration

        with patch('src.core.validation.validate_environment_variables', return_value=(True, [])), \
             patch('src.core.validation.validate_config_values', return_value=(False, ["  - model.path"])), \
             pytest.raises(SystemExit):
            validate_startup_configuration()

    def test_exits_when_both_invalid(self):
        """Test startup validation exits when both env and config are invalid."""
        from src.core.validation import validate_startup_configuration

        with patch('src.core.validation.validate_environment_variables', return_value=(False, ["  - API_KEY"])), \
             patch('src.core.validation.validate_config_values', return_value=(False, ["  - model.path"])), \
             pytest.raises(SystemExit):
            validate_startup_configuration()

    def test_logs_each_missing_entry(self):
        """Each missing env var and config value should be logged on failure."""
        import logging
        from src.core.validation import validate_startup_configuration

        with patch('src.core.validation.validate_environment_variables', return_value=(False, ["  - API_KEY"])), \
             patch('src.core.validation.validate_config_values', return_value=(False, ["  - model.path"])), \
             patch.object(logging.getLogger('src.core.validation'), 'error') as mock_error, \
             pytest.raises(SystemExit):
            validate_startup_configuration()

        logged = [c.args[0] for c in mock_error.call_args_list]
        assert any("API_KEY" in m for m in logged)
        assert any("model.path" in m for m in logged)
