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

        env = {"APP_ENVIRO": "onprem"}
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

        env = {"APP_ENVIRO": "cloud"}
        with patch.dict("os.environ", env, clear=True):
            is_valid, missing = validate_environment_variables()

        assert is_valid is False
        assert any("SA_TYPE" in v for v in missing)


class TestValidateConfigValues:
    """Tests for validate_config_values."""

    def test_valid_config(self):
        """Test validation with default config."""
        from src.core.validation import validate_config_values

        is_valid, missing = validate_config_values()
        assert isinstance(is_valid, bool)
        assert isinstance(missing, list)

    def test_invalid_image_size(self):
        """Test validation reports invalid image size."""
        from src.core.validation import validate_config_values
        from src.core.config import config

        original = config.model.image_size
        try:
            config.model.image_size = 0
            is_valid, missing = validate_config_values()
            assert any("image_size" in m for m in missing)
        finally:
            config.model.image_size = original


class TestValidateStartupConfiguration:
    """Tests for validate_startup_configuration."""

    def test_passes_when_valid(self):
        """Test startup validation passes when all config is valid."""
        from src.core.validation import validate_startup_configuration

        with patch('src.core.validation.validate_environment_variables', return_value=(True, [])), \
             patch('src.core.validation.validate_config_values', return_value=(True, [])):
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
