"""Tests for configuration management."""

import pytest
from unittest.mock import patch, mock_open


class TestConfig:
    """Test cases for Config class."""

    @pytest.fixture(autouse=True)
    def reset_config_singleton(self):
        """Reset Config singleton before each test."""
        from src.core.config import Config
        Config._instance = None
        yield
        Config._instance = None

    def test_config_singleton(self):
        """Test that Config is a singleton."""
        from src.core.config import Config
        config1 = Config()
        config2 = Config()
        assert config1 is config2

    def test_config_get_with_default(self):
        """Test getting config value with default."""
        from src.core.config import Config
        config = Config()
        # Should return default for non-existent key
        assert config.get('non.existent.key', 'default') == 'default'

    def test_config_get_nested_value(self):
        """Test getting nested config value using dot notation."""
        from src.core.config import Config
        config = Config()
        # Test with actual config values or default fallback
        model_path = config.get('model.path')
        assert model_path is not None

    def test_config_properties(self):
        """Test config property accessors."""
        from src.core.config import Config
        config = Config()
        # Test that properties return values (not necessarily specific values)
        assert config.model_path is not None
        assert isinstance(config.server_host, str)
        assert isinstance(config.server_port, int)

    def test_config_default_values(self):
        """Test default config when file doesn't exist."""
        from src.core.config import Config
        with patch('pathlib.Path.exists', return_value=False):
            config = Config()
            defaults = config.get_all()
            assert 'model' in defaults
            assert 'server' in defaults
            assert 'logging' in defaults

    def test_config_missing_file_uses_defaults(self):
        """Test that missing config file results in defaults."""
        from src.core.config import Config
        with patch('pathlib.Path.exists', return_value=False):
            config = Config()
            # Default server port from _get_default_config
            assert config.server_port in [8000, 8001, 8002, 8080]  # Accept common defaults

    def test_config_get_all(self):
        """Test get_all returns copy of config."""
        from src.core.config import Config
        config = Config()
        all_config = config.get_all()
        assert isinstance(all_config, dict)
        # Modifying returned dict shouldn't affect internal state
        all_config['test'] = 'value'
        assert config.get('test') is None

    def test_config_invalid_yaml_uses_defaults(self):
        """Test that invalid YAML falls back to defaults."""
        from src.core.config import Config
        invalid_yaml = "invalid: yaml: content: ["
        with patch('builtins.open', mock_open(read_data=invalid_yaml)):
            with patch('pathlib.Path.exists', return_value=True):
                config = Config()
                # Should fall back to defaults
                assert config.server_port in [8000, 8001, 8002, 8080]

    def test_get_returns_default_when_intermediate_not_dict(self):
        """Dot-path lookup through a scalar intermediate returns the default."""
        from src.core.config import Config
        config = Config()
        config._config = {"server": {"port": 8002}}
        # server.port.extra cannot descend into an int — should fall through to default
        assert config.get("server.port.extra", "fallback") == "fallback"

    def test_all_property_accessors_return_defaults_when_empty_config(self):
        """Every @property should return its fallback when config is empty."""
        from src.core.config import Config
        config = Config()
        config._config = {}
        assert config.model_path.endswith(".pth")
        assert config.image_size == 76
        assert config.threshold == 0.5
        assert config.server_host == "0.0.0.0"
        assert config.server_port == 8002
        assert config.log_to_database is True
        assert config.db_pool_size == 5
        assert config.db_max_overflow == 10
        assert config.db_pool_pre_ping is True
        assert config.db_pool_recycle == 3600
        assert config.db_pool_timeout == 10
        assert config.db_connect_timeout == 5
        assert config.gcs_bucket_name == ""
        assert config.gcs_prefix == ""
        assert config.gcs_model_prefix == ""
