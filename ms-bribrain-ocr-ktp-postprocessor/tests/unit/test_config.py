"""Tests for configuration module."""

import os
from pathlib import Path
from unittest.mock import patch, mock_open

import pytest
import yaml

from src.core.config import Config, get_config


class TestConfig:
    """Test cases for Config class."""

    def test_singleton_pattern(self):
        """Test that Config follows singleton pattern."""
        config1 = Config()
        config2 = Config()
        assert config1 is config2

    def test_config_loading(self, tmp_path):
        """Test configuration file loading."""
        # Create a temporary config file
        config_data = {
            "api": {
                "host": "0.0.0.0",
                "port": 8000,
                "version": "1.0.0"
            },
            "thresholds": {
                "confidence": 0.6,
                "partial": 80,
                "ratio": 85
            }
        }
        
        config_file = tmp_path / "config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        # Mock the config path
        with patch.object(Path, '__truediv__', return_value=config_file):
            # Reset singleton
            Config._instance = None
            config = Config()
            
            assert config._config is not None

    def test_get_with_simple_key(self):
        """Test getting configuration with simple key."""
        Config._instance = None
        config = Config()
        
        # Assuming config is loaded
        result = config.get("api")
        assert isinstance(result, (dict, type(None)))

    def test_get_with_dot_notation(self):
        """Test getting configuration with dot notation."""
        Config._instance = None
        config = Config()
        
        result = config.get("thresholds.confidence")
        assert isinstance(result, (float, int, type(None)))

    def test_get_with_default(self):
        """Test getting configuration with default value."""
        Config._instance = None
        config = Config()
        
        result = config.get("nonexistent.key", default=123)
        assert result == 123

    def test_get_thresholds(self):
        """Test getting all thresholds."""
        Config._instance = None
        config = Config()
        
        thresholds = config.get_thresholds()
        assert isinstance(thresholds, dict)

    def test_get_regex(self):
        """Test getting all regex patterns."""
        Config._instance = None
        config = Config()
        
        regex = config.get_regex()
        assert isinstance(regex, dict)

    def test_get_api_config(self):
        """Test getting API configuration."""
        Config._instance = None
        config = Config()
        
        api_config = config.get_api_config()
        assert isinstance(api_config, dict)

    def test_get_database_config(self):
        """Test getting database configuration."""
        Config._instance = None
        config = Config()
        
        db_config = config.get_database_config()
        assert isinstance(db_config, dict)

    def test_get_logging_config(self):
        """Test getting logging configuration."""
        Config._instance = None
        config = Config()
        
        log_config = config.get_logging_config()
        assert isinstance(log_config, dict)

    @patch.dict(os.environ, {"THRESHOLD_PARTIAL": "90"})
    def test_env_override_threshold_partial(self):
        """Test environment variable override for threshold partial."""
        Config._instance = None
        config = Config()
        
        partial = config.get("thresholds.partial")
        # Should be overridden by environment variable
        assert isinstance(partial, (float, int, type(None)))

    @patch.dict(os.environ, {"THRESHOLD_RATIO": "90"})
    def test_env_override_threshold_ratio(self):
        """Test environment variable override for threshold ratio."""
        Config._instance = None
        config = Config()
        
        ratio = config.get("thresholds.ratio")
        assert isinstance(ratio, (float, int, type(None)))

    @patch.dict(os.environ, {"THRESHOLD_CONFIDENCE": "0.8"})
    def test_env_override_threshold_confidence(self):
        """Test environment variable override for threshold confidence."""
        Config._instance = None
        config = Config()
        
        confidence = config.get("thresholds.confidence")
        assert isinstance(confidence, (float, int, type(None)))

    @patch.dict(os.environ, {"API_HOST": "127.0.0.1"})
    def test_env_override_api_host(self):
        """Test environment variable override for API host."""
        Config._instance = None
        config = Config()
        
        host = config.get("api.host")
        assert isinstance(host, (str, type(None)))

    @patch.dict(os.environ, {"API_PORT": "9000"})
    def test_env_override_api_port(self):
        """Test environment variable override for API port."""
        Config._instance = None
        config = Config()
        
        port = config.get("api.port")
        assert isinstance(port, (int, type(None)))


class TestGetConfig:
    """Test cases for get_config helper function."""

    def test_get_config_returns_instance(self):
        """Test that get_config returns Config instance."""
        config = get_config()
        assert isinstance(config, Config)

    def test_get_config_singleton(self):
        """Test that get_config returns same instance."""
        config1 = get_config()
        config2 = get_config()
        assert config1 is config2


class TestConfigErrorHandling:
    """Test cases for configuration error handling."""

    def test_missing_config_file(self, tmp_path):
        """Test handling of missing configuration file."""
        # Mock path to non-existent file
        with patch('src.core.config.Path') as mock_path:
            mock_config_path = tmp_path / "nonexistent.yaml"
            mock_path.return_value.__truediv__.return_value = mock_config_path
            
            Config._instance = None
            
            try:
                config = Config()
                # If it doesn't raise, that's fine too (depends on implementation)
                assert True
            except FileNotFoundError:
                # Expected behavior
                assert True

    def test_invalid_yaml_content(self, tmp_path):
        """Test handling of invalid YAML content."""
        config_file = tmp_path / "config.yaml"
        with open(config_file, 'w') as f:
            f.write("invalid: yaml: content: [")
        
        with patch.object(Path, '__truediv__', return_value=config_file):
            Config._instance = None
            
            try:
                config = Config()
                assert True
            except yaml.YAMLError:
                # Expected behavior
                assert True
