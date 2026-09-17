"""
Unit tests for core configuration module
"""

import pytest
import yaml
from pydantic import ValidationError

from src.core.config import (
    Config,
    AppConfig,
    LoggingConfig,
    DeviceConfig,
    ModelConfig,
    ServerConfig,
    load_config,
    get_config,
    reload_config,
    _substitute_env_vars
)


@pytest.mark.unit
class TestConfigModels:
    """Tests for configuration model classes"""
    
    def test_app_config_defaults(self):
        """Test AppConfig default values"""
        config = AppConfig()
        
        assert config.name == "FastAPI Application"
        assert config.version == "1.0.0"
        assert config.environment == "development"
        assert config.debug is False
    
    def test_device_config_validates_device(self):
        """Test DeviceConfig validates device type"""
        config = DeviceConfig(preferred="cuda")
        assert config.preferred == "cuda"
        
        with pytest.raises(ValidationError):
            DeviceConfig(preferred="invalid_device")
    
    def test_logging_config_validates_level(self):
        """Test LoggingConfig validates log level"""
        config = LoggingConfig(level="INFO")
        assert config.level == "INFO"
        
        config = LoggingConfig(level="debug")  # Case insensitive
        assert config.level == "DEBUG"
        
        with pytest.raises(ValidationError):
            LoggingConfig(level="INVALID")
    
    def test_model_config(self):
        """Test ModelConfig"""
        config = ModelConfig(
            path="/models/my_model",
            image_size=224
        )
        
        assert config.path == "/models/my_model"
        assert config.image_size == 224
    
    def test_server_config_port_validation(self):
        """Test ServerConfig validates port range"""
        config = ServerConfig(host="localhost", port=8000)
        assert config.port == 8000
        
        with pytest.raises(ValidationError):
            ServerConfig(port=100)  # Port too low
        
        with pytest.raises(ValidationError):
            ServerConfig(port=70000)  # Port too high


@pytest.mark.unit
class TestSubstituteEnvVars:
    """Tests for environment variable substitution"""
    
    def test_substitute_simple_env_var(self, monkeypatch):
        """Test substituting simple environment variable"""
        monkeypatch.setenv("TEST_VAR", "test_value")
        
        result = _substitute_env_vars("${TEST_VAR}")
        
        assert result == "test_value"
    
    def test_substitute_with_default_value(self, monkeypatch):
        """Test substituting with default value when env var doesn't exist"""
        monkeypatch.delenv("MISSING_VAR", raising=False)
        
        result = _substitute_env_vars("${MISSING_VAR:-default}")
        
        assert result == "default"
    
    def test_substitute_with_default_when_var_exists(self, monkeypatch):
        """Test that actual value is used when var exists, even with default"""
        monkeypatch.setenv("EXISTING_VAR", "actual_value")
        
        result = _substitute_env_vars("${EXISTING_VAR:-default}")
        
        assert result == "actual_value"
    
    def test_substitute_in_nested_dict(self, monkeypatch):
        """Test substitution in nested dictionary"""
        monkeypatch.setenv("HOST", "localhost")
        monkeypatch.setenv("PORT", "8000")
        
        data = {
            "server": {
                "host": "${HOST}",
                "port": "${PORT:-9000}"
            }
        }
        
        result = _substitute_env_vars(data)
        
        assert result["server"]["host"] == "localhost"
        assert result["server"]["port"] == "8000"
    
    def test_substitute_in_list(self, monkeypatch):
        """Test substitution in lists"""
        monkeypatch.setenv("ORIGIN", "http://localhost")
        
        data = ["${ORIGIN}", "http://example.com"]
        
        result = _substitute_env_vars(data)
        
        assert result[0] == "http://localhost"
        assert result[1] == "http://example.com"
    
    def test_substitute_raises_on_missing_required_var(self, monkeypatch):
        """Test that missing required var raises ValueError"""
        monkeypatch.delenv("REQUIRED_VAR", raising=False)
        
        with pytest.raises(ValueError, match="Environment variable REQUIRED_VAR not found"):
            _substitute_env_vars("${REQUIRED_VAR}")
    
    def test_substitute_non_string_values(self):
        """Test that non-string values are returned unchanged"""
        assert _substitute_env_vars(123) == 123
        assert _substitute_env_vars(True) is True
        assert _substitute_env_vars(None) is None


@pytest.mark.unit
class TestLoadConfig:
    """Tests for load_config function"""
    
    def test_load_config_from_yaml(self, tmp_path):
        """Test loading configuration from YAML file"""
        config_file = tmp_path / "test_config.yaml"
        config_data = {
            "app": {"name": "Test App", "version": "1.0.0"},
            "server": {"host": "0.0.0.0", "port": 8000},
            "logging": {"level": "INFO"}
        }
        
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)
        
        config = load_config(str(config_file))
        
        assert config.app.name == "Test App"
        assert config.server.port == 8000
        assert config.logging.level == "INFO"
    
    def test_load_config_file_not_found(self):
        """Test that FileNotFoundError is raised for missing file"""
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent_config.yaml")
    
    def test_load_config_empty_file(self, tmp_path):
        """Test loading empty YAML file"""
        config_file = tmp_path / "empty.yaml"
        config_file.write_text("")
        
        config = load_config(str(config_file))
        
        # Should use default values
        assert isinstance(config, Config)
    
    def test_load_config_with_env_substitution(self, tmp_path, monkeypatch):
        """Test loading config with environment variable substitution"""
        monkeypatch.setenv("TEST_PORT", "9000")
        
        config_file = tmp_path / "test_config.yaml"
        config_data = {
            "server": {"port": "${TEST_PORT}"}
        }
        
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)
        
        config = load_config(str(config_file))
        
        assert config.server.port == 9000


@pytest.mark.unit
class TestGetConfig:
    """Tests for get_config singleton"""
    
    def test_get_config_returns_same_instance(self):
        """Test that get_config returns same instance"""
        config1 = get_config()
        config2 = get_config()
        
        assert config1 is config2
    
    def test_reload_config_creates_new_instance(self, tmp_path):
        """Test that reload_config creates new instance"""
        config_file = tmp_path / "test_config.yaml"
        config_data = {
            "app": {"name": "Test App"}
        }
        
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)
        
        get_config(str(config_file))
        
        # Update config file
        config_data["app"]["name"] = "Updated App"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)
        
        config2 = reload_config(str(config_file))
        
        assert config2.app.name == "Updated App"


@pytest.mark.unit
class TestFullConfigIntegration:
    """Integration tests for full configuration"""
    
    def test_full_config_structure(self, sample_config_dict, tmp_path):
        """Test loading a complete configuration"""
        config_file = tmp_path / "full_config.yaml"
        
        with open(config_file, "w") as f:
            yaml.dump(sample_config_dict, f)
        
        config = load_config(str(config_file))
        
        assert config.app.name == "Test App"
        assert config.server.port == 8000
        assert config.device.force_cpu is True
        assert config.model.image_size == 320
        assert config.logging.level == "INFO"
