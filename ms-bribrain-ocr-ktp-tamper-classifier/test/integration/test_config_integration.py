"""
Integration tests for configuration system
Tests configuration loading and usage across components
"""

import pytest
import yaml


@pytest.mark.integration
class TestConfigurationIntegration:
    """Integration tests for configuration system"""
    
    def test_config_loads_and_validates(self, tmp_path):
        """Test configuration loads and validates correctly"""
        from src.core.config import load_config
        
        config_file = tmp_path / "test_config.yaml"
        config_data = {
            "app": {
                "name": "Test App",
                "version": "1.0.0",
                "environment": "test"
            },
            "server": {
                "host": "0.0.0.0",
                "port": 8000
            },
            "logging": {
                "level": "INFO",
                "log_to_database": False
            },
            "device": {
                "force_cpu": True,
                "preferred": "cpu"
            },
            "model": {
                "path": "/models/test",
                "image_size": 320,
                "file_path": "/tmp/model.zip"
            },
            "prediction": {
                "threshold": 0.5,
                "batch_size_limit": 10
            },
            "cors": {
                "allow_origins": ["*"],
                "allow_credentials": True,
                "allow_methods": ["*"],
                "allow_headers": ["*"]
            },
            "minio": {
                "bucket": "test-bucket",
                "object": "model.zip"
            }
        }
        
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)
        
        config = load_config(str(config_file))
        
        assert config.app.name == "Test App"
        assert config.server.port == 8000
        assert config.device.force_cpu is True
        assert config.model.image_size == 320
    
    def test_config_with_environment_variables(self, tmp_path, monkeypatch):
        """Test configuration with environment variable substitution"""
        from src.core.config import load_config
        
        monkeypatch.setenv("TEST_HOST", "localhost")
        monkeypatch.setenv("TEST_PORT", "9000")
        
        config_file = tmp_path / "test_config.yaml"
        config_data = {
            "server": {
                "host": "${TEST_HOST}",
                "port": "${TEST_PORT}"
            }
        }
        
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)
        
        config = load_config(str(config_file))
        
        assert config.server.host == "localhost"
        assert config.server.port == 9000
    
    def test_config_validation_fails_on_invalid_data(self, tmp_path):
        """Test that config validation fails on invalid data"""
        from src.core.config import load_config
        from pydantic import ValidationError
        
        config_file = tmp_path / "invalid_config.yaml"
        config_data = {
            "server": {
                "port": 100  # Invalid port (too low)
            }
        }
        
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)
        
        with pytest.raises(ValidationError):
            load_config(str(config_file))
    
    def test_config_singleton_pattern(self, tmp_path):
        """Test that config uses singleton pattern"""
        from src.core.config import get_config
        
        config_file = tmp_path / "test_config.yaml"
        config_data = {"app": {"name": "Singleton Test"}}
        
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)
        
        config1 = get_config(str(config_file))
        config2 = get_config(str(config_file))
        
        assert config1 is config2
    
    def test_config_reload(self, tmp_path):
        """Test configuration can be reloaded"""
        from src.core.config import get_config, reload_config
        
        config_file = tmp_path / "test_config.yaml"
        
        # Initial config
        config_data = {"app": {"name": "Original"}}
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)
        
        config1 = get_config(str(config_file), force_reload=True)
        assert config1.app.name == "Original"
        
        # Update config
        config_data["app"]["name"] = "Updated"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)
        
        config2 = reload_config(str(config_file))
        assert config2.app.name == "Updated"


@pytest.mark.integration
class TestConfigUsageInServices:
    """Test configuration usage in services"""
    
    def test_config_used_by_tamper_detection_service(self):
        """Test that tamper detection service uses config"""
        from src.services.tamper_detection import TamperDetectionService
        from src.core.config import config
        
        service = TamperDetectionService(
            model_path=config.model.path,
            image_size=config.model.image_size,
            force_cpu=config.device.force_cpu
        )
        
        assert service.image_size == config.model.image_size
        assert service.force_cpu == config.device.force_cpu
    
    def test_config_used_by_api_routes(self):
        """Test that API routes use config"""
        from src.core.config import config
        
        # Config should have prediction threshold
        assert hasattr(config, 'prediction')
        assert hasattr(config.prediction, 'threshold')
        assert 0 <= config.prediction.threshold <= 1
