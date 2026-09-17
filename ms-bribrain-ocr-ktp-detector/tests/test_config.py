"""
Unit tests for core.config module
"""
import os
from unittest.mock import patch

import pytest

from src.core.config import Config


@pytest.mark.unit
class TestConfig:
    """Test Config class"""
    
    def test_config_singleton(self, clean_env):
        """Test that Config is a singleton"""
        config1 = Config()
        config2 = Config()
        assert config1 is config2
    
    def test_load_config_from_file(self, temp_config_file, clean_env):
        """Test loading configuration from YAML file"""
        os.environ['CONFIG_PATH'] = str(temp_config_file)
        
        # Force reload by clearing singleton
        Config._instance = None
        config = Config()
        
        assert config.model_path == './src/models/test_model.pt'
        assert config.confidence == 0.5
        assert config.iou_threshold == 0.45
    
    def test_load_config_default_when_file_not_found(self, clean_env):
        """Test default configuration when file not found"""
        os.environ['CONFIG_PATH'] = './nonexistent_config.yaml'
        
        Config._instance = None
        config = Config()
        
        # Should have default values
        assert config.model_path == './src/models/best.pt'
        assert config.confidence == 0.5
    
    def test_get_with_dot_notation(self, mock_config, clean_env):
        """Test get method with dot notation"""
        Config._instance = None
        config = Config()
        config._config = mock_config
        
        assert config.get('model.path') == './src/models/test_model.pt'
        assert config.get('model.confidence') == 0.5
        # Note: config.get doesn't support integer keys in path, test dict access instead
        class_names = config.get('classes.names')
        assert class_names[0] == 'ktp'
    
    def test_get_with_default(self, mock_config, clean_env):
        """Test get method with default value"""
        Config._instance = None
        config = Config()
        config._config = mock_config
        
        assert config.get('nonexistent.key', 'default') == 'default'
        assert config.get('model.nonexistent', 100) == 100
    
    def test_get_all(self, mock_config, clean_env):
        """Test get_all method"""
        Config._instance = None
        config = Config()
        config._config = mock_config
        
        all_config = config.get_all()
        assert isinstance(all_config, dict)
        assert 'model' in all_config
        assert 'server' in all_config
    
    def test_model_path_property(self, mock_config, clean_env):
        """Test model_path property"""
        Config._instance = None
        config = Config()
        config._config = mock_config
        
        assert config.model_path == './src/models/test_model.pt'
    
    def test_confidence_property(self, mock_config, clean_env):
        """Test confidence property"""
        Config._instance = None
        config = Config()
        config._config = mock_config
        
        assert config.confidence == 0.5
    
    def test_iou_threshold_property(self, mock_config, clean_env):
        """Test iou_threshold property"""
        Config._instance = None
        config = Config()
        config._config = mock_config
        
        assert config.iou_threshold == 0.45
    
    def test_class_names_property(self, mock_config, clean_env):
        """Test class_names property"""
        Config._instance = None
        config = Config()
        config._config = mock_config
        
        class_names = config.class_names
        assert isinstance(class_names, dict)
        assert class_names[0] == 'ktp'
        assert class_names[1] == 'non-ktp'
    
    def test_server_host_property(self, mock_config, clean_env):
        """Test server_host property"""
        Config._instance = None
        config = Config()
        config._config = mock_config
        
        assert config.server_host == '0.0.0.0'
    
    def test_server_port_property(self, mock_config, clean_env):
        """Test server_port property"""
        Config._instance = None
        config = Config()
        config._config = mock_config
        
        assert config.server_port == 8003
    
    def test_log_to_database_property(self, mock_config, clean_env):
        """Test log_to_database property"""
        Config._instance = None
        config = Config()
        config._config = mock_config
        
        assert not config.log_to_database
    
    def test_cors_config_property(self, mock_config, clean_env):
        """Test cors_config property"""
        Config._instance = None
        config = Config()
        config._config = mock_config
        
        cors = config.cors_config
        assert cors['allow_origins'] == ["*"]
        assert cors['allow_credentials']
        assert cors['allow_methods'] == ["*"]
        assert cors['allow_headers'] == ["*"]
    
    def test_config_load_failure_uses_default(self, clean_env):
        """Test that config load failure falls back to defaults"""
        with patch('builtins.open', side_effect=PermissionError("Access denied")):
            Config._instance = None
            config = Config()
            
            # Should have default values
            assert config.model_path == './src/models/best.pt'
