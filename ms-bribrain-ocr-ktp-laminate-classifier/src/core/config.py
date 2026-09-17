"""
Configuration Management Module
Loads and validates configuration from config.yaml
"""

import os
import re
import yaml
from dataclasses import dataclass
from typing import Any, Dict, Optional
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _substitute_env_vars(value: Any) -> Any:
    """Recursively substitute ${VAR} or ${VAR:-default} in config values."""
    if isinstance(value, dict):
        return {k: _substitute_env_vars(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_substitute_env_vars(item) for item in value]
    if isinstance(value, str):
        pattern = r'\$\{([^}]+)\}'

        def replace_var(match):
            expr = match.group(1)
            if ':-' in expr:
                var_name, default = expr.split(':-', 1)
                return os.environ.get(var_name.strip(), default)
            return os.environ.get(expr.strip(), '')

        return re.sub(pattern, replace_var, value)
    return value


@dataclass
class ElasticApmConfig:
    """Elastic APM configuration."""
    enabled: bool = False
    server_url: str = ""
    service_name: str = "ocr-ktp-laminate-classifier"
    environment: str = "dev"
    secret_token: Optional[str] = None
    verify_server_cert: bool = False


class Config:
    """Singleton configuration manager"""
    
    _instance = None
    _config: Dict[str, Any] = {}
    _startup_messages: list = []  # Store messages for deferred logging
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_config()
        return cls._instance
    
    def _load_config(self):
        """Load configuration from YAML file"""
        config_path = os.getenv("CONFIG_PATH", "./config.yaml")
        
        try:
            config_file = Path(config_path)
            if config_file.exists():
                with open(config_file, 'r') as f:
                    raw = yaml.safe_load(f) or {}
                self._config = _substitute_env_vars(raw)
                # Use print here since logger isn't available yet (circular import)
                # This runs at module import time before logging is configured
                print(f"Configuration loaded from {config_path}")
            else:
                # Default configuration
                self._config = self._get_default_config()
                print(f"Config file not found at {config_path}, using defaults")
        except Exception as e:
            print(f"Failed to load config: {str(e)}")
            self._config = self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'model': {
                'path': './models/best_cnn_detector_corrected.pth',
                'image_size': 76
            },
            'prediction': {
                'threshold': 0.5
            },
            'server': {
                'host': '0.0.0.0',
                'port': 8002
            },
            'logging': {
                'level': 'INFO',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                'file': {
                    'enabled': True,
                    'directory': './logs',
                    'filename': 'lamination_{date}.log',
                    'max_bytes': 10485760,  # 10MB
                    'backup_count': 5
                }
            },
            'device': {
                'prefer_gpu': True,
                'force_cpu': False
            }
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key (supports dot notation)"""
        keys = key.split('.')
        value: Any = self._config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        
        return value
    
    def get_all(self) -> Dict[str, Any]:
        """Get all configuration"""
        return self._config.copy()
    
    @property
    def model_path(self) -> str:
        """Get model path"""
        return self.get('model.path', './models/best_cnn_detector_corrected.pth')
    
    @property
    def image_size(self) -> int:
        """Get image size"""
        return self.get('model.image_size', 76)
    
    @property
    def threshold(self) -> float:
        """Get prediction threshold"""
        return self.get('prediction.threshold', 0.5)
    
    @property
    def server_host(self) -> str:
        """Get server host"""
        return self.get('server.host', '0.0.0.0')
    
    @property
    def server_port(self) -> int:
        """Get server port"""
        return self.get('server.port', 8002)
    @property
    def log_to_database(self) -> bool:
        """Check if logging to database is enabled"""
        return self.get('logging.log_to_database', False)
    
    @property
    def torch_compile(self) -> bool:
        """Check if torch.compile is enabled for model optimization"""
        return self.get('model.torch_compile', False)
    
    @property
    def thread_pool_workers(self) -> int:
        """Get thread pool workers for image processing"""
        return self.get('server.thread_pool_workers', 4)
    
    @property
    def max_file_size_mb(self) -> int:
        """Get maximum file size in MB"""
        return self.get('server.max_file_size_mb', 10)
    
    @property
    def max_batch_size(self) -> int:
        """Get maximum batch size"""
        return self.get('server.max_batch_size', 10)
    
    @property
    def db_pool_size(self) -> int:
        return self.get('database.pool_size', 5)

    @property
    def db_max_overflow(self) -> int:
        return self.get('database.max_overflow', 10)

    @property
    def db_pool_pre_ping(self) -> bool:
        return self.get('database.pool_pre_ping', True)

    @property
    def db_pool_recycle(self) -> int:
        return self.get('database.pool_recycle', 3600)

    @property
    def db_pool_timeout(self) -> int:
        return self.get('database.pool_timeout', 10)

    @property
    def db_connect_timeout(self) -> int:
        return self.get('database.connect_timeout', 5)

    # GCS settings
    @property
    def gcs_bucket_name(self) -> str:
        return self.get('gcs.bucket_name', '')

    @property
    def gcs_prefix(self) -> str:
        return self.get('gcs.prefix', '')

    @property
    def gcs_model_prefix(self) -> str:
        return self.get('gcs.model_prefix', '')

    @property
    def elastic_apm(self) -> ElasticApmConfig:
        """Typed Elastic APM configuration."""
        data = self.get('elastic_apm') or {}
        return ElasticApmConfig(
            enabled=bool(data.get('enabled', False)),
            server_url=data.get('server_url', '') or '',
            service_name=data.get('service_name') or 'ocr-ktp-laminate-classifier',
            environment=data.get('environment') or 'dev',
            secret_token=data.get('secret_token') or None,
            verify_server_cert=bool(data.get('verify_server_cert', False)),
        )

    @property
    def cors_config(self) -> Dict[str, Any]:
        """Get CORS configuration"""
        cors = self.get('cors', {})
        return {
            'allow_origins': cors.get('allow_origins', ["*"]) if cors else ["*"],
            'allow_credentials': cors.get('allow_credentials', True) if cors else True,
            'allow_methods': cors.get('allow_methods', ["*"]) if cors else ["*"],
            'allow_headers': cors.get('allow_headers', ["*"]) if cors else ["*"],
        }


# Global config instance
config = Config()
