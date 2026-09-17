"""
Configuration management for OCR Graycopy service
Loads and validates configuration from YAML file
"""

import os
import re
from pathlib import Path
from typing import Any, Optional, List

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

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


class ModelConfig(BaseModel):
    """Model configuration"""

    path: str = Field(default="./models/graycopy_model.pth")
    image_size: int = Field(default=224)
    normalize_size: int = Field(default=512)
    dropout_rate: float = Field(default=0.6)
    normalize_mean: list[float] = Field(default=[0.485, 0.456, 0.406])
    normalize_std: list[float] = Field(default=[0.229, 0.224, 0.225])


class PredictionConfig(BaseModel):
    """Prediction configuration"""

    threshold: float = Field(default=0.5, ge=0.0, le=1.0)


class ServerConfig(BaseModel):
    """Server configuration"""

    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8020, ge=1, le=65535)
    thread_pool_workers: int = Field(default=4, ge=1, le=32)
    max_file_size_mb: int = Field(default=10, ge=1, description="Maximum uploaded image size in MB")


class LoggingConfig(BaseModel):
    """Logging configuration"""

    level: str = Field(default="INFO")
    file: str = Field(default="logs/ocr_graycopy.log")
    max_bytes: int = Field(default=10485760)  # 10MB
    backup_count: int = Field(default=5)
    log_to_database: bool = Field(default=False)


class DeviceConfig(BaseModel):
    """Device configuration"""

    prefer_gpu: bool = Field(default=True)
    force_cpu: bool = Field(
        default=False, description="Force CPU usage even if GPU is available"
    )
    compile_model: bool = Field(
        default=True, description="Enable torch.compile() for CUDA devices"
    )
    use_jit_cpu: bool = Field(
        default=True,
        description="Enable JIT (TorchScript) optimization for CPU devices",
    )
    compile_mode: str = Field(
        default="reduce-overhead",
        description="Compilation mode: 'default', 'reduce-overhead', or 'max-autotune'",
    )

    @property
    def is_valid_compile_mode(self) -> bool:
        """Validate compile mode"""
        return self.compile_mode in ["default", "reduce-overhead", "max-autotune"]


class MinioConfig(BaseModel):
    """MinIO configuration for model downloads"""

    bucket: str = Field(default="")
    object: str = Field(default="")


class GcsConfig(BaseModel):
    """GCS configuration for model downloads"""
    model_config = {"protected_namespaces": ()}

    bucket: str = Field(default="")
    prefix: str = Field(default="")
    model_prefix: str = Field(default="")


class DatabaseConfig(BaseModel):
    """Database connection pool configuration"""

    pool_size: int = Field(default=5, ge=1)
    max_overflow: int = Field(default=10, ge=0)
    pool_pre_ping: bool = Field(default=True)
    pool_recycle: int = Field(default=3600)
    pool_timeout: int = Field(default=10)
    connect_timeout: int = Field(default=5)


class CorsConfig(BaseModel):
    """CORS configuration"""

    allow_origins: List[str] = Field(default=["*"])
    allow_credentials: bool = Field(default=True)
    allow_methods: List[str] = Field(default=["*"])
    allow_headers: List[str] = Field(default=["*"])


class ElasticApmConfig(BaseModel):
    """Elastic APM configuration"""

    enabled: bool = Field(default=False)
    server_url: str = Field(default="")
    service_name: str = Field(default="ocr-ktp-graycopy-classifier")
    environment: str = Field(default="dev")
    secret_token: Optional[str] = Field(default=None)
    verify_server_cert: bool = Field(default=False)


class Config(BaseModel):
    """Main configuration"""

    model: ModelConfig = Field(default_factory=ModelConfig)
    prediction: PredictionConfig = Field(default_factory=PredictionConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    device: DeviceConfig = Field(default_factory=DeviceConfig)
    minio: MinioConfig = Field(default_factory=MinioConfig)
    gcs: GcsConfig = Field(default_factory=GcsConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    cors: CorsConfig = Field(default_factory=CorsConfig)
    elastic_apm: ElasticApmConfig = Field(default_factory=ElasticApmConfig)


def load_config(config_path: Optional[str] = None) -> Config:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config file. If None, uses CONFIG_PATH env var or default.

    Returns:
        Config object with validated configuration

    Raises:
        FileNotFoundError: If config file not found
        yaml.YAMLError: If config file is invalid YAML
        ValueError: If config values are invalid
    """
    if config_path is None:
        config_path = os.getenv("CONFIG_PATH", "./config.yaml")

    config_file = Path(config_path)

    if not config_file.exists():
        # Return default configuration
        return Config()

    with open(config_file, "r", encoding="utf-8") as f:
        config_data = yaml.safe_load(f)

    if config_data is None:
        config_data = {}

    config_data = _substitute_env_vars(config_data)

    return Config(**config_data)


# Global config instance (loaded once)
_config: Optional[Config] = None


def get_config() -> Config:
    """
    Get global configuration instance.

    Returns:
        Config object
    """
    global _config
    if _config is None:
        _config = load_config()
    return _config


# Initialize config instance for easy import
config = get_config()
