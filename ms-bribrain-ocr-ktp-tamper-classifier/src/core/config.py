"""
Core configuration management module.

This module provides centralized configuration loading from YAML files with
Pydantic validation for type safety. Supports environment variable overrides
and singleton pattern for global configuration access.

Usage:
    from src.core.config import get_config

    config = get_config()
    app_name = config.app.name
    log_level = config.logging.level
"""

import os
from pathlib import Path
from typing import Any, List, Optional

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()


class AppConfig(BaseModel):
    """Application-level configuration."""

    name: str = Field(default="FastAPI Application", description="Application name")
    version: str = Field(default="1.0.0", description="Application version")
    environment: str = Field(
        default="development", description="Environment (development/staging/production)"
    )
    description: str = Field(
        default="FastAPI application with async support", description="Application description"
    )
    debug: bool = Field(default=False, description="Enable debug mode")


class APIConfig(BaseModel):
    """API server configuration."""

    host: str = Field(default="0.0.0.0", description="API host")
    port: int = Field(default=8000, description="API port", ge=1024, le=65535)
    workers: int = Field(default=1, description="Number of worker processes", ge=1)
    reload: bool = Field(default=False, description="Enable auto-reload on code changes")
    api_prefix: str = Field(default="/v1", description="API route prefix")
    cors_origins: List[str] = Field(default=["*"], description="CORS allowed origins")
    cors_credentials: bool = Field(default=True, description="Allow credentials in CORS")
    cors_methods: List[str] = Field(default=["*"], description="Allowed HTTP methods")
    cors_headers: List[str] = Field(default=["*"], description="Allowed HTTP headers")


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: str = Field(default="INFO", description="Log level (DEBUG/INFO/WARNING/ERROR/CRITICAL)")
    format: str = Field(default="json", description="Log format (json/text)")
    log_dir: str = Field(default="logs", description="Directory for log files")
    app_log_file: str = Field(default="app.log", description="Application log file name")
    error_log_file: str = Field(default="error.log", description="Error log file name")
    max_file_size_mb: int = Field(default=10, description="Max log file size in MB", ge=1)
    backup_count: int = Field(default=5, description="Number of backup log files to keep", ge=1)
    include_request_id: bool = Field(default=True, description="Include request ID in logs")
    database_url: str = Field(default="", description="Database connection URL")
    log_to_database: bool = Field(default=False, description="Log to database")

    @field_validator("level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level is valid."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        v_upper = v.upper()
        if v_upper not in valid_levels:
            raise ValueError(f"Invalid log level: {v}. Must be one of {valid_levels}")
        return v_upper


class DeviceConfig(BaseModel):
    """Device (GPU/CPU) configuration."""

    preferred: str = Field(default="auto", description="Preferred device (auto/cuda/mps/cpu)")
    cuda_device_id: int = Field(default=0, description="CUDA device ID to use", ge=0)
    allow_cpu_fallback: bool = Field(
        default=True, description="Allow CPU fallback if GPU unavailable"
    )
    memory_fraction: float = Field(
        default=0.9, description="Fraction of GPU memory to use", ge=0.1, le=1.0
    )
    force_cpu: bool = Field(default=False, description="Force CPU usage even if GPU is available")

    @field_validator("preferred")
    @classmethod
    def validate_device(cls, v: str) -> str:
        """Validate device type."""
        valid_devices = ["auto", "cuda", "mps", "cpu"]
        v_lower = v.lower()
        if v_lower not in valid_devices:
            raise ValueError(f"Invalid device: {v}. Must be one of {valid_devices}")
        return v_lower


class ModelConfig(BaseModel):
    """Model configuration."""

    path: str = Field(
        default="./models/font_finetuned_model_v3_advanced", description="Path to ML model weights"
    )
    file_path: str = Field(
        default="", description="Path to downloaded model file (e.g., model.safetensors)"
    )
    image_size: int = Field(default=320, description="Image size for preprocessing", ge=1)


class PredictionConfig(BaseModel):
    """Prediction configuration."""

    threshold: float = Field(
        default=0.5, description="Threshold for classification", ge=0.0, le=1.0
    )
    batch_size_limit: int = Field(
        default=10, description="Maximum number of files per batch request", ge=1
    )


class ImageConfig(BaseModel):
    """Image upload configuration."""

    max_size_mb: int = Field(default=10, ge=1, description="Maximum uploaded image size in MB")


class ServerConfig(BaseModel):
    """Server configuration."""

    host: str = Field(default="0.0.0.0", description="Host address to bind the server")
    port: int = Field(default=8030, description="Port number for the service", ge=1024, le=65535)


class MinioConfig(BaseModel):
    """MinIO object storage configuration."""

    bucket: str = Field(default="", description="MinIO bucket name")
    object: str = Field(default="", description="MinIO object path")


class GcsConfig(BaseModel):
    """GCS configuration for model downloads."""

    bucket_name: str = Field(default="")
    prefix: str = Field(default="")
    model_prefix: str = Field(default="")


class DatabaseConfig(BaseModel):
    """Database connection pool configuration."""

    pool_size: int = Field(default=5, ge=1)
    max_overflow: int = Field(default=10, ge=0)
    pool_pre_ping: bool = Field(default=True)
    pool_recycle: int = Field(default=3600)
    pool_timeout: int = Field(default=10)
    connect_timeout: int = Field(default=5)


class CorsConfig(BaseModel):
    """CORS configuration."""

    allow_origins: List[str] = Field(default=["*"], description="Allowed origins")
    allow_credentials: bool = Field(default=True, description="Allow credentials")
    allow_methods: List[str] = Field(default=["*"], description="Allowed HTTP methods")
    allow_headers: List[str] = Field(default=["*"], description="Allowed HTTP headers")


class ElasticApmConfig(BaseModel):
    """Elastic APM configuration."""

    enabled: bool = Field(default=False)
    server_url: str = Field(default="")
    service_name: str = Field(default="ocr-ktp-tamper-classifier")
    environment: str = Field(default="dev")
    secret_token: Optional[str] = Field(default=None)
    verify_server_cert: bool = Field(default=False)


class ServiceConfig(BaseModel):
    """Service-specific configuration (customize per project)."""

    # Example configurations - customize based on your services
    model_path: Optional[str] = Field(default=None, description="Path to ML model weights")
    batch_size: int = Field(default=1, description="Processing batch size", ge=1)
    timeout_seconds: int = Field(default=30, description="Service timeout in seconds", ge=1)
    max_retries: int = Field(default=3, description="Maximum retry attempts", ge=0)
    enable_caching: bool = Field(default=True, description="Enable result caching")

    # Add your project-specific configurations here
    # Example: ocr_confidence_threshold: float = Field(default=0.8, ge=0.0, le=1.0)


class Config(BaseSettings):
    """Main configuration class combining all config sections."""

    model_config = SettingsConfigDict(
        extra='allow',
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        case_sensitive=False
    )

    model: ModelConfig = Field(default_factory=ModelConfig)
    prediction: PredictionConfig = Field(default_factory=PredictionConfig)
    image: ImageConfig = Field(default_factory=ImageConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    device: DeviceConfig = Field(default_factory=DeviceConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    app: AppConfig = Field(default_factory=AppConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    service: ServiceConfig = Field(default_factory=ServiceConfig)
    minio: MinioConfig = Field(default_factory=MinioConfig)
    gcs: GcsConfig = Field(default_factory=GcsConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    cors: CorsConfig = Field(default_factory=CorsConfig)
    elastic_apm: ElasticApmConfig = Field(default_factory=ElasticApmConfig)


# Singleton instance
_config_instance: Optional[Config] = None


def load_config(config_path: str = "config.yaml") -> Config:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config YAML file (default: config.yaml)

    Returns:
        Config: Loaded and validated configuration

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If YAML is invalid
        ValueError: If configuration validation fails
    """
    config_file = Path(config_path)

    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    # Load YAML
    with open(config_file, "r", encoding="utf-8") as f:
        yaml_data = yaml.safe_load(f)

    if yaml_data is None:
        yaml_data = {}

    # Perform environment variable substitution
    yaml_data = _substitute_env_vars(yaml_data)

    # Create Config object with Pydantic validation
    config = Config(**yaml_data)

    return config


def _substitute_env_vars(data: Any) -> Any:
    """
    Recursively substitute environment variables in config data.

    Supports ${VAR_NAME} and ${VAR_NAME:-default_value} syntax.

    Args:
        data: Configuration data (dict, list, or primitive)

    Returns:
        Data with environment variables substituted
    """
    if isinstance(data, dict):
        return {k: _substitute_env_vars(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [_substitute_env_vars(item) for item in data]
    elif isinstance(data, str):
        # Simple ${VAR} substitution
        if data.startswith("${") and data.endswith("}"):
            var_expr = data[2:-1]

            # Check for default value syntax: ${VAR:-default}
            if ":-" in var_expr:
                var_name, default_value = var_expr.split(":-", 1)
                return os.getenv(var_name.strip(), default_value.strip())
            else:
                var_name = var_expr.strip()
                value = os.getenv(var_name)
                if value is None:
                    raise ValueError(
                        f"Environment variable {var_name} not found and no default provided"
                    )
                return value
        return data
    else:
        return data


def get_config(config_path: str = "config.yaml", force_reload: bool = False) -> Config:
    """
    Get configuration singleton instance.

    Args:
        config_path: Path to config YAML file
        force_reload: Force reload configuration from file

    Returns:
        Config: Configuration singleton
    """
    global _config_instance

    if _config_instance is None or force_reload:
        _config_instance = load_config(config_path)

    return _config_instance


def reload_config(config_path: str = "config.yaml") -> Config:
    """
    Reload configuration from file.

    Args:
        config_path: Path to config YAML file

    Returns:
        Config: Reloaded configuration
    """
    return get_config(config_path, force_reload=True)


# Initialize default config instance
config = get_config()
