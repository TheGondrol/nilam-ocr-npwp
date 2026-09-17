"""Configuration management using YAML."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import yaml


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
    service_name: str = "ocr-ktp-iqa-dl"
    environment: str = "dev"
    secret_token: Optional[str] = None
    verify_server_cert: bool = False


class Config:
    """Singleton configuration loader from YAML file."""

    _instance: Config | None = None
    _config: dict[str, Any] = {}

    def __new__(cls) -> Config:
        """Ensure only one instance of Config exists."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_config()
        return cls._instance

    def _load_config(self) -> None:
        """Load configuration from YAML file."""
        config_path = Path(__file__).parent.parent.parent / "config.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        self._config = _substitute_env_vars(raw)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.

        Args:
            key: Configuration key in dot notation (e.g., 'model.path')
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        keys = key.split(".")
        value = self._config

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default

        return value

    @property
    def model_path(self) -> str:
        """Get model file path."""
        return self.get("model.path", "./src/models/quality_model.pth")

    @property
    def num_classes(self) -> int:
        """Get number of model output classes."""
        return self.get("model.num_classes", 2)

    @property
    def img_height(self) -> int:
        """Get image height for input images."""
        return self.get("model.img_height", 64)

    @property
    def img_width(self) -> int:
        """Get image width for input images."""
        return self.get("model.img_width", 320)

    @property
    def normalize_mean(self) -> list[float]:
        """Get normalization mean values."""
        return self.get("model.normalize_mean", [0.485, 0.456, 0.406])

    @property
    def normalize_std(self) -> list[float]:
        """Get normalization std values."""
        return self.get("model.normalize_std", [0.229, 0.224, 0.225])

    @property
    def use_compile(self) -> bool:
        """Whether to use torch.compile for optimization."""
        return self.get("model.use_compile", True)

    @property
    def compile_mode(self) -> str:
        """Compilation mode for torch.compile."""
        return self.get("model.compile_mode", "reduce-overhead")

    @property
    def bad_crop_threshold(self) -> int:
        """Get bad crop threshold for overall image quality."""
        return self.get("prediction.bad_crop_threshold", 4)

    @property
    def max_image_size_mb(self) -> int:
        """Get maximum allowed image upload size in megabytes."""
        return self.get("image.max_size_mb", 10)

    @property
    def min_width_ratio(self) -> float:
        """Get minimum width to height ratio for crop filtering."""
        return self.get("crop_filter.min_width_ratio", 1.0)

    @property
    def min_width(self) -> int:
        """Get minimum width in pixels for crop filtering."""
        return self.get("crop_filter.min_width", 0)

    @property
    def server_host(self) -> str:
        """Get server host."""
        return self.get("server.host", "0.0.0.0")

    @property
    def server_port(self) -> int:
        """Get server port."""
        return self.get("server.port", 8100)

    @property
    def thread_pool_workers(self) -> int:
        """Get thread pool workers count."""
        return self.get("server.thread_pool_workers", 4)

    @property
    def debug_mode(self) -> bool:
        """Whether debug mode is enabled."""
        return self.get("debug_mode", False)

    @property
    def log_to_database(self) -> bool:
        """Whether to log requests to database."""
        return self.get("logging.log_to_database", True)

    # Database settings
    @property
    def db_pool_size(self) -> int:
        return self.get("database.pool_size", 5)

    @property
    def db_max_overflow(self) -> int:
        return self.get("database.max_overflow", 10)

    @property
    def db_pool_pre_ping(self) -> bool:
        return self.get("database.pool_pre_ping", True)

    @property
    def db_pool_recycle(self) -> int:
        return self.get("database.pool_recycle", 3600)

    @property
    def db_pool_timeout(self) -> int:
        return self.get("database.pool_timeout", 10)

    @property
    def db_connect_timeout(self) -> int:
        return self.get("database.connect_timeout", 5)

    @property
    def elastic_apm(self) -> ElasticApmConfig:
        """Typed Elastic APM configuration."""
        data = self.get("elastic_apm") or {}
        return ElasticApmConfig(
            enabled=bool(data.get("enabled", False)),
            server_url=data.get("server_url", "") or "",
            service_name=data.get("service_name") or "ocr-ktp-iqa-dl",
            environment=data.get("environment") or "dev",
            secret_token=data.get("secret_token") or None,
            verify_server_cert=bool(data.get("verify_server_cert", False)),
        )

    # GCS settings
    @property
    def gcs_bucket_name(self) -> str:
        return self.get("gcs.bucket_name", "")

    @property
    def gcs_prefix(self) -> str:
        return self.get("gcs.prefix", "")

    @property
    def gcs_model_prefix(self) -> str:
        return self.get("gcs.model_prefix", "")


# Singleton instance
config = Config()
