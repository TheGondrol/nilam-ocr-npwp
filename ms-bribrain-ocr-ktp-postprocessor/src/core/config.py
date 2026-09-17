"""Configuration management module.

This module handles loading and accessing configuration from config.yaml.
It provides a singleton pattern for configuration access throughout the application.
"""

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
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
    service_name: str = "ocr-ktp-postprocessor"
    environment: str = "dev"
    secret_token: Optional[str] = None
    verify_server_cert: bool = False


class Config:
    """Configuration singleton class."""

    _instance = None
    _config: Dict[str, Any] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._load_config()
        return cls._instance

    def _load_config(self) -> None:
        """Load configuration from config.yaml file."""
        # Get the project root directory (parent of src)
        project_root = Path(__file__).parent.parent.parent
        config_path = project_root / "config.yaml"

        if not config_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found at {config_path}. "
                "Please ensure config.yaml exists in the project root."
            )

        with open(config_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        self._config = _substitute_env_vars(raw)

        # Allow environment variable overrides
        self._apply_env_overrides()

    def _apply_env_overrides(self) -> None:
        """Apply environment variable overrides to configuration."""
        # Override thresholds if env vars exist
        if "THRESHOLD_PARTIAL" in os.environ:
            self._config["thresholds"]["partial"] = float(os.environ["THRESHOLD_PARTIAL"])
        if "THRESHOLD_RATIO" in os.environ:
            self._config["thresholds"]["ratio"] = float(os.environ["THRESHOLD_RATIO"])
        if "THRESHOLD_CONFIDENCE" in os.environ:
            self._config["thresholds"]["confidence"] = float(os.environ["THRESHOLD_CONFIDENCE"])

        # Override API settings if env vars exist
        if "API_HOST" in os.environ:
            self._config["api"]["host"] = os.environ["API_HOST"]
        if "API_PORT" in os.environ:
            self._config["api"]["port"] = int(os.environ["API_PORT"])

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key.

        Args:
            key: Configuration key (supports dot notation, e.g., 'thresholds.ratio')
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        keys = key.split(".")
        value: Any = self._config

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default

        return value

    def get_thresholds(self) -> Dict[str, float]:
        """Get all threshold values.

        Returns:
            Dictionary of threshold values
        """
        return self._config.get("thresholds", {})

    def get_regex(self) -> Dict[str, str]:
        """Get all regex patterns.

        Returns:
            Dictionary of regex patterns
        """
        return self._config.get("regex", {})

    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration.

        Returns:
            Dictionary of logging configuration
        """
        return self._config.get("logging", {})

    def get_api_config(self) -> Dict[str, Any]:
        """Get API configuration.

        Returns:
            Dictionary of API configuration
        """
        return self._config.get("api", {})

    def get_database_config(self) -> Dict[str, Any]:
        """Get database configuration.

        Returns:
            Dictionary of database configuration
        """
        return self._config.get("database", {})

    def get_cors_config(self) -> Dict[str, Any]:
        """Get CORS configuration.

        Returns:
            Dictionary of CORS configuration with defaults
        """
        cors = self._config.get("cors", {})
        return {
            "allow_origins": cors.get("allow_origins", ["*"]),
            "allow_credentials": cors.get("allow_credentials", True),
            "allow_methods": cors.get("allow_methods", ["*"]),
            "allow_headers": cors.get("allow_headers", ["*"]),
        }

    def get_constants(self) -> Dict[str, Any]:
        """Get KTP field constants.

        Returns:
            Dictionary of constants for field identification
        """
        return self._config.get("constants", {})

    def get_mappings(self) -> Dict[str, Any]:
        """Get field value mappings.

        Returns:
            Dictionary of mappings for field normalization
        """
        return self._config.get("mappings", {})

    def get_bulan_dict(self) -> Dict[str, str]:
        """Get month mapping dictionary.

        Returns:
            Dictionary mapping month numbers to abbreviations
        """
        return self._config.get("bulan_dict", {})

    def get_messages(self) -> Dict[str, str]:
        """Get error messages.

        Returns:
            Dictionary of error messages
        """
        return self._config.get("messages", {})

    def get_character_mappings(self) -> Dict[str, Any]:
        """Get character correction mappings.

        Returns:
            Dictionary of character mappings for OCR correction
        """
        return self._config.get("character_mappings", {})

    @property
    def elastic_apm(self) -> ElasticApmConfig:
        """Typed Elastic APM configuration."""
        data = self._config.get("elastic_apm", {}) or {}
        return ElasticApmConfig(
            enabled=bool(data.get("enabled", False)),
            server_url=data.get("server_url", "") or "",
            service_name=data.get("service_name") or "ocr-ktp-postprocessor",
            environment=data.get("environment") or "dev",
            secret_token=data.get("secret_token") or None,
            verify_server_cert=bool(data.get("verify_server_cert", False)),
        )

    @property
    def all(self) -> Dict[str, Any]:
        """Get all configuration.

        Returns:
            Complete configuration dictionary
        """
        return self._config.copy()


def get_config() -> Config:
    """Get the configuration singleton instance.

    Returns:
        Config instance
    """
    return Config()
