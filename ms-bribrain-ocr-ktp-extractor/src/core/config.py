"""Configuration settings loader"""
import os
import re
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional

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


class Settings:
    """Application settings loaded from config.yaml"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize settings by loading config.yaml
        
        Args:
            config_path: Path to the configuration file
        """
        self.config_path = Path(config_path)
        self._config: Dict[str, Any] = {}
        self._load_config()
    
    def _load_config(self) -> None:
        """Load configuration from YAML file"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            self._config = _substitute_env_vars(yaml.safe_load(f))
    
    # Server settings
    @property
    def server_host(self) -> str:
        return self._config.get("server", {}).get("host", "0.0.0.0")
    
    @property
    def server_port(self) -> int:
        return self._config.get("server", {}).get("port", 8001)
    
    @property
    def server_reload(self) -> bool:
        return self._config.get("server", {}).get("reload", False)

    # CORS settings
    @property
    def cors_allow_origins(self) -> List[str]:
        return self._config.get("cors", {}).get("allow_origins", ["*"])

    @property
    def cors_allow_credentials(self) -> bool:
        return self._config.get("cors", {}).get("allow_credentials", True)

    @property
    def cors_allow_methods(self) -> List[str]:
        return self._config.get("cors", {}).get("allow_methods", ["*"])

    @property
    def cors_allow_headers(self) -> List[str]:
        return self._config.get("cors", {}).get("allow_headers", ["*"])

    # OCR settings
    @property
    def ocr_server_config_path(self) -> str:
        return self._config.get("ocr", {}).get("server_config_path", "src/resources/PaddleOCR.yaml")
    
    @property
    def ocr_mobile_config_path(self) -> str:
        return self._config.get("ocr", {}).get("mobile_config_path", "src/resources/PaddleOCR.yaml")
    
    @property
    def ocr_max_size_mb(self) -> int:
        return self._config.get("ocr", {}).get("image", {}).get("max_size_mb", 10)
    
    @property
    def ocr_allowed_types(self) -> List[str]:
        return self._config.get("ocr", {}).get("image", {}).get("allowed_types", ["image/jpeg", "image/png"])
    
    @property
    def ocr_width_threshold_ratio(self) -> float:
        return self._config.get("ocr", {}).get("filter", {}).get("width_threshold_ratio", 0.8)
    
    # Device settings
    @property
    def device_preferred(self) -> str:
        return self._config.get("device", {}).get("preferred", "auto")
    
    @property
    def device_force_cpu(self) -> bool:
        return self._config.get("device", {}).get("force_cpu", False)
    
    # Logging settings
    @property
    def log_level(self) -> str:
        return self._config.get("logging", {}).get("level", "INFO")
    
    @property
    def log_console_enabled(self) -> bool:
        return self._config.get("logging", {}).get("console", {}).get("enabled", True)
    
    @property
    def log_console_level(self) -> str:
        return self._config.get("logging", {}).get("console", {}).get("level", "INFO")
    
    @property
    def log_file_enabled(self) -> bool:
        return self._config.get("logging", {}).get("file", {}).get("enabled", True)
    
    @property
    def log_file_level(self) -> str:
        return self._config.get("logging", {}).get("file", {}).get("level", "DEBUG")
    
    @property
    def log_file_path(self) -> str:
        return self._config.get("logging", {}).get("file", {}).get("path", "logs/app.log")
    
    @property
    def log_file_max_bytes(self) -> int:
        return self._config.get("logging", {}).get("file", {}).get("max_bytes", 10485760)
    
    @property
    def log_file_backup_count(self) -> int:
        return self._config.get("logging", {}).get("file", {}).get("backup_count", 5)
    
    @property
    def log_format(self) -> str:
        return self._config.get("logging", {}).get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    
    @property
    def log_date_format(self) -> str:
        return self._config.get("logging", {}).get("date_format", "%Y-%m-%d %H:%M:%S")

    @property
    def log_to_database(self) -> bool:
        return self._config.get("logging", {}).get("insert_to_database", False)

    # Database settings
    @property
    def db_pool_size(self) -> int:
        return self._config.get("database", {}).get("pool_size", 5)

    @property
    def db_max_overflow(self) -> int:
        return self._config.get("database", {}).get("max_overflow", 10)

    @property
    def db_pool_pre_ping(self) -> bool:
        return self._config.get("database", {}).get("pool_pre_ping", True)

    @property
    def db_pool_recycle(self) -> int:
        return self._config.get("database", {}).get("pool_recycle", 3600)

    @property
    def db_pool_timeout(self) -> int:
        return self._config.get("database", {}).get("pool_timeout", 10)

    @property
    def db_connect_timeout(self) -> int:
        return self._config.get("database", {}).get("connect_timeout", 5)

    # Elastic APM settings
    @property
    def elastic_apm_enabled(self) -> bool:
        return self._config.get("elastic_apm", {}).get("enabled", False)

    @property
    def elastic_apm_server_url(self) -> str:
        return self._config.get("elastic_apm", {}).get("server_url", "")

    @property
    def elastic_apm_service_name(self) -> str:
        return self._config.get("elastic_apm", {}).get("service_name", "ocr-ktp-extractor")

    @property
    def elastic_apm_environment(self) -> str:
        return self._config.get("elastic_apm", {}).get("environment", "dev")

    @property
    def elastic_apm_secret_token(self) -> Optional[str]:
        return self._config.get("elastic_apm", {}).get("secret_token") or None

    @property
    def elastic_apm_verify_server_cert(self) -> bool:
        return self._config.get("elastic_apm", {}).get("verify_server_cert", False)

    # GCS settings
    @property
    def gcs_bucket_name(self) -> str:
        return self._config.get("gcs", {}).get("bucket_name", "")

    @property
    def gcs_model_prefix(self) -> str:
        return self._config.get("gcs", {}).get("model_prefix", "inference")

    @property
    def gcs_mobile_prefix(self) -> str:
        return self._config.get("gcs", {}).get("mobile_prefix", "ocr_models/paddle_models/mobile/")

    @property
    def gcs_mobile_model_path(self) -> str:
        return self._config.get("gcs", {}).get("mobile_model_path", "./src/models/mobile_models/mobile_261125/inference.pdiparams")

    # Server PyTorch .pth flow — detection and recognition fetched separately
    @property
    def gcs_server_det_prefix(self) -> str:
        return self._config.get("gcs", {}).get("server_det_prefix", "ocr_models/detection_models/")

    @property
    def gcs_server_rec_prefix(self) -> str:
        return self._config.get("gcs", {}).get("server_rec_prefix", "ocr_models/recognition_models/")

    @property
    def gcs_server_det_filename_prefix(self) -> str:
        return self._config.get("gcs", {}).get("server_det_filename_prefix", "server_det")

    @property
    def gcs_server_rec_filename_prefix(self) -> str:
        return self._config.get("gcs", {}).get("server_rec_filename_prefix", "server_rec")

    @property
    def gcs_server_det_local_path(self) -> str:
        return self._config.get("gcs", {}).get("server_det_local_path", "./autokernel/workspace/ppocrv5/server_det.pth")

    @property
    def gcs_server_rec_local_path(self) -> str:
        return self._config.get("gcs", {}).get("server_rec_local_path", "./autokernel/workspace/ppocrv5/server_rec.pth")


# Singleton instance
settings = Settings()
