"""
Configuration management for OCR Quality Service.
Loads and validates configuration from config.yaml.
"""
import os
import re
import yaml
from pathlib import Path
from typing import Optional, Any, List
from dotenv import load_dotenv
from pydantic import BaseModel, Field
# from pydantic_settings import BaseSettings

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


class AppConfig(BaseModel):
    """Application configuration."""
    name: str = "OCR Quality Service"
    version: str = "0.1.0"
    debug: bool = False


class ServerConfig(BaseModel):
    """Server configuration."""
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    max_file_size_mb: int = Field(default=10, ge=1, description="Maximum uploaded image size in MB")


class DeviceConfig(BaseModel):
    """Device configuration for GPU/CPU selection."""
    prefer_gpu: bool = True
    fallback_to_cpu: bool = True


class BlurConfig(BaseModel):
    """Blur detection configuration."""
    enabled: bool = True
    threshold: float = 100.0


class ConfidenceConfig(BaseModel):
    """Confidence threshold configuration."""
    enabled: bool = True
    threshold_median: float = 0.8


class GlareConfig(BaseModel):
    """Glare detection configuration."""
    enabled: bool = True
    min_area: int = 100
    padding_size: int = 20
    kernel_size: int = 5  # Morphological kernel size
    min_text_confidence: float = 0.6  # Min OCR confidence for text analysis
    affected_percentage_threshold: float = 5.0  # Min % affected to flag glare


class RotationConfig(BaseModel):
    """Rotation detection configuration."""
    enabled: bool = True
    min_face_proportion: float = 0.12
    face_detection_confidence: float = 0.5  # InsightFace RetinaFace confidence threshold
    det_size: List[int] = Field(default_factory=lambda: [640, 640])  # InsightFace detector input size
    verification_box_width_ratio: float = 0.3  # Box width as ratio of image width
    verification_box_height_ratio: float = 0.6  # Box height as ratio of image height
    verification_box_margin_ratio: float = 0.02  # Right margin as ratio
    verification_box_vertical_center: float = 0.44  # Vertical center position


class QualityConfig(BaseModel):
    """Image quality thresholds configuration."""
    blur: BlurConfig = Field(default_factory=BlurConfig)
    confidence: ConfidenceConfig = Field(default_factory=ConfidenceConfig)
    glare: GlareConfig = Field(default_factory=GlareConfig)
    rotation: RotationConfig = Field(default_factory=RotationConfig)


class LogFileConfig(BaseModel):
    """Log file configuration."""
    enabled: bool = True
    path: str = "logs/ocr_quality.log"
    max_bytes: int = 10485760  # 10MB
    backup_count: int = 5


class LogConsoleConfig(BaseModel):
    """Log console configuration."""
    enabled: bool = True


class LoggingConfig(BaseModel):
    """Logging configuration."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(request_id)s - %(message)s"
    file: LogFileConfig = Field(default_factory=LogFileConfig)
    console: LogConsoleConfig = Field(default_factory=LogConsoleConfig)
    log_to_database: bool = False


class DatabaseConfig(BaseModel):
    """Database configuration."""
    table_name: str = "bribrain_ocr_ktp_iqa_rulebased"
    db_schema: str = Field(default="public", alias="schema")
    pool_size: int = 5
    max_overflow: int = 10
    pool_pre_ping: bool = True
    pool_recycle: int = 3600
    pool_timeout: int = 10
    connect_timeout: int = 5

    model_config = {"populate_by_name": True}  # Allow both 'schema' and 'db_schema'


class CorsConfig(BaseModel):
    """CORS configuration."""
    allow_origins: List[str] = ["*"]
    allow_credentials: bool = True
    allow_methods: List[str] = ["*"]
    allow_headers: List[str] = ["*"]


class ElasticApmConfig(BaseModel):
    """Elastic APM configuration."""
    enabled: bool = False
    server_url: str = ""
    service_name: str = "ocr-ktp-iqa-rulebased"
    environment: str = "dev"
    secret_token: Optional[str] = None
    verify_server_cert: bool = False


class Config(BaseModel):
    """Main configuration class."""
    app: AppConfig = Field(default_factory=AppConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    device: DeviceConfig = Field(default_factory=DeviceConfig)
    quality: QualityConfig = Field(default_factory=QualityConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    cors: CorsConfig = Field(default_factory=CorsConfig)
    elastic_apm: ElasticApmConfig = Field(default_factory=ElasticApmConfig)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key: Dot-separated path like 'database.table_name'
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        try:
            parts = key.split('.')
            value = self
            for part in parts:
                if hasattr(value, part):
                    value = getattr(value, part)
                else:
                    return default
            return value
        except (AttributeError, KeyError):
            return default


# Global configuration instance
_config: Optional[Config] = None


def load_config(config_path: str = "config.yaml") -> Config:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Config: Validated configuration object
    """
    # Get the project root directory
    project_root = Path(__file__).parent.parent.parent
    config_file = project_root / config_path

    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_file}")

    with open(config_file, 'r') as f:
        config_data = yaml.safe_load(f) or {}

    config_data = _substitute_env_vars(config_data)

    return Config(**config_data)


def get_config() -> Config:
    """
    Get the global configuration instance.
    Loads configuration on first call.
    
    Returns:
        Config: Global configuration object
    """
    global _config
    if _config is None:
        _config = load_config()
    return _config


def reload_config(config_path: str = "config.yaml") -> Config:
    """
    Reload configuration from file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Config: Reloaded configuration object
    """
    global _config
    _config = load_config(config_path)
    return _config
