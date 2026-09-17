"""
Configuration management module.

Loads configuration from config.yaml and provides typed access to settings.
Environment variables can override config values using ${VAR_NAME} syntax.
"""

import os
import re
import yaml
from pathlib import Path
from typing import Optional, List, Any
from dataclasses import dataclass, field

import logging

# Configure logging for config module
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load .env file from project root
try:
    from dotenv import load_dotenv
    _project_root = Path(__file__).parent.parent.parent
    _env_path = _project_root / ".env"
    
    # Check if .env is a symlink or file and exists
    if _env_path.exists():
        logger.info(f"Loading .env from {_env_path}")
        load_dotenv(_env_path)
    else:
        logger.warning(f".env file not found at {_env_path}")
except ImportError:
    # python-dotenv not installed, rely on system environment variables
    logger.warning("python-dotenv not installed, skipping .env loading")
    pass


def _substitute_env_vars(value: Any) -> Any:
    """
    Recursively substitute environment variables in config values.
    
    Supports ${VAR_NAME} syntax.
    
    Args:
        value: Configuration value (can be dict, list, or primitive)
        
    Returns:
        Value with environment variables substituted
    """
    if isinstance(value, dict):
        return {k: _substitute_env_vars(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [_substitute_env_vars(item) for item in value]
    elif isinstance(value, str):
        # Pattern to match ${VAR} or ${VAR:-default}
        pattern = r'\$\{([^}]+)\}'

        def replace_var(match):
            expr = match.group(1)
            if ':-' in expr:
                var_name, default = expr.split(':-', 1)
                return os.environ.get(var_name.strip(), default)
            var_name = expr.strip()
            env_value = os.environ.get(var_name)
            if env_value is None:
                logger.warning(f"Environment variable {var_name} not found, substituting empty string")
                return ""
            return env_value

        return re.sub(pattern, replace_var, value)
    else:
        return value


@dataclass
class ServiceConfig:
    """Configuration for an external service."""
    url: str
    timeout: int = 15
    overall_timeout: Optional[int] = None  # Overall timeout for entire pipeline
    api_key: Optional[str] = None


@dataclass
class ServicesConfig:
    """Configuration for all external services."""
    ocr: ServiceConfig
    lamination: ServiceConfig
    recapture: ServiceConfig
    graycopy: ServiceConfig
    temper: ServiceConfig
    classifier: ServiceConfig
    quality: ServiceConfig
    qualitydl: ServiceConfig
    postprocess: ServiceConfig
    orchestrator: ServiceConfig


@dataclass
class AppConfig:
    """Application configuration."""
    name: str
    version: str
    host: str
    port: int
    health_check_timeout: int = 5


@dataclass
class HttpClientConfig:
    """HTTP client connection pool configuration."""
    limit: int = 100
    limit_per_host: int = 30
    ttl_dns_cache: int = 300
    keepalive_timeout: int = 30


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str
    format: str
    file: str
    max_bytes: int
    backup_count: int
    console: bool
    log_to_database: bool
    log_images: bool = False


@dataclass
class DeviceConfig:
    """Device configuration for GPU/CPU selection."""
    prefer_gpu: bool
    fallback_to_cpu: bool

@dataclass
class RunServicesConfig:
    """Configuration for running services."""
    ocr: bool
    lamination: bool
    recapture: bool
    graycopy: bool
    temper: bool
    classifier: bool
    quality: bool
    qualitydl: bool
    postprocess: bool
    orchestrator: bool


@dataclass
class MinioImageConfig:
    """Configuration for MinIO image processing."""
    format: str
    content_type: str
    initial_quality: int
    min_quality: int
    quality_step: int
    scale_step: float
    min_scale: float


@dataclass
class MinioConfig:
    """Configuration for MinIO service."""
    bucket: str
    max_size: int
    pool_size: int
    pool_timeout: int
    ttl_dns_cache: int
    image: MinioImageConfig


@dataclass
class DatabaseConfig:
    """Configuration for database."""
    table_name: str
    schema: str
    result_table_name: str = "bribrain_ocr_ktp_result"
    pool_size: int = 5
    max_overflow: int = 10
    pool_pre_ping: bool = True
    pool_recycle: int = 3600
    pool_timeout: int = 10
    connect_timeout: int = 5


@dataclass
class CorsConfig:
    """CORS configuration."""
    allow_origins: List[str] = field(default_factory=lambda: ["*"])
    allow_credentials: bool = True
    allow_methods: List[str] = field(default_factory=lambda: ["*"])
    allow_headers: List[str] = field(default_factory=lambda: ["*"])


@dataclass
class RateLimitConfig:
    """Rate limiting configuration."""
    enabled: bool = True
    requests_per_minute: int = 60
    requests_per_second: int = 10
    burst_size: int = 20
    cleanup_interval: int = 60
    exclude_paths: List[str] = field(default_factory=lambda: ["/health", "/", "/docs", "/openapi.json", "/redoc"])


@dataclass
class ElasticApmConfig:
    """Elastic APM configuration."""
    enabled: bool = False
    server_url: str = ""
    service_name: str = "ocr-ktp-orchestrator"
    environment: str = "dev"
    secret_token: Optional[str] = None
    verify_server_cert: bool = True


@dataclass
class GcsConfig:
    """Configuration for Google Cloud Storage."""
    bucket_name: str = ""
    prefix: str = ""
    model_prefix: str = "laminate_model"


@dataclass
class Settings:
    """Application settings loaded from config.yaml."""
    services: ServicesConfig
    app: AppConfig
    logging: LoggingConfig
    device: DeviceConfig
    run_services: RunServicesConfig
    minio: MinioConfig
    database: DatabaseConfig
    cors: CorsConfig
    rate_limit: RateLimitConfig
    elastic_apm: ElasticApmConfig
    http_client: HttpClientConfig = field(default_factory=HttpClientConfig)
    gcs: GcsConfig = field(default_factory=GcsConfig)

class ConfigLoader:
    """Singleton configuration loader."""
    
    _instance: Optional['ConfigLoader'] = None
    _settings: Optional[Settings] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def load_config(self, config_path: Optional[Path] = None) -> Settings:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to config.yaml. If None, looks in project root.
            
        Returns:
            Settings object with loaded configuration.
            
        Raises:
            FileNotFoundError: If config file doesn't exist.
            ValueError: If config file is invalid.
        """
        if self._settings is not None:
            return self._settings
        
        # Determine config file path
        if config_path is None:
            # Look for config.yaml in project root
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / "config.yaml"
        else:
            config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {config_path}\n"
                "Please create config.yaml in the project root directory."
            )
        
        # Load YAML
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        # Substitute environment variables in config values
        config_data = _substitute_env_vars(config_data)
        
        # Parse configuration with environment variable overrides
        try:
            services_config = ServicesConfig(
                ocr=ServiceConfig(**config_data['services']['ocr']),
                lamination=ServiceConfig(**config_data['services']['lamination']),
                recapture=ServiceConfig(**config_data['services']['recapture']),
                graycopy=ServiceConfig(**config_data['services']['graycopy']),
                temper=ServiceConfig(**config_data['services']['temper']),
                classifier=ServiceConfig(**config_data['services']['classifier']),
                quality=ServiceConfig(**config_data['services']['quality']),
                qualitydl=ServiceConfig(**config_data['services']['qualitydl']),
                postprocess=ServiceConfig(**config_data['services']['postprocess']),
                orchestrator=ServiceConfig(**config_data['services']['orchestrator']),
            )
            
            app_data = config_data['app']
            app_config = AppConfig(
                name=app_data['name'],
                version=app_data['version'],
                host=app_data['host'],
                port=app_data['port'],
                health_check_timeout=app_data.get('health_check_timeout', 5),
            )
            logging_config = LoggingConfig(**config_data['logging'])
            device_config = DeviceConfig(**config_data['device'])
            run_services_config = RunServicesConfig(**config_data['run_services'])
            
            # Parse MinIO config
            minio_data = config_data.get('minio', {})
            minio_image_config = MinioImageConfig(**minio_data.get('image', {}))
            minio_config = MinioConfig(
                bucket=minio_data.get('bucket', 'ocr-ktp-images'),
                max_size=minio_data.get('max_size', 10485760),
                pool_size=minio_data.get('pool_size', 100),
                pool_timeout=minio_data.get('pool_timeout', 30),
                ttl_dns_cache=minio_data.get('ttl_dns_cache', 300),
                image=minio_image_config
            )
            
            # Parse Database config
            database_data = config_data.get('database', {})
            database_config = DatabaseConfig(
                table_name=database_data.get('table_name', 'ocr_ktp_log'),
                schema=database_data.get('schema', 'public'),
                result_table_name=database_data.get('result_table_name', 'bribrain_ocr_ktp_result'),
                pool_size=database_data.get('pool_size', 5),
                max_overflow=database_data.get('max_overflow', 10),
                pool_pre_ping=database_data.get('pool_pre_ping', True),
                pool_recycle=database_data.get('pool_recycle', 3600),
                pool_timeout=database_data.get('pool_timeout', 10),
                connect_timeout=database_data.get('connect_timeout', 5),
            )
            
            # Parse CORS config
            cors_data = config_data.get('cors', {})
            cors_config = CorsConfig(
                allow_origins=cors_data.get('allow_origins', ['*']),
                allow_credentials=cors_data.get('allow_credentials', True),
                allow_methods=cors_data.get('allow_methods', ['*']),
                allow_headers=cors_data.get('allow_headers', ['*'])
            )
            
            # Parse Rate Limit config
            rate_limit_data = config_data.get('rate_limit', {})
            rate_limit_config = RateLimitConfig(
                enabled=rate_limit_data.get('enabled', True),
                requests_per_minute=rate_limit_data.get('requests_per_minute', 60),
                requests_per_second=rate_limit_data.get('requests_per_second', 10),
                burst_size=rate_limit_data.get('burst_size', 20),
                cleanup_interval=rate_limit_data.get('cleanup_interval', 60),
                exclude_paths=rate_limit_data.get('exclude_paths', ["/health", "/", "/docs", "/openapi.json", "/redoc"])
            )

            # Parse Elastic APM config
            elastic_apm_data = config_data.get('elastic_apm', {})
            elastic_apm_config = ElasticApmConfig(
                enabled=elastic_apm_data.get('enabled', False),
                server_url=elastic_apm_data.get('server_url', ""),
                service_name=elastic_apm_data.get('service_name', "ocr-ktp-orchestrator"),
                environment=elastic_apm_data.get('environment', "dev"),
                secret_token=elastic_apm_data.get('secret_token'),
                verify_server_cert=elastic_apm_data.get('verify_server_cert', True)
            )

            # Parse GCS config
            gcs_data = config_data.get('gcs', {})
            gcs_config = GcsConfig(
                bucket_name=gcs_data.get('bucket_name', ""),
                prefix=gcs_data.get('prefix', ""),
                model_prefix=gcs_data.get('model_prefix', "laminate_model"),
            )

            # Parse HTTP Client config
            http_client_data = config_data.get('http_client', {})
            http_client_config = HttpClientConfig(
                limit=http_client_data.get('limit', 100),
                limit_per_host=http_client_data.get('limit_per_host', 30),
                ttl_dns_cache=http_client_data.get('ttl_dns_cache', 300),
                keepalive_timeout=http_client_data.get('keepalive_timeout', 30),
            )

            self._settings = Settings(
                services=services_config,
                app=app_config,
                logging=logging_config,
                device=device_config,
                run_services=run_services_config,
                minio=minio_config,
                database=database_config,
                cors=cors_config,
                rate_limit=rate_limit_config,
                elastic_apm=elastic_apm_config,
                http_client=http_client_config,
                gcs=gcs_config,
            )
            
            return self._settings
            
        except (KeyError, TypeError) as e:
            raise ValueError(f"Invalid configuration file: {e}")
    
    def get_settings(self) -> Settings:
        """
        Get loaded settings.
        
        Returns:
            Settings object.
            
        Raises:
            RuntimeError: If settings haven't been loaded yet.
        """
        if self._settings is None:
            # Auto-load on first access
            return self.load_config()
        return self._settings


# Global function for easy access
def get_settings() -> Settings:
    """
    Get application settings.
    
    Returns:
        Settings object with configuration.
    """
    loader = ConfigLoader()
    return loader.get_settings()


class Config:
    """
    Config wrapper that provides dot-notation access to settings.
    Supports get('minio.bucket') style access for backward compatibility.
    """
    
    def __init__(self):
        self._settings = None
    
    @property
    def settings(self) -> Settings:
        """Lazy load settings on first access."""
        if self._settings is None:
            self._settings = get_settings()
        return self._settings
    
    def get(self, key: str, default=None):
        """
        Get configuration value using dot notation.
        
        Args:
            key: Dot-separated path like 'minio.bucket' or 'minio.image.format'
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        try:
            parts = key.split('.')
            value = self.settings
            for part in parts:
                if hasattr(value, part):
                    value = getattr(value, part)
                else:
                    return default
            return value
        except (AttributeError, KeyError):
            return default


# Global config instance for dot-notation access
config = Config()
