"""
Environment configuration module for HMAS services.
Handles environment variables, configuration loading, and environment-specific settings.
"""

import os
from enum import Enum
from typing import Any, Dict, Optional
from pathlib import Path
import json
import yaml
from pydantic import BaseModel, Field
from dotenv import load_dotenv

class EnvironmentType(str, Enum):
    """Environment types supported by HMAS"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"

class DatabaseConfig(BaseModel):
    """Database connection configuration"""
    host: str = Field(..., description="Database host")
    port: int = Field(..., description="Database port")
    username: str = Field(..., description="Database username")
    password: str = Field(..., description="Database password")
    database: str = Field(..., description="Database name")
    max_connections: int = Field(10, description="Maximum number of connections")
    timeout: float = Field(30.0, description="Connection timeout in seconds")
    ssl_enabled: bool = Field(False, description="Enable SSL for database connection")

class RedisConfig(BaseModel):
    """Redis connection configuration"""
    host: str = Field(..., description="Redis host")
    port: int = Field(..., description="Redis port")
    password: Optional[str] = Field(None, description="Redis password")
    db: int = Field(0, description="Redis database number")
    ssl_enabled: bool = Field(False, description="Enable SSL for Redis connection")

class SecurityConfig(BaseModel):
    """Security configuration"""
    secret_key: str = Field(..., description="Secret key for encryption")
    jwt_secret: str = Field(..., description="JWT secret key")
    token_expiry: int = Field(3600, description="Token expiry in seconds")
    allowed_origins: list[str] = Field(default_factory=list, description="CORS allowed origins")
    ssl_enabled: bool = Field(False, description="Enable SSL/TLS")
    ssl_cert_path: Optional[str] = Field(None, description="SSL certificate path")
    ssl_key_path: Optional[str] = Field(None, description="SSL private key path")

class LoggingConfig(BaseModel):
    """Logging configuration"""
    level: str = Field("INFO", description="Logging level")
    format: str = Field(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log format"
    )
    file_path: Optional[str] = Field(None, description="Log file path")
    rotate_size: int = Field(10485760, description="Log rotation size in bytes")
    rotate_count: int = Field(5, description="Number of log files to keep")
    json_format: bool = Field(False, description="Use JSON format for logs")

class MonitoringConfig(BaseModel):
    """Monitoring and metrics configuration"""
    enabled: bool = Field(True, description="Enable monitoring")
    prometheus_enabled: bool = Field(False, description="Enable Prometheus metrics")
    metrics_port: int = Field(9090, description="Metrics port")
    health_check_interval: int = Field(30, description="Health check interval in seconds")
    tracing_enabled: bool = Field(False, description="Enable distributed tracing")
    tracing_sample_rate: float = Field(0.1, description="Tracing sample rate")

class ResourceLimits(BaseModel):
    """Resource limits configuration"""
    cpu_limit: str = Field("1", description="CPU limit")
    memory_limit: str = Field("1Gi", description="Memory limit")
    storage_limit: str = Field("10Gi", description="Storage limit")
    max_workers: int = Field(4, description="Maximum number of workers")
    request_timeout: int = Field(30, description="Request timeout in seconds")

class EnvironmentConfig(BaseModel):
    """Main environment configuration"""
    env_type: EnvironmentType = Field(..., description="Environment type")
    debug: bool = Field(False, description="Debug mode")
    testing: bool = Field(False, description="Testing mode")
    
    # Service configuration
    service_name: str = Field(..., description="Service name")
    host: str = Field("0.0.0.0", description="Service host")
    port: int = Field(..., description="Service port")
    base_url: str = Field(..., description="Service base URL")
    
    # Component configurations
    database: DatabaseConfig = Field(..., description="Database configuration")
    redis: RedisConfig = Field(..., description="Redis configuration")
    security: SecurityConfig = Field(..., description="Security configuration")
    logging: LoggingConfig = Field(..., description="Logging configuration")
    monitoring: MonitoringConfig = Field(..., description="Monitoring configuration")
    resource_limits: ResourceLimits = Field(..., description="Resource limits")
    
    # Additional settings
    temp_dir: str = Field("/tmp", description="Temporary directory")
    data_dir: str = Field("/data", description="Data directory")
    config_dir: str = Field("/config", description="Configuration directory")
    
    class Config:
        """Pydantic model configuration"""
        use_enum_values = True

class EnvironmentManager:
    """
    Environment manager for handling configuration loading and environment setup.
    
    Features:
    - Environment variable loading
    - Configuration file loading (JSON, YAML)
    - Environment-specific settings
    - Secret management
    - Configuration validation
    """
    
    def __init__(
        self,
        service_name: str,
        env_type: Optional[EnvironmentType] = None,
        config_path: Optional[str] = None
    ):
        self.service_name = service_name
        self.env_type = env_type or self._get_env_type()
        self.config_path = config_path
        self.config: Optional[EnvironmentConfig] = None
        
        # Load environment variables
        self._load_env_vars()
        
        # Load configuration
        self._load_config()
    
    def _get_env_type(self) -> EnvironmentType:
        """Get environment type from environment variable"""
        env = os.getenv("HMAS_ENV", "development").lower()
        return EnvironmentType(env)
    
    def _load_env_vars(self) -> None:
        """Load environment variables from .env file"""
        env_file = os.getenv("HMAS_ENV_FILE", ".env")
        if os.path.exists(env_file):
            load_dotenv(env_file)
    
    def _load_config(self) -> None:
        """Load configuration from file or environment variables"""
        config_data = self._load_config_file() or self._load_config_from_env()
        
        # Add service-specific configuration
        config_data.update({
            "service_name": self.service_name,
            "env_type": self.env_type
        })
        
        # Create configuration object
        self.config = EnvironmentConfig(**config_data)
    
    def _load_config_file(self) -> Optional[Dict[str, Any]]:
        """Load configuration from file"""
        if not self.config_path:
            return None
            
        config_file = Path(self.config_path)
        if not config_file.exists():
            return None
            
        with config_file.open() as f:
            if config_file.suffix == ".json":
                return json.load(f)
            elif config_file.suffix in (".yaml", ".yml"):
                return yaml.safe_load(f)
        
        return None
    
    def _load_config_from_env(self) -> Dict[str, Any]:
        """Load configuration from environment variables"""
        config_data = {
            "debug": os.getenv("HMAS_DEBUG", "false").lower() == "true",
            "testing": os.getenv("HMAS_TESTING", "false").lower() == "true",
            "host": os.getenv("HMAS_HOST", "0.0.0.0"),
            "port": int(os.getenv("HMAS_PORT", "8000")),
            "base_url": os.getenv("HMAS_BASE_URL", f"http://localhost:8000"),
            
            # Database configuration
            "database": {
                "host": os.getenv("HMAS_DB_HOST", "localhost"),
                "port": int(os.getenv("HMAS_DB_PORT", "27017")),
                "username": os.getenv("HMAS_DB_USERNAME", "admin"),
                "password": os.getenv("HMAS_DB_PASSWORD", ""),
                "database": os.getenv("HMAS_DB_NAME", "hmas"),
                "max_connections": int(os.getenv("HMAS_DB_MAX_CONNECTIONS", "10")),
                "timeout": float(os.getenv("HMAS_DB_TIMEOUT", "30.0")),
                "ssl_enabled": os.getenv("HMAS_DB_SSL", "false").lower() == "true"
            },
            
            # Redis configuration
            "redis": {
                "host": os.getenv("HMAS_REDIS_HOST", "localhost"),
                "port": int(os.getenv("HMAS_REDIS_PORT", "6379")),
                "password": os.getenv("HMAS_REDIS_PASSWORD", ""),
                "db": int(os.getenv("HMAS_REDIS_DB", "0")),
                "ssl_enabled": os.getenv("HMAS_REDIS_SSL", "false").lower() == "true"
            },
            
            # Security configuration
            "security": {
                "secret_key": os.getenv("HMAS_SECRET_KEY", ""),
                "jwt_secret": os.getenv("HMAS_JWT_SECRET", ""),
                "token_expiry": int(os.getenv("HMAS_TOKEN_EXPIRY", "3600")),
                "allowed_origins": os.getenv("HMAS_ALLOWED_ORIGINS", "").split(","),
                "ssl_enabled": os.getenv("HMAS_SSL_ENABLED", "false").lower() == "true",
                "ssl_cert_path": os.getenv("HMAS_SSL_CERT_PATH", ""),
                "ssl_key_path": os.getenv("HMAS_SSL_KEY_PATH", "")
            },
            
            # Logging configuration
            "logging": {
                "level": os.getenv("HMAS_LOG_LEVEL", "INFO"),
                "format": os.getenv("HMAS_LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"),
                "file_path": os.getenv("HMAS_LOG_FILE", ""),
                "rotate_size": int(os.getenv("HMAS_LOG_ROTATE_SIZE", "10485760")),
                "rotate_count": int(os.getenv("HMAS_LOG_ROTATE_COUNT", "5")),
                "json_format": os.getenv("HMAS_LOG_JSON", "false").lower() == "true"
            },
            
            # Monitoring configuration
            "monitoring": {
                "enabled": os.getenv("HMAS_MONITORING_ENABLED", "true").lower() == "true",
                "prometheus_enabled": os.getenv("HMAS_PROMETHEUS_ENABLED", "false").lower() == "true",
                "metrics_port": int(os.getenv("HMAS_METRICS_PORT", "9090")),
                "health_check_interval": int(os.getenv("HMAS_HEALTH_CHECK_INTERVAL", "30")),
                "tracing_enabled": os.getenv("HMAS_TRACING_ENABLED", "false").lower() == "true",
                "tracing_sample_rate": float(os.getenv("HMAS_TRACING_SAMPLE_RATE", "0.1"))
            },
            
            # Resource limits
            "resource_limits": {
                "cpu_limit": os.getenv("HMAS_CPU_LIMIT", "1"),
                "memory_limit": os.getenv("HMAS_MEMORY_LIMIT", "1Gi"),
                "storage_limit": os.getenv("HMAS_STORAGE_LIMIT", "10Gi"),
                "max_workers": int(os.getenv("HMAS_MAX_WORKERS", "4")),
                "request_timeout": int(os.getenv("HMAS_REQUEST_TIMEOUT", "30"))
            }
        }
        
        return config_data
    
    def get_config(self) -> EnvironmentConfig:
        """Get the environment configuration"""
        if not self.config:
            raise RuntimeError("Configuration not loaded")
        return self.config
    
    def update_config(self, **kwargs) -> None:
        """Update configuration values"""
        if not self.config:
            raise RuntimeError("Configuration not loaded")
            
        # Update configuration
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
    
    def get_service_url(self, service_name: str) -> str:
        """Get the URL for a specific service"""
        if not self.config:
            raise RuntimeError("Configuration not loaded")
            
        # Use service discovery in production
        if self.env_type == EnvironmentType.PRODUCTION:
            return f"http://{service_name}"
        
        # Use local URLs in development
        return f"http://localhost:{self.config.port}" 