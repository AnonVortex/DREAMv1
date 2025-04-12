"""Configuration management for H-MAS."""

from typing import Any, Dict, Optional
from pydantic import BaseSettings, PostgresDsn, RedisDsn, SecretStr, validator
import os
from pathlib import Path

class Settings(BaseSettings):
    """Global settings for H-MAS."""
    
    # Application Settings
    APP_NAME: str = "H-MAS"
    APP_ENV: str = "development"
    APP_DEBUG: bool = True
    APP_VERSION: str = "0.1.0"
    APP_SECRET_KEY: SecretStr
    
    # Database Settings
    DATABASE_URL: PostgresDsn
    DATABASE_POOL_SIZE: int = 5
    DATABASE_MAX_OVERFLOW: int = 10
    
    # Redis Settings
    REDIS_URL: RedisDsn
    REDIS_MAX_CONNECTIONS: int = 10
    
    # Security Settings
    JWT_SECRET_KEY: SecretStr
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    JWT_REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    
    # Agent Settings
    MAX_AGENTS: int = 100
    AGENT_MEMORY_LIMIT: int = 1000
    AGENT_TIMEOUT: int = 30
    
    # Monitoring Settings
    MONITORING_ENABLED: bool = True
    MONITORING_INTERVAL: int = 60
    ALERT_THRESHOLD_CPU: int = 80
    ALERT_THRESHOLD_MEMORY: int = 85
    ALERT_THRESHOLD_DISK: int = 90
    
    # Logging Settings
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"
    LOG_FILE: Optional[Path] = None
    
    @validator("DATABASE_URL", "REDIS_URL", pre=True)
    def validate_urls(cls, v: Optional[str], field: str) -> Any:
        """Validate and format database and redis URLs."""
        if isinstance(v, str):
            return v
        raise ValueError(f"{field} must be a valid connection URL")
    
    @validator("LOG_LEVEL")
    def validate_log_level(cls, v: str) -> str:
        """Validate log level."""
        allowed_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in allowed_levels:
            raise ValueError(f"Log level must be one of {allowed_levels}")
        return v.upper()
    
    class Config:
        """Pydantic model configuration."""
        
        case_sensitive = True
        env_file = ".env"
        env_file_encoding = "utf-8"

def get_settings() -> Settings:
    """Get application settings.
    
    Returns:
        Settings: Application settings instance
    """
    return Settings()

# Global settings instance
settings = get_settings() 