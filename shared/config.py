"""
Configuration management for HMAS services.
"""

from shared.branding import (
    PROJECT_NAME,
    PROJECT_FULL_NAME,
    AGI_NAME,
    AGI_FULL_NAME,
    VERSION,
    BUILD,
    ORGANIZATION_NAME,
    ORGANIZATION_DOMAIN
)

from shared.environment import (
    EnvironmentConfig,
    EnvironmentManager,
    EnvironmentType,
    DatabaseConfig,
    RedisConfig,
    SecurityConfig,
    LoggingConfig,
    MonitoringConfig,
    ResourceLimits
)

# Re-export environment configuration classes
__all__ = [
    "EnvironmentConfig",
    "EnvironmentManager",
    "EnvironmentType",
    "DatabaseConfig",
    "RedisConfig",
    "SecurityConfig",
    "LoggingConfig",
    "MonitoringConfig",
    "ResourceLimits",
    "get_config",
    "initialize_config"
]

# Global configuration instance
_config_instance = None

def initialize_config(
    service_name: str,
    env_type: EnvironmentType = None,
    config_path: str = None
) -> EnvironmentConfig:
    """Initialize the global configuration."""
    global _config_instance
    
    if _config_instance is None:
        env_manager = EnvironmentManager(
            service_name=service_name,
            env_type=env_type,
            config_path=config_path
        )
        _config_instance = env_manager.get_config()
    
    return _config_instance

def get_config() -> EnvironmentConfig:
    """Get the global configuration instance."""
    if _config_instance is None:
        raise RuntimeError(
            "Configuration not initialized. Call initialize_config first."
        )
    return _config_instance 