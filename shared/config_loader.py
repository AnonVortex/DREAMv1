"""
Configuration loader utility for HMAS.
Handles loading and validating configurations from various sources.
"""

import os
from typing import Optional, Type, TypeVar, Dict, Any
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError

from shared.config import BaseHMASConfig, ServiceConfig

T = TypeVar('T', bound=BaseHMASConfig)

class ConfigurationError(Exception):
    """Configuration related errors"""
    pass

class ConfigLoader:
    """Configuration loader utility"""
    
    @staticmethod
    def load_env_file(env_file: str) -> None:
        """
        Load environment variables from a file.
        
        Args:
            env_file: Path to the environment file
            
        Raises:
            ConfigurationError: If file not found or invalid
        """
        if not os.path.exists(env_file):
            raise ConfigurationError(f"Environment file not found: {env_file}")
        
        try:
            load_dotenv(env_file)
        except Exception as e:
            raise ConfigurationError(f"Failed to load environment file: {e}")
    
    @staticmethod
    def load_config(config_class: Type[T], env_file: Optional[str] = None) -> T:
        """
        Load configuration from environment and optional file.
        
        Args:
            config_class: Configuration class to instantiate
            env_file: Optional path to environment file
            
        Returns:
            Configuration instance
            
        Raises:
            ConfigurationError: If configuration loading fails
        """
        # Load environment file if provided
        if env_file:
            ConfigLoader.load_env_file(env_file)
        
        try:
            # Create configuration instance
            config = config_class()
            
            # Validate configuration
            config_dict = config.dict()
            ConfigLoader._validate_config(config_dict)
            
            return config
            
        except ValidationError as e:
            raise ConfigurationError(f"Configuration validation failed: {e}")
        except Exception as e:
            raise ConfigurationError(f"Failed to load configuration: {e}")
    
    @staticmethod
    def _validate_config(config: Dict[str, Any]) -> None:
        """
        Validate configuration values.
        
        Args:
            config: Configuration dictionary
            
        Raises:
            ConfigurationError: If validation fails
        """
        # Example validation rules
        if config.get('environment') not in ['development', 'staging', 'production']:
            raise ConfigurationError("Invalid environment value")
            
        if config.get('debug', False) and config.get('environment') == 'production':
            raise ConfigurationError("Debug mode not allowed in production")
            
        # Add more validation rules as needed

def load_service_config(
    service_name: str,
    config_class: Type[ServiceConfig],
    env_file: Optional[str] = None
) -> ServiceConfig:
    """
    Load service-specific configuration.
    
    Args:
        service_name: Name of the service
        config_class: Service configuration class
        env_file: Optional path to environment file
        
    Returns:
        Service configuration instance
    """
    # Determine environment
    env = os.getenv('ENVIRONMENT', 'development')
    
    # Set default config file path if not provided
    if not env_file:
        config_dir = Path(__file__).parent.parent / 'config'
        env_file = config_dir / f"{env}.env"
    
    return ConfigLoader.load_config(
        config_class=config_class,
        env_file=str(env_file)
    ) 