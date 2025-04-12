"""
Integration service configuration module.
Extends the base HMAS configuration with integration-specific settings.
"""

from typing import List, Optional, Dict, Any
from enum import Enum
from pydantic import Field

from shared.config import ServiceConfig

class IntegrationType(str, Enum):
    """Types of integrations"""
    REST_API = "rest_api"
    GRPC = "grpc"
    WEBHOOK = "webhook"
    EVENT_STREAM = "event_stream"
    MESSAGE_QUEUE = "message_queue"

class AuthType(str, Enum):
    """Types of authentication"""
    API_KEY = "api_key"
    JWT = "jwt"
    OAUTH2 = "oauth2"
    BASIC = "basic"
    NONE = "none"

class IntegrationConfig(ServiceConfig):
    """
    Integration service specific configuration.
    Extends the base service configuration with integration-specific settings.
    """
    # Integration Types Configuration
    enabled_integrations: List[IntegrationType] = Field(
        default=[IntegrationType.REST_API, IntegrationType.WEBHOOK],
        description="Enabled integration types"
    )
    default_integration: IntegrationType = Field(
        default=IntegrationType.REST_API,
        description="Default integration type"
    )
    
    # Authentication Configuration
    auth_types: List[AuthType] = Field(
        default=[AuthType.API_KEY, AuthType.JWT],
        description="Supported authentication types"
    )
    default_auth: AuthType = Field(
        default=AuthType.API_KEY,
        description="Default authentication type"
    )
    token_expiry: int = Field(
        default=3600,
        description="Token expiry time in seconds"
    )
    
    # Connection Settings
    redis_url: str = Field(
        default="redis://localhost:6379",
        description="Redis connection URL"
    )
    max_connections: int = Field(
        default=100,
        description="Maximum number of concurrent connections"
    )
    connection_timeout: float = Field(
        default=30.0,
        description="Connection timeout in seconds"
    )
    
    # Rate Limiting
    enable_rate_limiting: bool = Field(
        default=True,
        description="Enable rate limiting"
    )
    rate_limit: int = Field(
        default=1000,
        description="Maximum requests per minute"
    )
    burst_limit: int = Field(
        default=50,
        description="Maximum burst size"
    )
    
    # Retry Configuration
    max_retries: int = Field(
        default=3,
        description="Maximum number of retries"
    )
    retry_delay: float = Field(
        default=1.0,
        description="Delay between retries in seconds"
    )
    retry_backoff: float = Field(
        default=2.0,
        description="Exponential backoff factor"
    )
    
    # Webhook Configuration
    webhook_timeout: float = Field(
        default=5.0,
        description="Webhook request timeout in seconds"
    )
    webhook_retry_count: int = Field(
        default=3,
        description="Number of webhook delivery attempts"
    )
    webhook_batch_size: int = Field(
        default=10,
        description="Maximum events per webhook request"
    )
    
    # Event Stream Configuration
    event_buffer_size: int = Field(
        default=1000,
        description="Maximum events in buffer"
    )
    event_batch_size: int = Field(
        default=100,
        description="Events per batch"
    )
    event_flush_interval: float = Field(
        default=1.0,
        description="Event flush interval in seconds"
    )
    
    # Monitoring Configuration
    enable_monitoring: bool = Field(
        default=True,
        description="Enable integration monitoring"
    )
    monitor_interval: int = Field(
        default=60,
        description="Monitoring interval in seconds"
    )
    health_check_interval: int = Field(
        default=30,
        description="Health check interval in seconds"
    )
    
    def get_integration_config(self) -> Dict[str, Any]:
        """Get integration configuration dictionary"""
        return {
            "enabled": self.enabled_integrations,
            "default": self.default_integration,
            "max_connections": self.max_connections,
            "timeout": self.connection_timeout
        }
    
    def get_auth_config(self) -> Dict[str, Any]:
        """Get authentication configuration dictionary"""
        return {
            "types": self.auth_types,
            "default": self.default_auth,
            "token_expiry": self.token_expiry
        }
    
    def get_rate_limit_config(self) -> Dict[str, Any]:
        """Get rate limiting configuration dictionary"""
        return {
            "enabled": self.enable_rate_limiting,
            "rate_limit": self.rate_limit,
            "burst_limit": self.burst_limit
        }
    
    def get_webhook_config(self) -> Dict[str, Any]:
        """Get webhook configuration dictionary"""
        return {
            "timeout": self.webhook_timeout,
            "retry_count": self.webhook_retry_count,
            "batch_size": self.webhook_batch_size
        }
    
    def get_event_stream_config(self) -> Dict[str, Any]:
        """Get event stream configuration dictionary"""
        return {
            "buffer_size": self.event_buffer_size,
            "batch_size": self.event_batch_size,
            "flush_interval": self.event_flush_interval
        }
    
    class Config:
        """Pydantic configuration"""
        env_prefix = "INTEGRATION_"

def load_integration_config(config_file: Optional[str] = None) -> IntegrationConfig:
    """
    Load integration service configuration.
    
    Args:
        config_file: Optional path to config file
        
    Returns:
        IntegrationConfig instance
    """
    return IntegrationConfig(
        service_name="integration",
        config_file=config_file if config_file else None
    )
