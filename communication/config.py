"""
Communication service configuration module.
Extends the base HMAS configuration with communication-specific settings.
"""

from typing import List, Optional, Dict, Any
from enum import Enum
from pydantic import Field

from shared.config import ServiceConfig

class ProtocolType(str, Enum):
    """Types of communication protocols"""
    HTTP = "http"
    WEBSOCKET = "websocket"
    GRPC = "grpc"
    MQTT = "mqtt"
    REDIS_PUBSUB = "redis_pubsub"

class MessageFormat(str, Enum):
    """Supported message formats"""
    JSON = "json"
    PROTOBUF = "protobuf"
    AVRO = "avro"
    MSGPACK = "msgpack"

class CommunicationConfig(ServiceConfig):
    """
    Communication service specific configuration.
    Extends the base service configuration with communication-specific settings.
    """
    # Protocol Configuration
    enabled_protocols: List[ProtocolType] = Field(
        default=[ProtocolType.HTTP, ProtocolType.WEBSOCKET],
        description="Enabled communication protocols"
    )
    default_protocol: ProtocolType = Field(
        default=ProtocolType.HTTP,
        description="Default communication protocol"
    )
    
    # Message Configuration
    message_formats: List[MessageFormat] = Field(
        default=[MessageFormat.JSON],
        description="Supported message formats"
    )
    default_format: MessageFormat = Field(
        default=MessageFormat.JSON,
        description="Default message format"
    )
    max_message_size: int = Field(
        default=1024 * 1024,  # 1MB
        description="Maximum message size in bytes"
    )
    
    # Connection Settings
    redis_url: str = Field(
        default="redis://localhost:6379",
        description="Redis connection URL for pub/sub"
    )
    mqtt_broker: Optional[str] = Field(
        default=None,
        description="MQTT broker URL"
    )
    grpc_port: int = Field(
        default=50051,
        description="gRPC server port"
    )
    
    # Security Settings
    enable_tls: bool = Field(
        default=True,
        description="Enable TLS encryption"
    )
    require_authentication: bool = Field(
        default=True,
        description="Require client authentication"
    )
    auth_token_expiry: int = Field(
        default=3600,
        description="Authentication token expiry in seconds"
    )
    
    # Rate Limiting
    rate_limit_enabled: bool = Field(
        default=True,
        description="Enable rate limiting"
    )
    max_requests_per_minute: int = Field(
        default=1000,
        description="Maximum requests per minute per client"
    )
    burst_size: int = Field(
        default=50,
        description="Maximum burst size for rate limiting"
    )
    
    # Retry Configuration
    max_retries: int = Field(
        default=3,
        description="Maximum number of retry attempts"
    )
    retry_delay: float = Field(
        default=1.0,
        description="Delay between retries in seconds"
    )
    retry_backoff: float = Field(
        default=2.0,
        description="Exponential backoff factor"
    )
    
    # Monitoring
    enable_metrics: bool = Field(
        default=True,
        description="Enable communication metrics"
    )
    metrics_port: int = Field(
        default=8901,
        description="Metrics server port"
    )
    log_messages: bool = Field(
        default=True,
        description="Log message metadata"
    )
    
    def get_protocol_config(self) -> Dict[str, Any]:
        """Get protocol configuration dictionary"""
        return {
            "enabled": self.enabled_protocols,
            "default": self.default_protocol,
            "grpc_port": self.grpc_port,
            "mqtt_broker": self.mqtt_broker
        }
    
    def get_message_config(self) -> Dict[str, Any]:
        """Get message configuration dictionary"""
        return {
            "formats": self.message_formats,
            "default_format": self.default_format,
            "max_size": self.max_message_size
        }
    
    def get_security_config(self) -> Dict[str, Any]:
        """Get security configuration dictionary"""
        return {
            "tls_enabled": self.enable_tls,
            "auth_required": self.require_authentication,
            "token_expiry": self.auth_token_expiry
        }
    
    def get_rate_limit_config(self) -> Dict[str, Any]:
        """Get rate limiting configuration dictionary"""
        return {
            "enabled": self.rate_limit_enabled,
            "max_rpm": self.max_requests_per_minute,
            "burst_size": self.burst_size
        }
    
    def get_retry_config(self) -> Dict[str, Any]:
        """Get retry configuration dictionary"""
        return {
            "max_retries": self.max_retries,
            "delay": self.retry_delay,
            "backoff": self.retry_backoff
        }
    
    class Config:
        """Pydantic configuration"""
        env_prefix = "COMM_"

def load_communication_config(config_file: Optional[str] = None) -> CommunicationConfig:
    """
    Load communication service configuration.
    
    Args:
        config_file: Optional path to config file
        
    Returns:
        CommunicationConfig instance
    """
    return CommunicationConfig(
        service_name="communication",
        config_file=config_file if config_file else None
    )
