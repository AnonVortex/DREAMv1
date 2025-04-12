"""
Feedback service configuration module.
Extends the base HMAS configuration with feedback-specific settings.
"""

from typing import List, Optional, Dict, Any
from enum import Enum
from pydantic import Field

from shared.config import ServiceConfig

class FeedbackType(str, Enum):
    """Types of feedback"""
    REWARD = "reward"
    CORRECTION = "correction"
    GUIDANCE = "guidance"
    CRITIQUE = "critique"
    PREFERENCE = "preference"

class FeedbackSource(str, Enum):
    """Sources of feedback"""
    HUMAN = "human"
    AGENT = "agent"
    ENVIRONMENT = "environment"
    SYSTEM = "system"

class FeedbackConfig(ServiceConfig):
    """
    Feedback service specific configuration.
    Extends the base service configuration with feedback-specific settings.
    """
    # Feedback Types Configuration
    enabled_feedback_types: List[FeedbackType] = Field(
        default=[FeedbackType.REWARD, FeedbackType.CORRECTION],
        description="Enabled feedback types"
    )
    allowed_sources: List[FeedbackSource] = Field(
        default=[FeedbackSource.HUMAN, FeedbackSource.AGENT],
        description="Allowed feedback sources"
    )
    
    # Storage Configuration
    redis_url: str = Field(
        default="redis://localhost:6379",
        description="Redis connection URL"
    )
    feedback_ttl: int = Field(
        default=604800,  # 1 week
        description="Feedback storage TTL in seconds"
    )
    max_feedback_size: int = Field(
        default=1024 * 1024,  # 1MB
        description="Maximum feedback size in bytes"
    )
    
    # Processing Configuration
    batch_size: int = Field(
        default=32,
        description="Batch size for feedback processing"
    )
    processing_interval: float = Field(
        default=1.0,
        description="Feedback processing interval in seconds"
    )
    max_processing_time: float = Field(
        default=5.0,
        description="Maximum processing time per feedback in seconds"
    )
    
    # Priority Configuration
    enable_priority: bool = Field(
        default=True,
        description="Enable feedback priority"
    )
    priority_levels: int = Field(
        default=3,
        description="Number of priority levels"
    )
    priority_weights: Dict[str, float] = Field(
        default={
            "high": 1.0,
            "medium": 0.5,
            "low": 0.1
        },
        description="Priority level weights"
    )
    
    # Integration Configuration
    learning_service_url: Optional[str] = Field(
        default=None,
        description="Learning service URL for feedback integration"
    )
    memory_service_url: Optional[str] = Field(
        default=None,
        description="Memory service URL for feedback storage"
    )
    
    # Aggregation Configuration
    enable_aggregation: bool = Field(
        default=True,
        description="Enable feedback aggregation"
    )
    aggregation_window: int = Field(
        default=3600,  # 1 hour
        description="Aggregation window in seconds"
    )
    min_samples: int = Field(
        default=5,
        description="Minimum samples for aggregation"
    )
    
    # Monitoring Configuration
    enable_monitoring: bool = Field(
        default=True,
        description="Enable feedback monitoring"
    )
    monitor_interval: int = Field(
        default=60,
        description="Monitoring interval in seconds"
    )
    alert_threshold: float = Field(
        default=0.8,
        description="Alert threshold for negative feedback ratio"
    )
    
    def get_feedback_config(self) -> Dict[str, Any]:
        """Get feedback configuration dictionary"""
        return {
            "types": self.enabled_feedback_types,
            "sources": self.allowed_sources,
            "max_size": self.max_feedback_size
        }
    
    def get_processing_config(self) -> Dict[str, Any]:
        """Get processing configuration dictionary"""
        return {
            "batch_size": self.batch_size,
            "interval": self.processing_interval,
            "max_time": self.max_processing_time
        }
    
    def get_priority_config(self) -> Dict[str, Any]:
        """Get priority configuration dictionary"""
        return {
            "enabled": self.enable_priority,
            "levels": self.priority_levels,
            "weights": self.priority_weights
        }
    
    def get_aggregation_config(self) -> Dict[str, Any]:
        """Get aggregation configuration dictionary"""
        return {
            "enabled": self.enable_aggregation,
            "window": self.aggregation_window,
            "min_samples": self.min_samples
        }
    
    def get_monitoring_config(self) -> Dict[str, Any]:
        """Get monitoring configuration dictionary"""
        return {
            "enabled": self.enable_monitoring,
            "interval": self.monitor_interval,
            "alert_threshold": self.alert_threshold
        }
    
    class Config:
        """Pydantic configuration"""
        env_prefix = "FEEDBACK_"

def load_feedback_config(config_file: Optional[str] = None) -> FeedbackConfig:
    """
    Load feedback service configuration.
    
    Args:
        config_file: Optional path to config file
        
    Returns:
        FeedbackConfig instance
    """
    return FeedbackConfig(
        service_name="feedback",
        config_file=config_file if config_file else None
    )
