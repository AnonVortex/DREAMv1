"""
Perception service configuration module.
Extends the base HMAS configuration with perception-specific settings.
"""

from typing import List, Optional, Dict, Any
from pydantic import Field
from shared.config import ServiceConfig
from enum import Enum

class ModelType(str, Enum):
    """Types of perception models"""
    VISION = "vision"
    AUDIO = "audio"
    TEXT = "text"
    MULTIMODAL = "multimodal"

class PerceptionConfig(ServiceConfig):
    """
    Perception service specific configuration.
    Extends the base service configuration with perception-specific settings.
    """
    # Model Configuration
    model_type: ModelType = Field(
        default=ModelType.VISION,
        description="Type of perception model to use"
    )
    model_path: str = Field(
        default="models/perception",
        description="Path to model files"
    )
    model_version: str = Field(
        default="1.0.0",
        description="Model version"
    )
    
    # Processing Configuration
    batch_size: int = Field(
        default=32,
        description="Batch size for model inference"
    )
    max_input_size: int = Field(
        default=1024 * 1024,  # 1MB
        description="Maximum input size in bytes"
    )
    supported_formats: List[str] = Field(
        default=["jpg", "png", "mp3", "wav", "txt"],
        description="List of supported input formats"
    )
    
    # Performance Settings
    use_gpu: bool = Field(
        default=True,
        description="Whether to use GPU for inference"
    )
    num_inference_threads: int = Field(
        default=2,
        description="Number of inference threads"
    )
    inference_timeout: int = Field(
        default=10,
        description="Inference timeout in seconds"
    )
    
    # Cache Settings
    result_cache_ttl: int = Field(
        default=3600,
        description="Cache TTL for perception results in seconds"
    )
    max_cache_size: int = Field(
        default=1000,
        description="Maximum number of cached results"
    )

    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration dictionary"""
        return {
            "type": self.model_type,
            "path": self.model_path,
            "version": self.model_version,
            "use_gpu": self.use_gpu,
            "batch_size": self.batch_size
        }

    def get_processing_config(self) -> Dict[str, Any]:
        """Get processing configuration dictionary"""
        return {
            "max_input_size": self.max_input_size,
            "supported_formats": self.supported_formats,
            "num_threads": self.num_inference_threads,
            "timeout": self.inference_timeout
        }

    def get_cache_config(self) -> Dict[str, Any]:
        """Get cache configuration dictionary"""
        return {
            "ttl": self.result_cache_ttl,
            "max_size": self.max_cache_size
        }

    class Config:
        """Pydantic configuration"""
        env_prefix = "PERCEPTION_"  # Environment variables prefix

def load_perception_config(config_file: Optional[str] = None) -> PerceptionConfig:
    """
    Load perception service configuration.
    
    Args:
        config_file: Optional path to config file
        
    Returns:
        PerceptionConfig instance
    """
    return PerceptionConfig(
        service_name="perception",
        config_file=config_file if config_file else None
    )
