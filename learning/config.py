"""
Learning service configuration module.
Extends the base HMAS configuration with learning-specific settings.
"""

from typing import List, Optional, Dict, Any
from enum import Enum
from pydantic import Field

from shared.config import ServiceConfig

class LearningType(str, Enum):
    """Types of learning approaches"""
    SUPERVISED = "supervised"
    UNSUPERVISED = "unsupervised"
    REINFORCEMENT = "reinforcement"
    META = "meta"
    CURRICULUM = "curriculum"
    TRANSFER = "transfer"

class LearningConfig(ServiceConfig):
    """
    Learning service specific configuration.
    Extends the base service configuration with learning-specific settings.
    """
    # Learning Type Configuration
    learning_type: LearningType = Field(
        default=LearningType.SUPERVISED,
        description="Type of learning approach"
    )
    
    # Basic Learning Parameters
    learning_rate: float = Field(
        default=0.001,
        description="Base learning rate"
    )
    batch_size: int = Field(
        default=32,
        description="Training batch size"
    )
    epochs: int = Field(
        default=10,
        description="Number of training epochs"
    )
    
    # Curriculum Learning Settings
    enable_curriculum: bool = Field(
        default=True,
        description="Enable curriculum learning"
    )
    max_difficulty_level: int = Field(
        default=10,
        description="Maximum difficulty level for curriculum"
    )
    task_completion_threshold: float = Field(
        default=0.8,
        description="Threshold for advancing difficulty"
    )
    
    # Transfer Learning Settings
    enable_transfer: bool = Field(
        default=True,
        description="Enable transfer learning"
    )
    knowledge_base_path: str = Field(
        default="knowledge_base/",
        description="Path to knowledge base"
    )
    adaptation_rate: float = Field(
        default=0.1,
        description="Rate of adaptation for transfer learning"
    )
    
    # Meta-Learning Settings
    enable_meta_learning: bool = Field(
        default=True,
        description="Enable meta-learning"
    )
    inner_learning_rate: float = Field(
        default=0.01,
        description="Inner loop learning rate"
    )
    outer_learning_rate: float = Field(
        default=0.001,
        description="Outer loop learning rate"
    )
    adaptation_steps: int = Field(
        default=5,
        description="Number of adaptation steps"
    )
    
    # Model Management
    model_save_path: str = Field(
        default="models/",
        description="Path to save models"
    )
    checkpoint_interval: int = Field(
        default=1000,
        description="Steps between checkpoints"
    )
    
    # Performance Metrics
    metrics: List[str] = Field(
        default=[
            "accuracy",
            "loss",
            "learning_speed",
            "adaptation_success_rate"
        ],
        description="Metrics to track"
    )
    
    # Resource Management
    max_memory_usage: float = Field(
        default=0.8,
        description="Maximum fraction of memory to use"
    )
    gpu_memory_fraction: float = Field(
        default=0.7,
        description="Fraction of GPU memory to use"
    )
    
    def get_learning_config(self) -> Dict[str, Any]:
        """Get learning configuration dictionary"""
        return {
            "type": self.learning_type,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "epochs": self.epochs
        }
    
    def get_curriculum_config(self) -> Dict[str, Any]:
        """Get curriculum learning configuration"""
        return {
            "enabled": self.enable_curriculum,
            "max_difficulty": self.max_difficulty_level,
            "completion_threshold": self.task_completion_threshold
        }
    
    def get_transfer_config(self) -> Dict[str, Any]:
        """Get transfer learning configuration"""
        return {
            "enabled": self.enable_transfer,
            "knowledge_base_path": self.knowledge_base_path,
            "adaptation_rate": self.adaptation_rate
        }
    
    def get_meta_learning_config(self) -> Dict[str, Any]:
        """Get meta-learning configuration"""
        return {
            "enabled": self.enable_meta_learning,
            "inner_lr": self.inner_learning_rate,
            "outer_lr": self.outer_learning_rate,
            "adaptation_steps": self.adaptation_steps
        }
    
    def get_resource_config(self) -> Dict[str, Any]:
        """Get resource management configuration"""
        return {
            "max_memory": self.max_memory_usage,
            "gpu_memory": self.gpu_memory_fraction
        }
    
    class Config:
        """Pydantic configuration"""
        env_prefix = "LEARNING_"

def load_learning_config(config_file: Optional[str] = None) -> LearningConfig:
    """
    Load learning service configuration.
    
    Args:
        config_file: Optional path to config file
        
    Returns:
        LearningConfig instance
    """
    return LearningConfig(
        service_name="learning",
        config_file=config_file if config_file else None
    ) 