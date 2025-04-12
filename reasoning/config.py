"""
Reasoning service configuration module.
Extends the base HMAS configuration with reasoning-specific settings.
"""

from typing import List, Optional, Dict, Any
from enum import Enum
from pydantic import Field

from shared.config import ServiceConfig

class ReasoningType(str, Enum):
    """Types of reasoning approaches"""
    SYMBOLIC = "symbolic"
    CAUSAL = "causal"
    COMMON_SENSE = "common_sense"
    PROBABILISTIC = "probabilistic"
    HYBRID = "hybrid"

class ReasoningConfig(ServiceConfig):
    """
    Reasoning service specific configuration.
    Extends the base service configuration with reasoning-specific settings.
    """
    # Reasoning Type Configuration
    reasoning_types: List[ReasoningType] = Field(
        default=[ReasoningType.SYMBOLIC, ReasoningType.CAUSAL, ReasoningType.COMMON_SENSE],
        description="Enabled reasoning types"
    )
    inference_depth: int = Field(
        default=3,
        description="Maximum inference depth"
    )
    
    # Knowledge Graph Settings
    knowledge_graph_path: str = Field(
        default="knowledge_graph/",
        description="Path to knowledge graph storage"
    )
    graph_update_interval: int = Field(
        default=3600,
        description="Graph update interval in seconds"
    )
    max_graph_size: int = Field(
        default=1000000,
        description="Maximum number of nodes in knowledge graph"
    )
    
    # Symbolic Reasoning Settings
    rule_base_path: str = Field(
        default="rules/",
        description="Path to rule base"
    )
    max_rule_chain_length: int = Field(
        default=10,
        description="Maximum length of rule inference chain"
    )
    confidence_threshold: float = Field(
        default=0.7,
        description="Minimum confidence threshold for rule application"
    )
    
    # Causal Reasoning Settings
    causal_graph_path: str = Field(
        default="causal_graphs/",
        description="Path to causal graph storage"
    )
    max_causal_chain_length: int = Field(
        default=5,
        description="Maximum length of causal inference chain"
    )
    min_causal_strength: float = Field(
        default=0.3,
        description="Minimum strength for causal relationships"
    )
    
    # Common Sense Settings
    common_sense_rules_path: str = Field(
        default="common_sense_rules/",
        description="Path to common sense rules"
    )
    domain_specific_rules: Dict[str, str] = Field(
        default={
            "general": "general_rules.json",
            "physical": "physical_rules.json",
            "social": "social_rules.json"
        },
        description="Domain-specific rule files"
    )
    
    # Performance Settings
    max_reasoning_time: float = Field(
        default=5.0,
        description="Maximum time for reasoning in seconds"
    )
    memory_limit: int = Field(
        default=1024 * 1024 * 1024,  # 1GB
        description="Maximum memory usage in bytes"
    )
    
    # Cache Settings
    enable_cache: bool = Field(
        default=True,
        description="Enable reasoning cache"
    )
    cache_size: int = Field(
        default=1000,
        description="Maximum number of cached results"
    )
    cache_ttl: int = Field(
        default=3600,
        description="Cache TTL in seconds"
    )
    
    def get_reasoning_config(self) -> Dict[str, Any]:
        """Get reasoning configuration dictionary"""
        return {
            "types": self.reasoning_types,
            "inference_depth": self.inference_depth,
            "confidence_threshold": self.confidence_threshold
        }
    
    def get_knowledge_graph_config(self) -> Dict[str, Any]:
        """Get knowledge graph configuration"""
        return {
            "path": self.knowledge_graph_path,
            "update_interval": self.graph_update_interval,
            "max_size": self.max_graph_size
        }
    
    def get_symbolic_config(self) -> Dict[str, Any]:
        """Get symbolic reasoning configuration"""
        return {
            "rule_base_path": self.rule_base_path,
            "max_chain_length": self.max_rule_chain_length,
            "confidence_threshold": self.confidence_threshold
        }
    
    def get_causal_config(self) -> Dict[str, Any]:
        """Get causal reasoning configuration"""
        return {
            "graph_path": self.causal_graph_path,
            "max_chain_length": self.max_causal_chain_length,
            "min_strength": self.min_causal_strength
        }
    
    def get_common_sense_config(self) -> Dict[str, Any]:
        """Get common sense reasoning configuration"""
        return {
            "rules_path": self.common_sense_rules_path,
            "domain_rules": self.domain_specific_rules
        }
    
    def get_performance_config(self) -> Dict[str, Any]:
        """Get performance configuration"""
        return {
            "max_time": self.max_reasoning_time,
            "memory_limit": self.memory_limit
        }
    
    class Config:
        """Pydantic configuration"""
        env_prefix = "REASONING_"

def load_reasoning_config(config_file: Optional[str] = None) -> ReasoningConfig:
    """
    Load reasoning service configuration.
    
    Args:
        config_file: Optional path to config file
        
    Returns:
        ReasoningConfig instance
    """
    return ReasoningConfig(
        service_name="reasoning",
        config_file=config_file if config_file else None
    ) 