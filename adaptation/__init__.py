"""
HMAS Adaptation Module

This module provides adaptive architecture capabilities for the Hierarchical Multi-Agent System,
including dynamic scaling, load balancing, and self-modification.
"""

from .adaptation_service import (
    app,
    ResourceMetrics,
    AgentConfig,
    AdaptationRule,
    ArchitectureConfig,
    ResourceManager,
    DynamicScaler,
    LoadBalancer,
    SelfModifier,
    EvolutionaryArchitect
)

__all__ = [
    'app',
    'ResourceMetrics',
    'AgentConfig',
    'AdaptationRule',
    'ArchitectureConfig',
    'ResourceManager',
    'DynamicScaler',
    'LoadBalancer',
    'SelfModifier',
    'EvolutionaryArchitect'
] 