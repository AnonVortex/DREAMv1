"""
Core module for the H-MAS AGI framework.

This module provides the foundational classes for building a hierarchical multi-agent system:
- Agent: Base class for individual AI agents
- Team: Manages collaborative agent groups
- Organization: Coordinates teams and functional modules
- Federation: Orchestrates the complete AGI pipeline
"""

from .agent import Agent
from .team import Team
from .organization import Organization
from .federation import Federation

__all__ = [
    'Agent',
    'Team',
    'Organization',
    'Federation'
]

# Version information
__version__ = '0.1.0' 