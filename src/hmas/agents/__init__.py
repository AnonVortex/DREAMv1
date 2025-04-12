"""
Specialized agents for the H-MAS AGI framework.

This module provides implementations of specialized agents for different cognitive tasks:
- PerceptionAgent: Multi-modal input processing and feature extraction
- MemoryAgent: Memory management across different types of memory systems
- ReasoningAgent: Causal and logical reasoning capabilities
- LearningAgent: Various types of learning and adaptation capabilities
"""

from .perception_agent import PerceptionAgent
from .memory_agent import MemoryAgent
from .reasoning_agent import ReasoningAgent
from .learning_agent import LearningAgent

__all__ = [
    'PerceptionAgent',
    'MemoryAgent',
    'ReasoningAgent',
    'LearningAgent'
]

# Version information
__version__ = '0.1.0' 