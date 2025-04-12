"""
Base Agent Class

This module defines the base agent class that all specialized agents will inherit from.
It provides the core functionality and interface for agent operations.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod

@dataclass
class AgentState:
    """Represents the current state of an agent."""
    name: str
    capabilities: List[str]
    memory_size: int
    status: str = "idle"
    last_activity: Optional[float] = None
    metrics: Dict[str, Any] = None

class BaseAgent(ABC):
    """
    Base class for all agents in the H-MAS system.
    Provides core functionality and interface for agent operations.
    """

    def __init__(self, name: str, capabilities: List[str], memory_size: int = 1000):
        """
        Initialize the base agent.

        Args:
            name: Unique identifier for the agent
            capabilities: List of capabilities this agent possesses
            memory_size: Maximum size of the agent's memory
        """
        self.state = AgentState(
            name=name,
            capabilities=capabilities,
            memory_size=memory_size,
            metrics={}
        )

    @abstractmethod
    async def process(self, input_data: Any) -> Any:
        """
        Process input data and return results.
        Must be implemented by all agent subclasses.

        Args:
            input_data: Input data to be processed

        Returns:
            Processed results
        """
        pass

    @abstractmethod
    async def learn(self, experience: Any) -> None:
        """
        Learn from experience.
        Must be implemented by all agent subclasses.

        Args:
            experience: Experience data to learn from
        """
        pass

    def get_state(self) -> AgentState:
        """
        Get the current state of the agent.

        Returns:
            Current agent state
        """
        return self.state

    def update_metrics(self, metric_name: str, value: Any) -> None:
        """
        Update agent metrics.

        Args:
            metric_name: Name of the metric to update
            value: New value for the metric
        """
        self.state.metrics[metric_name] = value

    def has_capability(self, capability: str) -> bool:
        """
        Check if the agent has a specific capability.

        Args:
            capability: Capability to check for

        Returns:
            True if the agent has the capability, False otherwise
        """
        return capability in self.state.capabilities

    async def initialize(self) -> None:
        """
        Initialize the agent.
        Can be overridden by subclasses for specific initialization needs.
        """
        self.state.status = "initialized"

    async def shutdown(self) -> None:
        """
        Shutdown the agent.
        Can be overridden by subclasses for specific shutdown needs.
        """
        self.state.status = "shutdown" 