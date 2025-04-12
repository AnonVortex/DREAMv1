"""
Environment Class

This module defines the main environment class that manages the multi-agent system,
including agent coordination, communication, and resource allocation.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import asyncio
import logging
from ..agents.base_agent import BaseAgent
from .resource_manager import ResourceManager
from .monitor import SystemMonitor

@dataclass
class EnvironmentState:
    """Represents the current state of the environment."""
    agents: Dict[str, BaseAgent]
    status: str = "initializing"
    metrics: Dict[str, Any] = None
    last_update: Optional[float] = None

class Environment:
    """
    Main environment class that manages the multi-agent system.
    Handles agent coordination, resource allocation, and system monitoring.
    """

    def __init__(self, agents: List[BaseAgent], communication_protocol: str = "http"):
        """
        Initialize the environment.

        Args:
            agents: List of agents to be managed
            communication_protocol: Protocol used for inter-agent communication
        """
        self.state = EnvironmentState(
            agents={agent.state.name: agent for agent in agents},
            metrics={}
        )
        self.resource_manager = ResourceManager()
        self.monitor = SystemMonitor()
        self.communication_protocol = communication_protocol
        self.logger = logging.getLogger(__name__)

    async def start(self) -> None:
        """
        Start the environment and initialize all agents.
        """
        self.logger.info("Starting environment...")
        self.state.status = "starting"

        # Initialize all agents
        for agent in self.state.agents.values():
            await agent.initialize()
            self.logger.info(f"Agent {agent.state.name} initialized")

        # Start monitoring
        await self.monitor.start()
        self.state.status = "running"
        self.logger.info("Environment started successfully")

    async def stop(self) -> None:
        """
        Stop the environment and shutdown all agents.
        """
        self.logger.info("Stopping environment...")
        self.state.status = "stopping"

        # Shutdown all agents
        for agent in self.state.agents.values():
            await agent.shutdown()
            self.logger.info(f"Agent {agent.state.name} shutdown")

        # Stop monitoring
        await self.monitor.stop()
        self.state.status = "stopped"
        self.logger.info("Environment stopped successfully")

    async def add_agent(self, agent: BaseAgent) -> None:
        """
        Add a new agent to the environment.

        Args:
            agent: Agent to be added
        """
        if agent.state.name in self.state.agents:
            raise ValueError(f"Agent with name {agent.state.name} already exists")

        await agent.initialize()
        self.state.agents[agent.state.name] = agent
        self.logger.info(f"Agent {agent.state.name} added to environment")

    async def remove_agent(self, agent_name: str) -> None:
        """
        Remove an agent from the environment.

        Args:
            agent_name: Name of the agent to remove
        """
        if agent_name not in self.state.agents:
            raise ValueError(f"Agent {agent_name} not found")

        agent = self.state.agents[agent_name]
        await agent.shutdown()
        del self.state.agents[agent_name]
        self.logger.info(f"Agent {agent_name} removed from environment")

    async def broadcast_message(self, message: Any, sender: str) -> None:
        """
        Broadcast a message to all agents in the environment.

        Args:
            message: Message to broadcast
            sender: Name of the sending agent
        """
        tasks = []
        for agent_name, agent in self.state.agents.items():
            if agent_name != sender:
                tasks.append(agent.process(message))
        
        await asyncio.gather(*tasks)
        self.logger.info(f"Message broadcast from {sender} completed")

    def get_agent(self, agent_name: str) -> Optional[BaseAgent]:
        """
        Get an agent by name.

        Args:
            agent_name: Name of the agent to retrieve

        Returns:
            The requested agent or None if not found
        """
        return self.state.agents.get(agent_name)

    def get_environment_state(self) -> EnvironmentState:
        """
        Get the current state of the environment.

        Returns:
            Current environment state
        """
        return self.state

    async def update_metrics(self) -> None:
        """
        Update environment metrics.
        """
        metrics = await self.monitor.get_metrics()
        self.state.metrics.update(metrics)
        self.state.last_update = asyncio.get_event_loop().time() 