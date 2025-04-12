"""Environment management for H-MAS."""

from typing import Dict, List, Optional, Type, Any
from uuid import UUID
import asyncio
from datetime import datetime
import logging
from contextlib import asynccontextmanager

from hmas.core.agent import Agent
from hmas.config import settings
from hmas.utils.monitoring import Monitor
from hmas.utils.messaging import MessageBroker

logger = logging.getLogger(__name__)

class Environment:
    """Environment for managing agents and their interactions.
    
    The Environment class provides the infrastructure for:
    - Agent lifecycle management
    - Inter-agent communication
    - Resource monitoring and management
    - System state coordination
    """
    
    def __init__(
        self,
        name: str = "default",
        max_agents: Optional[int] = None,
        monitoring: bool = True,
        **kwargs: Any
    ) -> None:
        """Initialize the environment.
        
        Args:
            name: Environment name
            max_agents: Maximum number of agents (default: from settings)
            monitoring: Enable monitoring (default: True)
            **kwargs: Additional configuration
        """
        self.name = name
        self.max_agents = max_agents or settings.MAX_AGENTS
        self.agents: Dict[UUID, Agent] = {}
        self.monitor = Monitor() if monitoring else None
        self.message_broker = MessageBroker()
        self.start_time = datetime.utcnow()
        self.config = kwargs
        self._running = False
        
    async def initialize(self) -> None:
        """Initialize environment resources."""
        logger.info(f"Initializing environment: {self.name}")
        if self.monitor:
            await self.monitor.start()
        await self.message_broker.connect()
        self._running = True
        
    async def shutdown(self) -> None:
        """Shutdown environment and cleanup resources."""
        logger.info(f"Shutting down environment: {self.name}")
        self._running = False
        
        # Shutdown all agents
        await asyncio.gather(
            *[agent.shutdown() for agent in self.agents.values()],
            return_exceptions=True
        )
        
        # Cleanup resources
        if self.monitor:
            await self.monitor.stop()
        await self.message_broker.disconnect()
        
    async def add_agent(self, agent: Agent) -> None:
        """Add an agent to the environment.
        
        Args:
            agent: Agent instance to add
            
        Raises:
            ValueError: If max agents limit is reached
        """
        if len(self.agents) >= self.max_agents:
            raise ValueError(f"Maximum number of agents ({self.max_agents}) reached")
            
        logger.info(f"Adding agent: {agent.name}")
        await agent.initialize()
        self.agents[agent.id] = agent
        
        if self.monitor:
            self.monitor.track_agent(agent)
            
    async def remove_agent(self, agent_id: UUID) -> None:
        """Remove an agent from the environment.
        
        Args:
            agent_id: UUID of agent to remove
            
        Raises:
            KeyError: If agent not found
        """
        if agent_id not in self.agents:
            raise KeyError(f"Agent {agent_id} not found")
            
        agent = self.agents[agent_id]
        logger.info(f"Removing agent: {agent.name}")
        
        await agent.shutdown()
        if self.monitor:
            self.monitor.untrack_agent(agent)
            
        del self.agents[agent_id]
        
    async def send_message(
        self,
        from_agent: UUID,
        to_agent: UUID,
        message: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Send message between agents.
        
        Args:
            from_agent: Sender agent UUID
            to_agent: Recipient agent UUID
            message: Message content
            
        Returns:
            Dict containing the response
            
        Raises:
            KeyError: If either agent not found
        """
        if from_agent not in self.agents or to_agent not in self.agents:
            raise KeyError("Invalid agent ID")
            
        sender = self.agents[from_agent]
        recipient = self.agents[to_agent]
        
        logger.debug(f"Message from {sender.name} to {recipient.name}")
        return await recipient.process_message(message)
        
    @property
    def is_running(self) -> bool:
        """Check if environment is running.
        
        Returns:
            bool: True if environment is running
        """
        return self._running
        
    @property
    def agent_count(self) -> int:
        """Get number of active agents.
        
        Returns:
            int: Number of agents
        """
        return len(self.agents)
        
    @asynccontextmanager
    async def run(self):
        """Context manager for running the environment.
        
        Example:
            async with env.run():
                # Environment is running
                await env.add_agent(agent)
        """
        try:
            await self.initialize()
            yield self
        finally:
            await self.shutdown()
            
    def __repr__(self) -> str:
        """Get string representation.
        
        Returns:
            str: Environment representation
        """
        return f"Environment(name={self.name}, agents={len(self.agents)})" 