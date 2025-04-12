"""Core Agent implementation for H-MAS."""

from typing import Dict, List, Optional, Any
from uuid import UUID, uuid4
from datetime import datetime
from pydantic import BaseModel, Field
from abc import ABC, abstractmethod
import asyncio
from enum import Enum
import logging

from hmas.config import settings

logger = logging.getLogger(__name__)

class MessagePriority(Enum):
    """Message priority levels."""
    LOW = 0
    NORMAL = 1 
    HIGH = 2
    CRITICAL = 3

class AgentState(BaseModel):
    """Agent state model."""
    
    is_active: bool = True
    memory_usage: int = 0
    last_action: Optional[str] = None
    last_action_time: Optional[datetime] = None
    performance_metrics: Dict[str, float] = Field(default_factory=dict)
    error_count: int = 0

class Agent(ABC):
    """Base class for all agents in the H-MAS framework."""
    
    def __init__(
        self,
        name: str,
        capabilities: List[str],
        memory_size: int = 1000,
        team_id: Optional[UUID] = None,
        org_id: Optional[UUID] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        self.id = uuid4()
        self.name = name
        self.capabilities = capabilities
        self.memory_size = memory_size
        self.team_id = team_id
        self.org_id = org_id
        self.config = config or {}
        self.state = {}
        self.active = False
        self._message_queue: asyncio.Queue = asyncio.Queue()
        self._priority_queues: Dict[MessagePriority, asyncio.Queue] = {
            priority: asyncio.Queue() for priority in MessagePriority
        }
        self._is_running = False
        self._message_handlers = {}
        self._error_count = 0
        self._max_retries = 3
        self._backoff_factor = 1.5
        
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize agent resources and connections."""
        pass
        
    @abstractmethod
    async def process(self, input_data: Any) -> Any:
        """Process input data and return results."""
        pass
        
    @abstractmethod
    async def communicate(self, message: Dict[str, Any], target_id: UUID) -> bool:
        """Send a message to another agent."""
        pass
        
    @abstractmethod
    async def learn(self, experience: Dict[str, Any]) -> bool:
        """Update agent knowledge based on experience."""
        pass
        
    @abstractmethod
    async def reflect(self) -> Dict[str, Any]:
        """Perform self-assessment and optimization."""
        pass
        
    async def start(self) -> bool:
        """Activate the agent."""
        try:
            success = await self.initialize()
            if success:
                self.active = True
            await self._process_priority_messages()
            await self._process_regular_messages()
            return success
        except Exception as e:
            print(f"Error starting agent {self.name}: {str(e)}")
            return False
            
    async def stop(self) -> bool:
        """Deactivate the agent."""
        self.active = False
        self._is_running = False
        return True
        
    def get_status(self) -> Dict[str, Any]:
        """Return current agent status."""
        return {
            "id": str(self.id),
            "name": self.name,
            "capabilities": self.capabilities,
            "active": self.active,
            "team_id": str(self.team_id) if self.team_id else None,
            "org_id": str(self.org_id) if self.org_id else None,
            "state": self.state
        }

    async def process_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Process incoming message and return response.
        
        Args:
            message: Input message to process
            
        Returns:
            Dict containing the response
        """
        raise NotImplementedError("Agents must implement process_message")
    
    async def update_state(self, metrics: Dict[str, float]) -> None:
        """Update agent state with new metrics.
        
        Args:
            metrics: Dictionary of performance metrics
        """
        self.state.performance_metrics.update(metrics)
        self.state.last_action = "state_update"
        self.state.last_action_time = datetime.utcnow()
    
    def has_capability(self, capability: str) -> bool:
        """Check if agent has a specific capability.
        
        Args:
            capability: Capability to check
            
        Returns:
            bool: True if agent has capability
        """
        return capability in self.capabilities
    
    @property
    def is_active(self) -> bool:
        """Check if agent is active.
        
        Returns:
            bool: True if agent is active
        """
        return self.active
    
    @property
    def memory_usage(self) -> int:
        """Get current memory usage.
        
        Returns:
            int: Current memory usage
        """
        return self.state.memory_usage
    
    def __repr__(self) -> str:
        """Get string representation of agent.
        
        Returns:
            str: Agent representation
        """
        return f"Agent(id={self.id}, name={self.name}, capabilities={self.capabilities})"

    async def _process_priority_messages(self) -> None:
        """Process messages from priority queues."""
        while self._is_running:
            # Process messages in priority order
            for priority in reversed(list(MessagePriority)):
                queue = self._priority_queues[priority]
                while not queue.empty():
                    message = await queue.get()
                    await self._handle_message_with_retry(message)
                    queue.task_done()
            await asyncio.sleep(0.1)

    async def _process_regular_messages(self) -> None:
        """Process messages from regular queue."""
        while self._is_running:
            message = await self._message_queue.get()
            await self._handle_message_with_retry(message)
            self._message_queue.task_done()

    async def _handle_message_with_retry(self, message: Dict[str, Any]) -> None:
        """Handle message with retry logic."""
        retries = 0
        while retries < self._max_retries:
            try:
                await self.process_message(message)
                return
            except Exception as e:
                retries += 1
                if retries == self._max_retries:
                    logger.error(f"Failed to process message after {retries} retries: {str(e)}")
                    self._error_count += 1
                    raise
                wait_time = self._backoff_factor ** retries
                logger.warning(f"Retrying message processing in {wait_time}s")
                await asyncio.sleep(wait_time)

    async def send_message(
        self,
        to_agent: UUID,
        message: Dict[str, Any],
        priority: MessagePriority = MessagePriority.NORMAL
    ) -> None:
        """Send message to another agent."""
        try:
            message_with_metadata = {
                "content": message,
                "priority": priority.value,
                "retry_count": 0
            }
            
            if priority in (MessagePriority.HIGH, MessagePriority.CRITICAL):
                await self._priority_queues[priority].put(message_with_metadata)
            else:
                await self._message_queue.put(message_with_metadata)
                
            logger.debug(f"Queued message to {to_agent} with priority {priority.name}")
            
        except Exception as e:
            logger.error(f"Error sending message: {str(e)}")
            raise

    @property
    def error_count(self) -> int:
        """Get number of message processing errors."""
        return self._error_count

    def register_message_handler(
        self,
        message_type: str,
        handler: callable
    ) -> None:
        """Register handler for specific message type."""
        self._message_handlers[message_type] = handler

    def get_queue_sizes(self) -> Dict[str, int]:
        """Get current sizes of message queues."""
        return {
            "regular": self._message_queue.qsize(),
            **{f"priority_{p.name}": q.qsize() 
               for p, q in self._priority_queues.items()}
        } 