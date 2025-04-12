"""
Message Broker

This module implements the message broker for handling inter-agent communication,
including message routing, queuing, and delivery.
"""

from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass
import asyncio
import logging
from datetime import datetime

@dataclass
class Message:
    """Represents a message between agents."""
    sender: str
    recipient: str
    content: Any
    timestamp: datetime
    priority: int = 0
    metadata: Dict[str, Any] = None

class MessageBroker:
    """
    Handles message routing and delivery between agents.
    Provides methods for sending, receiving, and managing messages.
    """

    def __init__(self):
        """Initialize the message broker."""
        self.message_queue = asyncio.PriorityQueue()
        self.subscribers: Dict[str, List[Callable]] = {}
        self.logger = logging.getLogger(__name__)
        self._processing_task = None

    async def start(self) -> None:
        """Start the message broker."""
        self._processing_task = asyncio.create_task(self._process_messages())
        self.logger.info("Message broker started")

    async def stop(self) -> None:
        """Stop the message broker."""
        if self._processing_task:
            self._processing_task.cancel()
            try:
                await self._processing_task
            except asyncio.CancelledError:
                pass
        self.logger.info("Message broker stopped")

    async def _process_messages(self) -> None:
        """Process messages from the queue."""
        while True:
            try:
                priority, message = await self.message_queue.get()
                await self._deliver_message(message)
                self.message_queue.task_done()
            except Exception as e:
                self.logger.error(f"Error processing message: {e}")
                await asyncio.sleep(1)

    async def _deliver_message(self, message: Message) -> None:
        """
        Deliver a message to its recipient.

        Args:
            message: Message to deliver
        """
        try:
            if message.recipient in self.subscribers:
                for callback in self.subscribers[message.recipient]:
                    await callback(message)
                self.logger.info(f"Message delivered from {message.sender} to {message.recipient}")
            else:
                self.logger.warning(f"No subscribers found for recipient: {message.recipient}")
        except Exception as e:
            self.logger.error(f"Error delivering message: {e}")

    async def send_message(self, sender: str, recipient: str, content: Any, 
                          priority: int = 0, metadata: Dict[str, Any] = None) -> None:
        """
        Send a message to a recipient.

        Args:
            sender: Sender's identifier
            recipient: Recipient's identifier
            content: Message content
            priority: Message priority (higher numbers = higher priority)
            metadata: Additional message metadata
        """
        message = Message(
            sender=sender,
            recipient=recipient,
            content=content,
            timestamp=datetime.now(),
            priority=priority,
            metadata=metadata or {}
        )
        await self.message_queue.put((priority, message))
        self.logger.info(f"Message queued from {sender} to {recipient}")

    def subscribe(self, agent_id: str, callback: Callable) -> None:
        """
        Subscribe an agent to receive messages.

        Args:
            agent_id: Agent's identifier
            callback: Callback function to handle received messages
        """
        if agent_id not in self.subscribers:
            self.subscribers[agent_id] = []
        self.subscribers[agent_id].append(callback)
        self.logger.info(f"Agent {agent_id} subscribed to messages")

    def unsubscribe(self, agent_id: str, callback: Callable) -> None:
        """
        Unsubscribe an agent from receiving messages.

        Args:
            agent_id: Agent's identifier
            callback: Callback function to remove
        """
        if agent_id in self.subscribers:
            self.subscribers[agent_id].remove(callback)
            if not self.subscribers[agent_id]:
                del self.subscribers[agent_id]
            self.logger.info(f"Agent {agent_id} unsubscribed from messages")

    def get_queue_size(self) -> int:
        """
        Get the current size of the message queue.

        Returns:
            Number of messages in the queue
        """
        return self.message_queue.qsize()

    def get_subscriber_count(self) -> int:
        """
        Get the total number of subscribers.

        Returns:
            Number of subscribers
        """
        return sum(len(callbacks) for callbacks in self.subscribers.values()) 