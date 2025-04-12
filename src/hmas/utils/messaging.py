"""Messaging utilities for H-MAS."""

import asyncio
from typing import Dict, Optional, Any, Callable, Awaitable, List
import json
import aio_pika
from uuid import UUID
import structlog
from datetime import datetime
import zlib
from collections import defaultdict

from hmas.config import settings

logger = structlog.get_logger(__name__)

MessageHandler = Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]]

class MessageBroker:
    """Message broker for agent communication."""
    
    def __init__(
        self,
        batch_size: int = 10,
        batch_timeout: float = 0.1,
        compression_threshold: int = 1024
    ):
        """Initialize message broker."""
        self._connection: Optional[aio_pika.Connection] = None
        self._channel: Optional[aio_pika.Channel] = None
        self._exchange: Optional[aio_pika.Exchange] = None
        self._handlers: Dict[UUID, MessageHandler] = {}
        self._queues: Dict[UUID, aio_pika.Queue] = {}
        self._message_batches: Dict[UUID, List[Dict[str, Any]]] = defaultdict(list)
        self._batch_size = batch_size
        self._batch_timeout = batch_timeout
        self._compression_threshold = compression_threshold
        self._batch_tasks = {}
        
    async def connect(self) -> None:
        """Establish connection to message broker."""
        try:
            self._connection = await aio_pika.connect_robust(
                host=settings.MESSAGE_BROKER_HOST,
                port=settings.MESSAGE_BROKER_PORT,
                login=settings.MESSAGE_BROKER_USER,
                password=settings.MESSAGE_BROKER_PASSWORD,
                virtualhost=settings.MESSAGE_BROKER_VHOST
            )
            
            self._channel = await self._connection.channel()
            self._exchange = await self._channel.declare_exchange(
                "hmas",
                aio_pika.ExchangeType.DIRECT,
                durable=True
            )
            
            logger.info("Connected to message broker")
            
        except Exception as e:
            logger.error("Failed to connect to message broker", error=str(e))
            raise
            
    async def disconnect(self) -> None:
        """Close message broker connection."""
        if self._connection:
            await self._connection.close()
            self._connection = None
            self._channel = None
            self._exchange = None
            logger.info("Disconnected from message broker")
            
    async def register_handler(
        self,
        agent_id: UUID,
        handler: MessageHandler
    ) -> None:
        """Register message handler for an agent.
        
        Args:
            agent_id: Agent UUID
            handler: Message handler function
        """
        if not self._channel or not self._exchange:
            raise RuntimeError("Not connected to message broker")
            
        queue_name = f"agent_{agent_id}"
        queue = await self._channel.declare_queue(
            queue_name,
            durable=True,
            auto_delete=True
        )
        
        await queue.bind(self._exchange, routing_key=str(agent_id))
        self._handlers[agent_id] = handler
        self._queues[agent_id] = queue
        
        # Start consuming messages
        await queue.consume(self._message_handler(agent_id))
        logger.info("Registered message handler", agent_id=str(agent_id))
        
    async def unregister_handler(self, agent_id: UUID) -> None:
        """Unregister message handler for an agent.
        
        Args:
            agent_id: Agent UUID
        """
        if agent_id in self._queues:
            await self._queues[agent_id].delete()
            del self._queues[agent_id]
            del self._handlers[agent_id]
            logger.info("Unregistered message handler", agent_id=str(agent_id))
            
    def _compress_message(self, message: Dict[str, Any]) -> bytes:
        """Compress message if it exceeds threshold."""
        message_bytes = json.dumps(message).encode()
        if len(message_bytes) > self._compression_threshold:
            return zlib.compress(message_bytes)
        return message_bytes

    def _decompress_message(self, message_bytes: bytes) -> Dict[str, Any]:
        """Decompress message if compressed."""
        try:
            return json.loads(zlib.decompress(message_bytes).decode())
        except zlib.error:
            return json.loads(message_bytes.decode())

    async def _process_batch(self, to_agent: UUID) -> None:
        """Process batched messages."""
        while True:
            if len(self._message_batches[to_agent]) >= self._batch_size:
                batch = self._message_batches[to_agent][:self._batch_size]
                self._message_batches[to_agent] = self._message_batches[to_agent][self._batch_size:]
                await self._send_batch(to_agent, batch)
            await asyncio.sleep(self._batch_timeout)

    async def _send_batch(self, to_agent: UUID, batch: List[Dict[str, Any]]) -> None:
        """Send a batch of messages."""
        if not self._exchange:
            raise RuntimeError("Not connected to message broker")

        batch_with_metadata = {
            "type": "batch",
            "messages": batch,
            "timestamp": datetime.utcnow().isoformat(),
            "size": len(batch)
        }

        compressed_batch = self._compress_message(batch_with_metadata)
        
        await self._exchange.publish(
            aio_pika.Message(
                body=compressed_batch,
                delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
                headers={"compressed": len(compressed_batch) != len(batch_with_metadata)}
            ),
            routing_key=str(to_agent)
        )

        logger.debug(f"Sent batch of {len(batch)} messages to {to_agent}")

    async def send_message(
        self,
        from_agent: UUID,
        to_agent: UUID,
        message: Dict[str, Any]
    ) -> None:
        """Send message from one agent to another."""
        if not self._exchange:
            raise RuntimeError("Not connected to message broker")

        message_with_metadata = {
            "from": str(from_agent),
            "to": str(to_agent),
            "timestamp": datetime.utcnow().isoformat(),
            "content": message
        }

        # Add to batch
        self._message_batches[to_agent].append(message_with_metadata)
        
        # Start batch processing task if not already running
        if to_agent not in self._batch_tasks:
            self._batch_tasks[to_agent] = asyncio.create_task(self._process_batch(to_agent))

        logger.debug(f"Queued message from {from_agent} to {to_agent}")

    def _message_handler(self, agent_id: UUID) -> Callable:
        """Create message handler for an agent."""
        async def handler(message: aio_pika.IncomingMessage) -> None:
            async with message.process():
                try:
                    # Decompress and parse message
                    content = self._decompress_message(message.body)
                    
                    if content.get("type") == "batch":
                        # Process batch of messages
                        for msg in content["messages"]:
                            if agent_id in self._handlers:
                                response = await self._handlers[agent_id](msg["content"])
                                if response and "from" in msg:
                                    await self.send_message(
                                        from_agent=agent_id,
                                        to_agent=UUID(msg["from"]),
                                        message=response
                                    )
                    else:
                        # Process single message
                        if agent_id in self._handlers:
                            response = await self._handlers[agent_id](content["content"])
                            if response and "from" in content:
                                await self.send_message(
                                    from_agent=agent_id,
                                    to_agent=UUID(content["from"]),
                                    message=response
                                )
                            
                except Exception as e:
                    logger.error(f"Error processing message: {str(e)}", agent_id=str(agent_id))
                    
        return handler
        
    @property
    def is_connected(self) -> bool:
        """Check if connected to message broker.
        
        Returns:
            bool: True if connected
        """
        return bool(self._connection and not self._connection.is_closed) 