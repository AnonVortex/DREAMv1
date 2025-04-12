"""
Inter-Coalition Communication Protocol for Multi-Agent Systems
"""

from enum import Enum, auto
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Any, Union
import numpy as np
import logging
from datetime import datetime
import asyncio
import json
import uuid

from .coalition_manager import Coalition, CoalitionStatus

logger = logging.getLogger(__name__)

class MessagePriority(Enum):
    """Priority levels for coalition messages."""
    LOW = auto()
    MEDIUM = auto()
    HIGH = auto()
    CRITICAL = auto()

class MessageType(Enum):
    """Types of messages exchanged between coalitions."""
    TASK_PROPOSAL = auto()          # Propose task collaboration
    RESOURCE_REQUEST = auto()       # Request resources from other coalitions
    KNOWLEDGE_SHARE = auto()        # Share learned knowledge/experience
    STATUS_UPDATE = auto()          # Update on coalition status
    COORDINATION_REQUEST = auto()    # Request for coordination
    NEGOTIATION_OFFER = auto()      # Negotiation for resources/tasks
    PERFORMANCE_REPORT = auto()      # Share performance metrics
    ALERT = auto()                  # Critical information/warning

@dataclass
class CommunicationConfig:
    """Configuration for coalition communication."""
    bandwidth_limit: float = 1000.0  # Maximum bandwidth per second
    latency_threshold: float = 0.1   # Maximum acceptable latency
    retry_limit: int = 3             # Maximum message retry attempts
    buffer_size: int = 1000          # Message buffer size
    compression_threshold: int = 1024 # Size threshold for message compression

@dataclass
class Message:
    """Structure for coalition messages."""
    id: str
    sender_id: str
    receiver_id: str
    msg_type: MessageType
    priority: MessagePriority
    content: Dict[str, Any]
    timestamp: float
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary format."""
        return {
            "id": self.id,
            "sender_id": self.sender_id,
            "receiver_id": self.receiver_id,
            "msg_type": self.msg_type.name,
            "priority": self.priority.name,
            "content": self.content,
            "timestamp": self.timestamp,
            "metadata": self.metadata or {}
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """Create message from dictionary format."""
        return cls(
            id=data["id"],
            sender_id=data["sender_id"],
            receiver_id=data["receiver_id"],
            msg_type=MessageType[data["msg_type"]],
            priority=MessagePriority[data["priority"]],
            content=data["content"],
            timestamp=data["timestamp"],
            metadata=data.get("metadata", {})
        )

class CoalitionCommunicationManager:
    """Manages communication between coalitions."""
    
    def __init__(
        self,
        config: Optional[CommunicationConfig] = None
    ):
        self.logger = logging.getLogger(__name__)
        self.config = config or CommunicationConfig()
        
        # Communication state
        self.message_buffer: Dict[str, List[Message]] = {}
        self.pending_responses: Dict[str, asyncio.Future] = {}
        self.bandwidth_usage: Dict[str, float] = {}
        self.communication_stats: Dict[str, Dict[str, float]] = {}
        
        # Performance tracking
        self.message_history: List[Dict[str, Any]] = []
        self.latency_metrics: Dict[str, List[float]] = {}
        self.retry_counts: Dict[str, int] = {}
        
    async def send_message(
        self,
        sender_coalition: Coalition,
        receiver_coalition: Coalition,
        msg_type: MessageType,
        content: Dict[str, Any],
        priority: MessagePriority = MessagePriority.MEDIUM,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Send a message from one coalition to another."""
        try:
            # Create message
            message = Message(
                id=str(uuid.uuid4()),
                sender_id=sender_coalition.id,
                receiver_id=receiver_coalition.id,
                msg_type=msg_type,
                priority=priority,
                content=content,
                timestamp=datetime.now().timestamp(),
                metadata=metadata
            )
            
            # Check bandwidth usage
            if not self._check_bandwidth(message):
                self.logger.warning(f"Bandwidth limit exceeded for message {message.id}")
                return None
            
            # Add to buffer
            if receiver_coalition.id not in self.message_buffer:
                self.message_buffer[receiver_coalition.id] = []
            self.message_buffer[receiver_coalition.id].append(message)
            
            # Update stats
            self._update_communication_stats(message)
            
            # Process message based on type
            await self._process_message(message)
            
            return message.id
            
        except Exception as e:
            self.logger.error(f"Error sending message: {str(e)}")
            return None
    
    async def _process_message(self, message: Message) -> None:
        """Process received message based on type."""
        try:
            if message.msg_type == MessageType.TASK_PROPOSAL:
                await self._handle_task_proposal(message)
            elif message.msg_type == MessageType.RESOURCE_REQUEST:
                await self._handle_resource_request(message)
            elif message.msg_type == MessageType.KNOWLEDGE_SHARE:
                await self._handle_knowledge_share(message)
            elif message.msg_type == MessageType.COORDINATION_REQUEST:
                await self._handle_coordination_request(message)
            elif message.msg_type == MessageType.NEGOTIATION_OFFER:
                await self._handle_negotiation(message)
            elif message.msg_type == MessageType.PERFORMANCE_REPORT:
                await self._handle_performance_report(message)
            elif message.msg_type == MessageType.ALERT:
                await self._handle_alert(message)
            
        except Exception as e:
            self.logger.error(f"Error processing message {message.id}: {str(e)}")
    
    async def _handle_task_proposal(self, message: Message) -> None:
        """Handle task collaboration proposal."""
        try:
            task_info = message.content.get("task_info", {})
            required_resources = task_info.get("required_resources", {})
            
            # Check resource availability
            available_resources = self._get_available_resources(message.receiver_id)
            can_collaborate = all(
                available_resources.get(resource, 0) >= amount
                for resource, amount in required_resources.items()
            )
            
            # Prepare response
            response_content = {
                "can_collaborate": can_collaborate,
                "available_resources": available_resources,
                "proposed_contribution": self._calculate_contribution(
                    task_info,
                    available_resources
                )
            }
            
            # Send response
            await self.send_message(
                sender_coalition=message.receiver_id,
                receiver_coalition=message.sender_id,
                msg_type=MessageType.NEGOTIATION_OFFER,
                content=response_content,
                priority=message.priority,
                metadata={"in_response_to": message.id}
            )
            
        except Exception as e:
            self.logger.error(f"Error handling task proposal: {str(e)}")
    
    async def _handle_resource_request(self, message: Message) -> None:
        """Handle resource sharing request."""
        try:
            requested_resources = message.content.get("requested_resources", {})
            available_resources = self._get_available_resources(message.receiver_id)
            
            # Calculate shareable resources
            shareable_resources = {}
            for resource, amount in requested_resources.items():
                available = available_resources.get(resource, 0)
                shareable = min(amount, available * 0.5)  # Share up to 50% of available
                if shareable > 0:
                    shareable_resources[resource] = shareable
            
            # Prepare response
            response_content = {
                "shareable_resources": shareable_resources,
                "sharing_conditions": self._get_sharing_conditions(message)
            }
            
            # Send response
            await self.send_message(
                sender_coalition=message.receiver_id,
                receiver_coalition=message.sender_id,
                msg_type=MessageType.NEGOTIATION_OFFER,
                content=response_content,
                priority=message.priority,
                metadata={"in_response_to": message.id}
            )
            
        except Exception as e:
            self.logger.error(f"Error handling resource request: {str(e)}")
    
    async def _handle_knowledge_share(self, message: Message) -> None:
        """Handle knowledge sharing message."""
        try:
            knowledge_data = message.content.get("knowledge_data", {})
            
            # Validate and process knowledge
            if self._validate_knowledge(knowledge_data):
                # Update local knowledge base
                self._update_knowledge_base(
                    message.receiver_id,
                    knowledge_data
                )
                
                # Send acknowledgment
                await self.send_message(
                    sender_coalition=message.receiver_id,
                    receiver_coalition=message.sender_id,
                    msg_type=MessageType.STATUS_UPDATE,
                    content={"status": "knowledge_received"},
                    priority=MessagePriority.LOW,
                    metadata={"in_response_to": message.id}
                )
            
        except Exception as e:
            self.logger.error(f"Error handling knowledge share: {str(e)}")
    
    async def _handle_coordination_request(self, message: Message) -> None:
        """Handle coordination request."""
        try:
            coordination_type = message.content.get("coordination_type")
            coordination_params = message.content.get("coordination_params", {})
            
            # Process coordination request
            coordination_response = self._process_coordination(
                coordination_type,
                coordination_params,
                message.receiver_id
            )
            
            # Send response
            await self.send_message(
                sender_coalition=message.receiver_id,
                receiver_coalition=message.sender_id,
                msg_type=MessageType.STATUS_UPDATE,
                content=coordination_response,
                priority=message.priority,
                metadata={"in_response_to": message.id}
            )
            
        except Exception as e:
            self.logger.error(f"Error handling coordination request: {str(e)}")
    
    async def _handle_negotiation(self, message: Message) -> None:
        """Handle negotiation message."""
        try:
            offer = message.content.get("offer", {})
            negotiation_type = message.content.get("negotiation_type")
            
            # Evaluate offer
            evaluation = self._evaluate_offer(
                offer,
                negotiation_type,
                message.receiver_id
            )
            
            # Prepare counter-offer or acceptance
            response_content = {
                "decision": evaluation["decision"],
                "counter_offer": evaluation.get("counter_offer"),
                "terms": evaluation.get("terms")
            }
            
            # Send response
            await self.send_message(
                sender_coalition=message.receiver_id,
                receiver_coalition=message.sender_id,
                msg_type=MessageType.NEGOTIATION_OFFER,
                content=response_content,
                priority=message.priority,
                metadata={"in_response_to": message.id}
            )
            
        except Exception as e:
            self.logger.error(f"Error handling negotiation: {str(e)}")
    
    def _check_bandwidth(self, message: Message) -> bool:
        """Check if sending message would exceed bandwidth limits."""
        try:
            message_size = len(json.dumps(message.to_dict()))
            
            # Get current bandwidth usage
            current_usage = self.bandwidth_usage.get(message.sender_id, 0)
            
            # Check against limit
            if current_usage + message_size > self.config.bandwidth_limit:
                return False
            
            # Update bandwidth usage
            self.bandwidth_usage[message.sender_id] = current_usage + message_size
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking bandwidth: {str(e)}")
            return False
    
    def _update_communication_stats(self, message: Message) -> None:
        """Update communication statistics."""
        try:
            coalition_pair = f"{message.sender_id}-{message.receiver_id}"
            
            if coalition_pair not in self.communication_stats:
                self.communication_stats[coalition_pair] = {
                    "messages_sent": 0,
                    "bytes_transferred": 0,
                    "avg_latency": 0.0,
                    "success_rate": 1.0
                }
            
            stats = self.communication_stats[coalition_pair]
            message_size = len(json.dumps(message.to_dict()))
            
            # Update stats
            stats["messages_sent"] += 1
            stats["bytes_transferred"] += message_size
            
            # Add to history
            self.message_history.append(message.to_dict())
            
        except Exception as e:
            self.logger.error(f"Error updating communication stats: {str(e)}")
    
    def get_communication_metrics(
        self,
        coalition_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get communication metrics for a coalition or all coalitions."""
        try:
            if coalition_id:
                # Get metrics for specific coalition
                metrics = {
                    pair: stats
                    for pair, stats in self.communication_stats.items()
                    if coalition_id in pair
                }
            else:
                # Get all metrics
                metrics = self.communication_stats.copy()
            
            # Calculate aggregate metrics
            total_messages = sum(
                stats["messages_sent"]
                for stats in metrics.values()
            )
            total_bytes = sum(
                stats["bytes_transferred"]
                for stats in metrics.values()
            )
            avg_latency = np.mean([
                stats["avg_latency"]
                for stats in metrics.values()
            ])
            
            return {
                "total_messages": total_messages,
                "total_bytes_transferred": total_bytes,
                "average_latency": avg_latency,
                "detailed_stats": metrics
            }
            
        except Exception as e:
            self.logger.error(f"Error getting communication metrics: {str(e)}")
            return {}
    
    def optimize_communication(
        self,
        coalition_id: str,
        optimization_params: Optional[Dict[str, Any]] = None
    ) -> None:
        """Optimize communication parameters for a coalition."""
        try:
            params = optimization_params or {}
            
            # Get current metrics
            metrics = self.get_communication_metrics(coalition_id)
            
            # Adjust bandwidth allocation
            if metrics["total_bytes_transferred"] > self.config.bandwidth_limit * 0.8:
                # Implement message prioritization
                self._prioritize_messages(coalition_id)
            
            # Adjust retry limits based on success rate
            coalition_stats = metrics["detailed_stats"]
            for pair, stats in coalition_stats.items():
                if stats["success_rate"] < 0.8:
                    self.retry_counts[pair] = min(
                        self.retry_counts.get(pair, 0) + 1,
                        self.config.retry_limit
                    )
            
            # Update configuration if needed
            if params.get("update_config"):
                self._update_communication_config(
                    coalition_id,
                    metrics,
                    params
                )
            
        except Exception as e:
            self.logger.error(f"Error optimizing communication: {str(e)}")
    
    def _prioritize_messages(self, coalition_id: str) -> None:
        """Prioritize messages for a coalition based on importance."""
        try:
            if coalition_id in self.message_buffer:
                # Sort messages by priority
                self.message_buffer[coalition_id].sort(
                    key=lambda m: (
                        m.priority.value,
                        -m.timestamp
                    ),
                    reverse=True
                )
                
                # Trim buffer if needed
                if len(self.message_buffer[coalition_id]) > self.config.buffer_size:
                    self.message_buffer[coalition_id] = self.message_buffer[coalition_id][:self.config.buffer_size]
            
        except Exception as e:
            self.logger.error(f"Error prioritizing messages: {str(e)}")
    
    def _update_communication_config(
        self,
        coalition_id: str,
        metrics: Dict[str, Any],
        params: Dict[str, Any]
    ) -> None:
        """Update communication configuration based on metrics."""
        try:
            # Adjust bandwidth limit
            if metrics["total_bytes_transferred"] > self.config.bandwidth_limit * 0.9:
                self.config.bandwidth_limit *= 1.2
            
            # Adjust latency threshold
            avg_latency = metrics["average_latency"]
            if avg_latency > self.config.latency_threshold:
                self.config.latency_threshold = min(
                    avg_latency * 1.1,
                    params.get("max_latency", 1.0)
                )
            
            # Adjust buffer size
            messages_in_buffer = len(self.message_buffer.get(coalition_id, []))
            if messages_in_buffer > self.config.buffer_size * 0.9:
                self.config.buffer_size = int(self.config.buffer_size * 1.2)
            
        except Exception as e:
            self.logger.error(f"Error updating communication config: {str(e)}") 