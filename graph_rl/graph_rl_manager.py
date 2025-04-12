"""
Graph-based Reinforcement Learning Manager for handling agent relationships and network optimization.
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Set, Any, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GraphConv
from torch_geometric.data import Data, Batch
import logging
from uuid import UUID
import zlib
import json
from datetime import datetime
from collections import defaultdict

class GraphNodeType(Enum):
    """Types of nodes in the agent relationship graph."""
    AGENT = auto()
    TASK = auto()
    RESOURCE = auto()
    ENVIRONMENT = auto()

class MessageType(Enum):
    """Types of messages that can be exchanged between nodes."""
    STATE_UPDATE = auto()  # Share state information
    ACTION_PROPOSAL = auto()  # Propose an action
    REWARD_SIGNAL = auto()  # Share reward information
    COORDINATION_REQUEST = auto()  # Request coordination
    ROLE_ASSIGNMENT = auto()  # Assign roles
    EMERGENCY = auto()  # High-priority emergency messages

class ProtocolType(Enum):
    """Available communication protocols."""
    STANDARD = auto()  # Regular message passing
    COMPRESSED = auto()  # Compressed messages for large payloads
    PRIORITY = auto()  # High-priority messages
    BROADCAST = auto()  # Messages to all neighbors
    SECURE = auto()  # Encrypted messages

@dataclass
class GraphConfig:
    """Configuration for graph neural networks."""
    node_feature_dim: int = 64
    edge_feature_dim: int = 32
    hidden_dim: int = 128
    num_layers: int = 3
    attention_heads: int = 4
    dropout_rate: float = 0.1
    aggregation_type: str = "mean"  # mean, sum, max

@dataclass
class Message:
    """Structured message for inter-node communication."""
    msg_type: MessageType
    protocol: ProtocolType
    content: torch.Tensor
    metadata: Dict[str, Any]
    priority: float = 1.0
    compression_threshold: int = 1024  # Bytes

    def serialize(self) -> bytes:
        """Serialize message content for transmission."""
        data = {
            "msg_type": self.msg_type.name,
            "protocol": self.protocol.name,
            "content": self.content.cpu().numpy().tobytes(),
            "metadata": self.metadata,
            "priority": self.priority
        }
        serialized = json.dumps(data).encode()
        
        # Compress if exceeds threshold
        if len(serialized) > self.compression_threshold:
            return zlib.compress(serialized)
        return serialized

    @classmethod
    def deserialize(cls, data: bytes) -> 'Message':
        """Deserialize message from bytes."""
        try:
            # Try to decompress
            try:
                data = zlib.decompress(data)
            except zlib.error:
                pass  # Data wasn't compressed
                
            parsed = json.loads(data.decode())
            return cls(
                msg_type=MessageType[parsed["msg_type"]],
                protocol=ProtocolType[parsed["protocol"]],
                content=torch.from_numpy(np.frombuffer(
                    parsed["content"]
                ).reshape(-1)),
                metadata=parsed["metadata"],
                priority=parsed["priority"]
            )
        except Exception as e:
            logging.error(f"Error deserializing message: {str(e)}")
            raise

@dataclass
class CommunicationConfig:
    """Configuration for communication in graph RL."""
    bandwidth_limit: float = 1.0  # Maximum bandwidth per timestep
    latency_penalty: float = 0.1  # Penalty for communication latency
    message_size: int = 64  # Size of message embeddings
    max_receivers: int = 5  # Maximum number of receivers per message
    compression_threshold: int = 1024  # Bytes before compression
    max_retries: int = 3  # Maximum message retry attempts
    timeout: float = 1.0  # Seconds before message timeout

class GraphRelation:
    """Manages relationships between nodes in the graph."""
    def __init__(self, source: int, target: int, relation_type: str, weight: float = 1.0):
        self.source = source
        self.target = target
        self.relation_type = relation_type
        self.weight = weight
        self.features: Optional[torch.Tensor] = None

class GraphRLNetwork(nn.Module):
    """Graph Neural Network for reinforcement learning."""
    def __init__(self, config: GraphConfig):
        super().__init__()
        self.config = config
        
        # Node embedding layers
        self.node_embedding = nn.Linear(config.node_feature_dim, config.hidden_dim)
        
        # Graph convolution layers
        self.conv_layers = nn.ModuleList([
            GATConv(
                in_channels=config.hidden_dim,
                out_channels=config.hidden_dim,
                heads=config.attention_heads,
                dropout=config.dropout_rate
            )
            for _ in range(config.num_layers)
        ])
        
        # Output layers
        self.output_layer = nn.Sequential(
            nn.Linear(config.hidden_dim * config.attention_heads, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_dim, config.node_feature_dim)
        )

    def forward(self, data: Batch) -> torch.Tensor:
        """Forward pass through the network."""
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        # Initial node embeddings
        x = self.node_embedding(x)
        
        # Graph convolution layers
        for conv in self.conv_layers:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.config.dropout_rate, training=self.training)
        
        # Output processing
        return self.output_layer(x)

class GraphRLManager:
    """Manages graph-based reinforcement learning processes."""
    
    def __init__(self, config: GraphConfig):
        """
        Initialize the graph RL manager.
        
        Args:
            config: Configuration for the graph neural network
        """
        self.config = config
        self.network = GraphRLNetwork(config)
        self.optimizer = torch.optim.Adam(self.network.parameters())
        self.node_dict: Dict[int, GraphNodeType] = {}
        self.relations: List[GraphRelation] = []
        self.node_features: Dict[int, torch.Tensor] = {}
        
        # Communication-specific initialization
        self.comm_config = CommunicationConfig()
        self.message_buffer = {}  # Buffer for pending messages
        self.bandwidth_usage = 0.0
        self.communication_stats = {
            "messages_sent": 0,
            "bandwidth_used": 0.0,
            "latency": []
        }
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

    def add_node(self, node_id: int, node_type: GraphNodeType, features: torch.Tensor):
        """Add a node to the graph."""
        if node_id in self.node_dict:
            self.logger.warning(f"Node {node_id} already exists, updating features")
        
        self.node_dict[node_id] = node_type
        self.node_features[node_id] = features

    def add_relation(self, relation: GraphRelation):
        """Add a relation between nodes."""
        if relation.source not in self.node_dict or relation.target not in self.node_dict:
            raise ValueError("Source or target node does not exist")
        
        self.relations.append(relation)

    def get_node_neighbors(self, node_id: int) -> Set[int]:
        """Get neighboring nodes for a given node."""
        if node_id not in self.node_dict:
            raise ValueError(f"Node {node_id} does not exist")
        
        neighbors = set()
        for relation in self.relations:
            if relation.source == node_id:
                neighbors.add(relation.target)
            elif relation.target == node_id:
                neighbors.add(relation.source)
        
        return neighbors

    def build_graph_batch(self, node_ids: List[int]) -> Batch:
        """Build a graph batch for the specified nodes."""
        x = []  # Node features
        edge_index = []  # Edge connectivity
        edge_attr = []  # Edge features
        
        # Create node feature matrix
        for node_id in node_ids:
            x.append(self.node_features[node_id])
        
        # Create edge index and attribute matrices
        node_id_to_idx = {node_id: idx for idx, node_id in enumerate(node_ids)}
        
        for relation in self.relations:
            if relation.source in node_id_to_idx and relation.target in node_id_to_idx:
                source_idx = node_id_to_idx[relation.source]
                target_idx = node_id_to_idx[relation.target]
                
                edge_index.append([source_idx, target_idx])
                if relation.features is not None:
                    edge_attr.append(relation.features)
        
        # Convert to tensors
        x = torch.stack(x)
        edge_index = torch.tensor(edge_index).t().contiguous()
        edge_attr = torch.stack(edge_attr) if edge_attr else None
        
        return Batch.from_data_list([Data(x=x, edge_index=edge_index, edge_attr=edge_attr)])

    async def update_node_embeddings(self, node_ids: List[int]) -> Dict[int, torch.Tensor]:
        """Update embeddings for specified nodes."""
        self.network.train()
        
        try:
            # Build graph batch
            batch = self.build_graph_batch(node_ids)
            
            # Forward pass
            self.optimizer.zero_grad()
            output = self.network(batch)
            
            # Calculate loss (example: reconstruction loss)
            loss = F.mse_loss(output, batch.x)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Update node features
            for idx, node_id in enumerate(node_ids):
                self.node_features[node_id] = output[idx].detach()
            
            return {node_id: self.node_features[node_id] for node_id in node_ids}
            
        except Exception as e:
            self.logger.error(f"Failed to update node embeddings: {str(e)}")
            raise

    def get_node_embedding(self, node_id: int) -> torch.Tensor:
        """Get the current embedding for a node."""
        if node_id not in self.node_features:
            raise ValueError(f"Node {node_id} does not exist")
        
        return self.node_features[node_id]

    def get_subgraph(self, center_node: int, depth: int = 1) -> List[int]:
        """Get nodes within specified depth from center node."""
        if center_node not in self.node_dict:
            raise ValueError(f"Node {center_node} does not exist")
        
        subgraph_nodes = {center_node}
        current_depth = 0
        frontier = {center_node}
        
        while current_depth < depth and frontier:
            next_frontier = set()
            for node in frontier:
                neighbors = self.get_node_neighbors(node)
                next_frontier.update(neighbors - subgraph_nodes)
            subgraph_nodes.update(next_frontier)
            frontier = next_frontier
            current_depth += 1
        
        return list(subgraph_nodes)

    def save_state(self, path: str):
        """Save the graph RL state."""
        state = {
            "network_state": self.network.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "node_dict": self.node_dict,
            "relations": self.relations,
            "node_features": self.node_features
        }
        torch.save(state, path)

    def load_state(self, path: str):
        """Load the graph RL state."""
        try:
            state = torch.load(path)
            self.network.load_state_dict(state["network_state"])
            self.optimizer.load_state_dict(state["optimizer_state"])
            self.node_dict = state["node_dict"]
            self.relations = state["relations"]
            self.node_features = state["node_features"]
        except Exception as e:
            self.logger.error(f"Failed to load state: {str(e)}")
            raise

    def get_metrics(self) -> Dict:
        """Get current metrics for the graph RL system."""
        return {
            "num_nodes": len(self.node_dict),
            "num_relations": len(self.relations),
            "node_type_distribution": {
                node_type.name: sum(1 for t in self.node_dict.values() if t == node_type)
                for node_type in GraphNodeType
            }
        }

    async def send_message(
        self,
        sender_id: int,
        receiver_ids: List[int],
        content: torch.Tensor,
        msg_type: MessageType = MessageType.STATE_UPDATE,
        protocol: Optional[ProtocolType] = None,
        priority: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Send a message through the graph network with protocol selection."""
        try:
            # Select protocol based on message characteristics
            if protocol is None:
                protocol = self._select_protocol(content, priority, receiver_ids)
            
            # Create structured message
            message = Message(
                msg_type=msg_type,
                protocol=protocol,
                content=content,
                metadata=metadata or {},
                priority=priority
            )
            
            # Check bandwidth constraints
            serialized = message.serialize()
            message_size = len(serialized)
            
            if self.bandwidth_usage + message_size > self.comm_config.bandwidth_limit:
                if protocol != ProtocolType.COMPRESSED:
                    # Retry with compression
                    return await self.send_message(
                        sender_id, receiver_ids, content,
                        msg_type, ProtocolType.COMPRESSED,
                        priority, metadata
                    )
                return False
            
            # Update bandwidth usage
            self.bandwidth_usage += message_size
            
            # Create message object for buffer
            buffer_entry = {
                "message": message,
                "sender": sender_id,
                "receivers": receiver_ids[:self.comm_config.max_receivers],
                "timestamp": torch.cuda.Event(enable_timing=True),
                "retries": 0
            }
            
            # Record start time
            buffer_entry["timestamp"].record()
            
            # Add to message buffer
            self.message_buffer[id(buffer_entry)] = buffer_entry
            
            # Update stats
            self.communication_stats["messages_sent"] += 1
            self.communication_stats["bandwidth_used"] += message_size
            
            return True
            
        except Exception as e:
            logging.error(f"Error sending message: {str(e)}")
            return False

    def _select_protocol(
        self,
        content: torch.Tensor,
        priority: float,
        receiver_ids: List[int]
    ) -> ProtocolType:
        """Select appropriate protocol based on message characteristics."""
        content_size = content.numel() * 4  # Size in bytes
        
        if priority > 0.8:
            return ProtocolType.PRIORITY
        elif len(receiver_ids) > self.comm_config.max_receivers // 2:
            return ProtocolType.BROADCAST
        elif content_size > self.comm_config.compression_threshold:
            return ProtocolType.COMPRESSED
        else:
            return ProtocolType.STANDARD

    async def process_messages(self) -> None:
        """Process all pending messages in the buffer with protocol handling."""
        try:
            completed_messages = set()
            
            for msg_id, buffer_entry in self.message_buffer.items():
                message = buffer_entry["message"]
                
                # Calculate message latency
                buffer_entry["timestamp"].synchronize()
                latency = buffer_entry["timestamp"].elapsed_time(
                    torch.cuda.Event(enable_timing=True)
                )
                
                # Handle message based on protocol
                success = await self._handle_message(buffer_entry, latency)
                
                if success or buffer_entry["retries"] >= self.comm_config.max_retries:
                    # Update stats
                    self.communication_stats["latency"].append(latency)
                    completed_messages.add(msg_id)
                else:
                    # Increment retry counter
                    buffer_entry["retries"] += 1
                
            # Clear processed messages
            for msg_id in completed_messages:
                del self.message_buffer[msg_id]
                
            # Reset bandwidth usage
            self.bandwidth_usage = 0.0
            
        except Exception as e:
            logging.error(f"Error processing messages: {str(e)}")

    async def _handle_message(self, buffer_entry: Dict, latency: float) -> bool:
        """Handle message based on its type and protocol."""
        try:
            message = buffer_entry["message"]
            
            # Apply protocol-specific handling
            if message.protocol == ProtocolType.PRIORITY:
                # Process immediately
                return await self._apply_message(buffer_entry)
                
            elif message.protocol == ProtocolType.BROADCAST:
                # Send to all neighbors within range
                neighbors = self.get_node_neighbors(buffer_entry["sender"])
                buffer_entry["receivers"] = list(neighbors)
                return await self._apply_message(buffer_entry)
                
            elif message.protocol == ProtocolType.COMPRESSED:
                # Decompress and process
                return await self._apply_message(buffer_entry)
                
            else:  # STANDARD protocol
                return await self._apply_message(buffer_entry)
                
        except Exception as e:
            logging.error(f"Error handling message: {str(e)}")
            return False

    async def _apply_message(self, buffer_entry: Dict) -> bool:
        """Apply message content to receiving nodes."""
        try:
            message = buffer_entry["message"]
            
            for receiver_id in buffer_entry["receivers"]:
                if receiver_id in self.node_dict:
                    # Get receiver features
                    receiver_features = self.node_features[receiver_id]
                    
                    # Process based on message type
                    if message.msg_type == MessageType.STATE_UPDATE:
                        # Update node state
                        self.node_features[receiver_id] = F.normalize(
                            receiver_features + message.content * message.priority,
                            dim=0
                        )
                        
                    elif message.msg_type == MessageType.ACTION_PROPOSAL:
                        # Store proposed action in node metadata
                        if "proposed_actions" not in self.node_metadata[receiver_id]:
                            self.node_metadata[receiver_id]["proposed_actions"] = []
                        self.node_metadata[receiver_id]["proposed_actions"].append(
                            message.content
                        )
                        
                    elif message.msg_type == MessageType.REWARD_SIGNAL:
                        # Update reward information
                        self.node_metadata[receiver_id]["last_reward"] = message.content
                        
                    elif message.msg_type == MessageType.COORDINATION_REQUEST:
                        # Handle coordination request
                        await self._handle_coordination_request(
                            receiver_id, message.content, message.metadata
                        )
                        
                    elif message.msg_type == MessageType.ROLE_ASSIGNMENT:
                        # Update node role
                        self.node_metadata[receiver_id]["role"] = message.metadata["role"]
                        
                    elif message.msg_type == MessageType.EMERGENCY:
                        # Handle emergency message
                        await self._handle_emergency(
                            receiver_id, message.content, message.metadata
                        )
            
            return True
            
        except Exception as e:
            logging.error(f"Error applying message: {str(e)}")
            return False

    async def _handle_coordination_request(
        self,
        node_id: int,
        content: torch.Tensor,
        metadata: Dict[str, Any]
    ) -> None:
        """Handle coordination request messages."""
        try:
            # Store coordination request
            if "coordination_requests" not in self.node_metadata[node_id]:
                self.node_metadata[node_id]["coordination_requests"] = []
            
            self.node_metadata[node_id]["coordination_requests"].append({
                "content": content,
                "metadata": metadata,
                "timestamp": datetime.now()
            })
            
        except Exception as e:
            logging.error(f"Error handling coordination request: {str(e)}")

    async def _handle_emergency(
        self,
        node_id: int,
        content: torch.Tensor,
        metadata: Dict[str, Any]
    ) -> None:
        """Handle emergency messages."""
        try:
            # Immediately propagate to neighbors
            neighbors = self.get_node_neighbors(node_id)
            if neighbors:
                await self.send_message(
                    sender_id=node_id,
                    receiver_ids=list(neighbors),
                    content=content,
                    msg_type=MessageType.EMERGENCY,
                    protocol=ProtocolType.PRIORITY,
                    priority=1.0,
                    metadata=metadata
                )
            
            # Store emergency state
            self.node_metadata[node_id]["emergency_state"] = {
                "content": content,
                "metadata": metadata,
                "timestamp": datetime.now()
            }
            
        except Exception as e:
            logging.error(f"Error handling emergency message: {str(e)}")

    def get_communication_stats(self) -> Dict[str, Any]:
        """Get current communication statistics."""
        stats = {
            "messages_sent": self.communication_stats["messages_sent"],
            "bandwidth_used": self.communication_stats["bandwidth_used"],
            "avg_latency": np.mean(self.communication_stats["latency"]) if self.communication_stats["latency"] else 0.0,
            "pending_messages": len(self.message_buffer)
        }
        
        # Add protocol-specific stats
        protocol_stats = defaultdict(int)
        message_type_stats = defaultdict(int)
        
        for buffer_entry in self.message_buffer.values():
            message = buffer_entry["message"]
            protocol_stats[message.protocol.name] += 1
            message_type_stats[message.msg_type.name] += 1
        
        stats.update({
            "protocol_usage": dict(protocol_stats),
            "message_types": dict(message_type_stats),
            "retry_rate": sum(
                1 for m in self.message_buffer.values()
                if m["retries"] > 0
            ) / max(1, len(self.message_buffer))
        })
        
        return stats
        
    def optimize_communication(self, metrics: Dict[str, float]) -> None:
        """Optimize communication based on performance metrics.
        
        Args:
            metrics: Dictionary of performance metrics
        """
        try:
            # Adjust max receivers based on bandwidth usage
            if metrics.get("bandwidth_utilization", 0) > 0.9:
                self.comm_config.max_receivers = max(2, self.comm_config.max_receivers - 1)
            elif metrics.get("bandwidth_utilization", 0) < 0.5:
                self.comm_config.max_receivers = min(10, self.comm_config.max_receivers + 1)
                
            # Adjust message size based on latency
            avg_latency = metrics.get("average_latency", 0)
            if avg_latency > 0.1:  # 100ms threshold
                self.comm_config.message_size = max(32, self.comm_config.message_size // 2)
            elif avg_latency < 0.05:  # 50ms threshold
                self.comm_config.message_size = min(256, self.comm_config.message_size * 2)
                
        except Exception as e:
            logging.error(f"Error optimizing communication: {str(e)}")
            
    def reset_communication_stats(self) -> None:
        """Reset all communication statistics."""
        self.communication_stats = {
            "messages_sent": 0,
            "bandwidth_used": 0.0,
            "latency": []
        }
        self.bandwidth_usage = 0.0
        self.message_buffer.clear() 