import os
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from ..memory.memory_main import Memory, MemoryType, ExperienceType

logger = logging.getLogger(__name__)

class ExperienceBuffer:
    """Local experience buffer for immediate learning."""
    def __init__(self, capacity: int = 1000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        
    def push(self, experience: Dict[str, Any]):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = experience
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size: int) -> List[Dict[str, Any]]:
        return np.random.choice(self.buffer, batch_size).tolist()
        
    def __len__(self):
        return len(self.buffer)

class LearningNetwork(nn.Module):
    """Neural network for learning agent behaviors."""
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class LearningAgent:
    """Main learning agent that integrates with the memory system."""
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int = 256,
        learning_rate: float = 0.001,
        memory_batch_size: int = 32,
        local_buffer_size: int = 1000,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = torch.device(device)
        self.network = LearningNetwork(input_size, hidden_size, output_size).to(self.device)
        self.target_network = LearningNetwork(input_size, hidden_size, output_size).to(self.device)
        self.target_network.load_state_dict(self.network.state_dict())
        
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        self.memory_batch_size = memory_batch_size
        self.local_buffer = ExperienceBuffer(local_buffer_size)
        
        # Training metrics
        self.writer = SummaryWriter(log_dir=os.path.join("logs", "learning_agent"))
        self.training_steps = 0
        
    async def store_experience(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        memory_client: Any
    ):
        """Store experience in both local buffer and memory system."""
        # Store in local buffer
        experience = {
            "state": state,
            "action": action,
            "reward": reward,
            "next_state": next_state,
            "done": done
        }
        self.local_buffer.push(experience)
        
        # Store in memory system
        memory = Memory(
            type=MemoryType.EPISODIC,
            experience_type=ExperienceType.SUCCESS if reward > 0 else ExperienceType.FAILURE,
            content={
                "state": state.tolist(),
                "action": action.tolist(),
                "reward": reward,
                "next_state": next_state.tolist(),
                "done": done
            },
            metadata={
                "timestamp": datetime.now().isoformat(),
                "reward_threshold": 0.5  # Configurable threshold
            },
            importance_score=min(1.0, abs(reward))  # Use reward magnitude as importance
        )
        await memory_client.store_memory(memory)
        
    async def learn_from_experiences(
        self,
        memory_client: Any,
        gamma: float = 0.99,
        local_batch_size: int = 32
    ):
        """Learn from both local and memory system experiences."""
        if len(self.local_buffer) < local_batch_size:
            return
            
        # Sample from local buffer
        local_experiences = self.local_buffer.sample(local_batch_size)
        local_loss = self._compute_loss(local_experiences, gamma)
        
        # Get experiences from memory system
        memory_response = await memory_client.get_replay_batch(batch_size=self.memory_batch_size)
        if memory_response and "memories" in memory_response:
            memory_experiences = [
                {
                    "state": np.array(m["content"]["state"]),
                    "action": np.array(m["content"]["action"]),
                    "reward": m["content"]["reward"],
                    "next_state": np.array(m["content"]["next_state"]),
                    "done": m["content"]["done"]
                }
                for m in memory_response["memories"]
            ]
            memory_loss = self._compute_loss(memory_experiences, gamma)
        else:
            memory_loss = 0
            
        # Combined learning step
        total_loss = local_loss + memory_loss
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        # Update metrics
        self.training_steps += 1
        self.writer.add_scalar("Loss/total", total_loss.item(), self.training_steps)
        self.writer.add_scalar("Loss/local", local_loss.item(), self.training_steps)
        self.writer.add_scalar("Loss/memory", memory_loss.item(), self.training_steps)
        
        # Periodically update target network
        if self.training_steps % 100 == 0:
            self.target_network.load_state_dict(self.network.state_dict())
            
    def _compute_loss(self, experiences: List[Dict[str, Any]], gamma: float) -> torch.Tensor:
        """Compute loss for a batch of experiences."""
        states = torch.FloatTensor([e["state"] for e in experiences]).to(self.device)
        actions = torch.FloatTensor([e["action"] for e in experiences]).to(self.device)
        rewards = torch.FloatTensor([e["reward"] for e in experiences]).to(self.device)
        next_states = torch.FloatTensor([e["next_state"] for e in experiences]).to(self.device)
        dones = torch.FloatTensor([e["done"] for e in experiences]).to(self.device)
        
        # Get current Q values
        current_q_values = self.network(states)
        
        # Get next Q values from target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states)
            max_next_q_values = next_q_values.max(1)[0]
            target_q_values = rewards + gamma * max_next_q_values * (1 - dones)
            
        # Compute loss
        loss = nn.MSELoss()(current_q_values, target_q_values.unsqueeze(1))
        return loss
        
    async def get_similar_experiences(
        self,
        state: np.ndarray,
        memory_client: Any,
        k: int = 5
    ) -> List[Dict[str, Any]]:
        """Retrieve similar experiences from memory system."""
        state_embedding = state.flatten().tolist()  # Simple embedding strategy
        response = await memory_client.find_similar_memories(
            embedding=state_embedding,
            k=k
        )
        
        if response and "memories" in response:
            return [m["memory"]["content"] for m in response["memories"]]
        return []
        
    def save_model(self, path: str):
        """Save model state."""
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_steps': self.training_steps
        }, path)
        
    def load_model(self, path: str):
        """Load model state."""
        checkpoint = torch.load(path)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_steps = checkpoint['training_steps'] 