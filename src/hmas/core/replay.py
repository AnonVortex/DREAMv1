"""Advanced experience replay module for H-MAS."""

import numpy as np
import torch
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass
from collections import deque
import random
from pathlib import Path
import json
import logging
from datetime import datetime
from .environments import EnvironmentType

@dataclass
class ReplayConfig:
    """Configuration for experience replay."""
    capacity: int = 1000000  # Maximum buffer size
    alpha: float = 0.6  # Priority exponent
    beta: float = 0.4  # Importance sampling exponent
    beta_increment: float = 0.001  # Beta increment per sampling
    epsilon: float = 1e-6  # Small constant to avoid zero priority
    n_step: int = 3  # Steps for N-step returns
    gamma: float = 0.99  # Discount factor
    batch_size: int = 64
    min_samples: int = 1000  # Minimum samples before sampling
    max_episode_length: int = 1000
    hindsight_k: int = 4  # Number of hindsight goals
    save_dir: str = "replay_buffer"

@dataclass
class Experience:
    """Single experience entry."""
    state: Any
    action: Any
    reward: float
    next_state: Any
    done: bool
    info: Dict[str, Any]
    env_type: EnvironmentType
    timestamp: datetime
    episode_id: str
    step: int
    goal: Optional[Any] = None
    achieved: Optional[Any] = None

class SumTree:
    """Binary sum tree for prioritized experience replay."""
    
    def __init__(self, capacity: int):
        """Initialize sum tree."""
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0
        self.write = 0
        
    def _propagate(self, idx: int, change: float) -> None:
        """Propagate changes up the tree."""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        
        if parent != 0:
            self._propagate(parent, change)
            
    def _retrieve(
        self,
        idx: int,
        s: float
    ) -> Tuple[int, float]:
        """Find sample on leaf node."""
        left = 2 * idx + 1
        right = left + 1
        
        if left >= len(self.tree):
            return idx, self.tree[idx]
            
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])
            
    def total(self) -> float:
        """Get total priority."""
        return self.tree[0]
        
    def add(
        self,
        priority: float,
        data: Experience
    ) -> None:
        """Add new data to tree."""
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, priority)
        
        self.write += 1
        if self.write >= self.capacity:
            self.write = 0
            
        if self.n_entries < self.capacity:
            self.n_entries += 1
            
    def update(
        self,
        idx: int,
        priority: float
    ) -> None:
        """Update priority."""
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)
        
    def get(
        self,
        s: float
    ) -> Tuple[int, float, Experience]:
        """Get experience by priority score."""
        idx, priority = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, priority, self.data[data_idx]

class ExperienceReplay:
    """Advanced experience replay buffer."""
    
    def __init__(self, config: ReplayConfig):
        """Initialize experience replay."""
        self.config = config
        self.tree = SumTree(config.capacity)
        self.episode_buffer: Dict[str, List[Experience]] = {}
        self.current_episode_id: Optional[str] = None
        self.beta = config.beta
        self.logger = logging.getLogger("experience_replay")
        
        # Create save directory
        Path(config.save_dir).mkdir(parents=True, exist_ok=True)
        
    def store(self, experience: Experience) -> None:
        """Store new experience."""
        # Calculate initial priority
        max_priority = np.max(self.tree.tree[-self.tree.capacity:])
        if max_priority == 0:
            max_priority = 1.0
            
        # Store in sum tree
        self.tree.add(max_priority, experience)
        
        # Store in episode buffer
        if experience.episode_id not in self.episode_buffer:
            self.episode_buffer[experience.episode_id] = []
        self.episode_buffer[experience.episode_id].append(experience)
        
        # Clean up old episodes if needed
        if len(self.episode_buffer) > 1000:  # Keep last 1000 episodes
            oldest_episode = min(self.episode_buffer.keys())
            del self.episode_buffer[oldest_episode]
            
    def sample(
        self,
        batch_size: Optional[int] = None
    ) -> Tuple[List[Experience], np.ndarray, np.ndarray]:
        """Sample batch of experiences."""
        batch_size = batch_size or self.config.batch_size
        
        if self.tree.n_entries < self.config.min_samples:
            return [], np.array([]), np.array([])
            
        experiences = []
        indices = []
        priorities = []
        
        # Get sampling segment
        segment = self.tree.total() / batch_size
        
        # Increase beta
        self.beta = min(1.0, self.beta + self.config.beta_increment)
        
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            
            idx, priority, experience = self.tree.get(s)
            indices.append(idx)
            priorities.append(priority)
            experiences.append(experience)
            
        # Calculate importance sampling weights
        sampling_probabilities = np.array(priorities) / self.tree.total()
        weights = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        weights = weights / weights.max()
        
        return experiences, indices, weights
        
    def update_priorities(
        self,
        indices: np.ndarray,
        priorities: np.ndarray
    ) -> None:
        """Update priorities for experiences."""
        for idx, priority in zip(indices, priorities):
            priority = (priority + self.config.epsilon) ** self.config.alpha
            self.tree.update(idx, priority)
            
    def sample_episode(
        self,
        episode_id: Optional[str] = None
    ) -> List[Experience]:
        """Sample complete episode."""
        if episode_id is None:
            # Sample random episode
            if not self.episode_buffer:
                return []
            episode_id = random.choice(list(self.episode_buffer.keys()))
            
        return self.episode_buffer.get(episode_id, [])
        
    def generate_hindsight_experiences(
        self,
        episode: List[Experience]
    ) -> List[Experience]:
        """Generate hindsight experiences from episode."""
        if not episode:
            return []
            
        # Get achieved states as potential goals
        achieved_states = [exp.achieved for exp in episode if exp.achieved is not None]
        if not achieved_states:
            return []
            
        # Sample goals for hindsight replay
        sampled_goals = random.sample(
            achieved_states,
            min(self.config.hindsight_k, len(achieved_states))
        )
        
        hindsight_experiences = []
        for goal in sampled_goals:
            for exp in episode:
                # Create new experience with different goal
                hindsight_exp = Experience(
                    state=exp.state,
                    action=exp.action,
                    next_state=exp.next_state,
                    reward=self._compute_reward(exp.achieved, goal),
                    done=exp.done,
                    info=exp.info,
                    env_type=exp.env_type,
                    timestamp=exp.timestamp,
                    episode_id=f"{exp.episode_id}_hindsight",
                    step=exp.step,
                    goal=goal,
                    achieved=exp.achieved
                )
                hindsight_experiences.append(hindsight_exp)
                
        return hindsight_experiences
        
    def _compute_reward(
        self,
        achieved: Any,
        goal: Any
    ) -> float:
        """Compute reward for hindsight experience."""
        # This is a simple example - customize based on your needs
        if isinstance(achieved, (np.ndarray, torch.Tensor)):
            distance = np.linalg.norm(achieved - goal)
            return float(distance < 0.1)  # Binary reward
        return float(achieved == goal)
        
    def sample_n_step_sequence(
        self,
        n: Optional[int] = None
    ) -> List[Experience]:
        """Sample N-step sequence of experiences."""
        n = n or self.config.n_step
        
        # Sample random episode
        episode = self.sample_episode()
        if len(episode) < n:
            return []
            
        # Sample start index
        start_idx = random.randint(0, len(episode) - n)
        return episode[start_idx:start_idx + n]
        
    def compute_n_step_return(
        self,
        experiences: List[Experience],
        gamma: Optional[float] = None
    ) -> float:
        """Compute N-step return for sequence of experiences."""
        gamma = gamma or self.config.gamma
        n_step_return = 0
        
        for i, exp in enumerate(experiences):
            n_step_return += (gamma ** i) * exp.reward
            
        # Add bootstrap value if sequence doesn't end in terminal state
        if experiences and not experiences[-1].done:
            n_step_return += (
                gamma ** len(experiences) *
                self._estimate_value(experiences[-1].next_state)
            )
            
        return n_step_return
        
    def _estimate_value(self, state: Any) -> float:
        """Estimate value of state for bootstrapping."""
        # TODO: Implement value estimation (e.g., using critic network)
        return 0.0
        
    def save(self, path: Optional[str] = None) -> None:
        """Save replay buffer state."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = path or f"{self.config.save_dir}/buffer_{timestamp}.json"
        
        # Convert experiences to serializable format
        buffer_state = {
            "config": self.config.__dict__,
            "experiences": [
                {
                    "priority": float(p),
                    "experience": self._serialize_experience(e)
                }
                for p, e in zip(
                    self.tree.tree[-self.tree.capacity:],
                    self.tree.data
                )
                if e is not None
            ],
            "beta": self.beta,
            "timestamp": timestamp
        }
        
        with open(path, "w") as f:
            json.dump(buffer_state, f, indent=2)
            
    def load(self, path: str) -> None:
        """Load replay buffer state."""
        with open(path, "r") as f:
            buffer_state = json.load(f)
            
        # Update configuration
        self.config = ReplayConfig(**buffer_state["config"])
        self.beta = buffer_state["beta"]
        
        # Rebuild tree and episode buffer
        self.tree = SumTree(self.config.capacity)
        self.episode_buffer.clear()
        
        for item in buffer_state["experiences"]:
            experience = self._deserialize_experience(item["experience"])
            self.store(experience)
            
    def _serialize_experience(self, experience: Experience) -> Dict[str, Any]:
        """Convert experience to serializable format."""
        if experience is None:
            return None
            
        return {
            "state": self._serialize_array(experience.state),
            "action": self._serialize_array(experience.action),
            "reward": float(experience.reward),
            "next_state": self._serialize_array(experience.next_state),
            "done": bool(experience.done),
            "info": experience.info,
            "env_type": experience.env_type.value,
            "timestamp": experience.timestamp.isoformat(),
            "episode_id": experience.episode_id,
            "step": experience.step,
            "goal": self._serialize_array(experience.goal),
            "achieved": self._serialize_array(experience.achieved)
        }
        
    def _deserialize_experience(self, data: Dict[str, Any]) -> Experience:
        """Convert serialized data back to Experience."""
        if data is None:
            return None
            
        return Experience(
            state=self._deserialize_array(data["state"]),
            action=self._deserialize_array(data["action"]),
            reward=data["reward"],
            next_state=self._deserialize_array(data["next_state"]),
            done=data["done"],
            info=data["info"],
            env_type=EnvironmentType(data["env_type"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            episode_id=data["episode_id"],
            step=data["step"],
            goal=self._deserialize_array(data["goal"]),
            achieved=self._deserialize_array(data["achieved"])
        )
        
    def _serialize_array(self, arr: Any) -> Any:
        """Convert numpy array or tensor to serializable format."""
        if arr is None:
            return None
        elif isinstance(arr, np.ndarray):
            return {
                "type": "numpy",
                "data": arr.tolist(),
                "dtype": str(arr.dtype),
                "shape": arr.shape
            }
        elif isinstance(arr, torch.Tensor):
            return {
                "type": "torch",
                "data": arr.cpu().numpy().tolist(),
                "dtype": str(arr.dtype),
                "shape": arr.shape
            }
        return arr
        
    def _deserialize_array(self, data: Any) -> Any:
        """Convert serialized array back to numpy array or tensor."""
        if data is None or not isinstance(data, dict):
            return data
            
        if data["type"] == "numpy":
            return np.array(
                data["data"],
                dtype=np.dtype(data["dtype"])
            ).reshape(data["shape"])
        elif data["type"] == "torch":
            return torch.tensor(
                data["data"],
                dtype=getattr(torch, str(data["dtype"]).split(".")[-1])
            ).reshape(data["shape"])
        return data
        
    def get_statistics(self) -> Dict[str, Any]:
        """Get replay buffer statistics."""
        return {
            "total_experiences": self.tree.n_entries,
            "unique_episodes": len(self.episode_buffer),
            "memory_usage": self._get_memory_usage(),
            "priority_stats": self._get_priority_stats(),
            "temporal_stats": self._get_temporal_stats()
        }
        
    def _get_memory_usage(self) -> Dict[str, float]:
        """Calculate memory usage of buffer."""
        import sys
        
        tree_size = sys.getsizeof(self.tree.tree)
        data_size = sum(
            sys.getsizeof(e) for e in self.tree.data if e is not None
        )
        episode_size = sum(
            sys.getsizeof(e) for episodes in self.episode_buffer.values()
            for e in episodes
        )
        
        return {
            "tree_mb": tree_size / (1024 * 1024),
            "data_mb": data_size / (1024 * 1024),
            "episode_mb": episode_size / (1024 * 1024),
            "total_mb": (tree_size + data_size + episode_size) / (1024 * 1024)
        }
        
    def _get_priority_stats(self) -> Dict[str, float]:
        """Calculate statistics about priorities."""
        priorities = self.tree.tree[-self.tree.capacity:]
        priorities = priorities[priorities > 0]
        
        if len(priorities) == 0:
            return {
                "min": 0.0,
                "max": 0.0,
                "mean": 0.0,
                "std": 0.0
            }
            
        return {
            "min": float(np.min(priorities)),
            "max": float(np.max(priorities)),
            "mean": float(np.mean(priorities)),
            "std": float(np.std(priorities))
        }
        
    def _get_temporal_stats(self) -> Dict[str, Any]:
        """Calculate statistics about temporal distribution."""
        if not self.episode_buffer:
            return {
                "avg_episode_length": 0,
                "max_episode_length": 0,
                "total_episodes": 0
            }
            
        episode_lengths = [
            len(episodes) for episodes in self.episode_buffer.values()
        ]
        
        return {
            "avg_episode_length": float(np.mean(episode_lengths)),
            "max_episode_length": float(np.max(episode_lengths)),
            "total_episodes": len(self.episode_buffer)
        } 