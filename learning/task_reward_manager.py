import numpy as np
from typing import Dict, List, Any, Tuple
from enum import Enum
from dataclasses import dataclass

from ..environments.tasks_3d import TaskType

@dataclass
class RewardConfig:
    """Configuration for task-specific rewards."""
    base_reward: float = 1.0
    time_penalty: float = -0.01
    collision_penalty: float = -0.5
    completion_bonus: float = 10.0
    exploration_bonus: float = 0.5
    cooperation_bonus: float = 2.0
    efficiency_bonus: float = 1.0

class RewardComponent(Enum):
    """Components that contribute to the final reward."""
    BASE = "base"
    TIME = "time"
    COLLISION = "collision"
    COMPLETION = "completion"
    EXPLORATION = "exploration"
    COOPERATION = "cooperation"
    EFFICIENCY = "efficiency"

class TaskRewardManager:
    """Manages task-specific reward calculations and metrics."""
    
    def __init__(self, task_type: TaskType, config: RewardConfig = None):
        self.task_type = task_type
        self.config = config or RewardConfig()
        self.reward_history: List[Dict[str, float]] = []
        self.explored_areas: Dict[Tuple[int, int, int], float] = {}  # For exploration tasks
        self.agent_cooperation_scores: Dict[str, float] = {}  # For multi-agent tasks
        
    def calculate_reward(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        info: Dict[str, Any]
    ) -> Tuple[float, Dict[str, float]]:
        """Calculate task-specific reward and component breakdown."""
        reward_components = {}
        
        # Base reward
        reward_components[RewardComponent.BASE.value] = self.config.base_reward
        
        # Time penalty
        reward_components[RewardComponent.TIME.value] = self.config.time_penalty
        
        # Collision penalty if applicable
        if info.get("collision", False):
            reward_components[RewardComponent.COLLISION.value] = self.config.collision_penalty
            
        # Task-specific rewards
        if self.task_type == TaskType.EXPLORATION:
            exploration_reward = self._calculate_exploration_reward(state, next_state, info)
            reward_components[RewardComponent.EXPLORATION.value] = exploration_reward
            
        elif self.task_type == TaskType.MULTI_AGENT:
            cooperation_reward = self._calculate_cooperation_reward(state, next_state, info)
            reward_components[RewardComponent.COOPERATION.value] = cooperation_reward
            
        # Completion bonus
        if info.get("task_completed", False):
            reward_components[RewardComponent.COMPLETION.value] = self.config.completion_bonus
            
        # Efficiency bonus based on action smoothness
        efficiency_reward = self._calculate_efficiency_reward(action)
        reward_components[RewardComponent.EFFICIENCY.value] = efficiency_reward
        
        # Calculate total reward
        total_reward = sum(reward_components.values())
        
        # Store reward breakdown
        self.reward_history.append(reward_components)
        
        return total_reward, reward_components
        
    def _calculate_exploration_reward(
        self,
        state: np.ndarray,
        next_state: np.ndarray,
        info: Dict[str, Any]
    ) -> float:
        """Calculate reward for exploration tasks."""
        reward = 0.0
        
        # Get agent position (assuming it's in the state)
        position = tuple(map(int, next_state[:3]))  # x, y, z coordinates
        
        # Reward for discovering new areas
        if position not in self.explored_areas:
            reward += self.config.exploration_bonus
            self.explored_areas[position] = 1.0
        
        # Reward for finding resources
        if info.get("resource_found", False):
            reward += self.config.exploration_bonus * 2
            
        # Reward for area coverage
        coverage = len(self.explored_areas) / info.get("total_area", 1000)
        reward += self.config.exploration_bonus * coverage
        
        return reward
        
    def _calculate_cooperation_reward(
        self,
        state: np.ndarray,
        next_state: np.ndarray,
        info: Dict[str, Any]
    ) -> float:
        """Calculate reward for multi-agent tasks."""
        reward = 0.0
        
        # Get agent ID
        agent_id = info.get("agent_id", "agent_0")
        
        # Reward for maintaining formation
        if info.get("formation_maintained", False):
            reward += self.config.cooperation_bonus
            
        # Reward for successful joint actions
        if info.get("joint_action_success", False):
            reward += self.config.cooperation_bonus * 2
            
        # Update cooperation score
        current_score = self.agent_cooperation_scores.get(agent_id, 0.0)
        self.agent_cooperation_scores[agent_id] = current_score + reward
        
        return reward
        
    def _calculate_efficiency_reward(self, action: np.ndarray) -> float:
        """Calculate reward based on action efficiency."""
        # Penalize extreme actions (encourage smooth movements)
        action_magnitude = np.linalg.norm(action)
        smoothness_factor = np.exp(-action_magnitude)
        return self.config.efficiency_bonus * smoothness_factor
        
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics for the task."""
        metrics = {
            "total_reward": sum(sum(r.values()) for r in self.reward_history),
            "average_reward": np.mean([sum(r.values()) for r in self.reward_history]),
            "reward_components": {
                component.value: np.mean([r.get(component.value, 0.0) for r in self.reward_history])
                for component in RewardComponent
            }
        }
        
        # Task-specific metrics
        if self.task_type == TaskType.EXPLORATION:
            metrics.update({
                "explored_area": len(self.explored_areas),
                "exploration_coverage": len(self.explored_areas) / 1000  # Assuming 1000 is max area
            })
        elif self.task_type == TaskType.MULTI_AGENT:
            metrics.update({
                "agent_cooperation_scores": self.agent_cooperation_scores,
                "average_cooperation": np.mean(list(self.agent_cooperation_scores.values()))
            })
            
        return metrics
        
    def reset(self):
        """Reset the reward manager state."""
        self.reward_history.clear()
        self.explored_areas.clear()
        self.agent_cooperation_scores.clear()
        
    def update_config(self, new_config: RewardConfig):
        """Update reward configuration."""
        self.config = new_config 