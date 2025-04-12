"""Base Environment implementation for H-MAS.

This module defines the core environment interface and functionality for the H-MAS system.
Environments provide the interface for agents to interact with and learn from.
"""

from typing import Dict, List, Any, Optional, Tuple, Union, Set
from dataclasses import dataclass
import numpy as np
import gym
from gym import spaces
import logging
from pathlib import Path
import json
from datetime import datetime
from uuid import UUID, uuid4
import asyncio

@dataclass
class EnvironmentConfig:
    """Configuration for environment."""
    name: str
    observation_space: Dict[str, Any]
    action_space: Dict[str, Any]
    max_steps: int = 1000
    reward_scale: float = 1.0
    time_limit: Optional[int] = None
    save_dir: str = "environment_data"
    record_episodes: bool = False

class TaskGenerator:
    """Generator for environment tasks."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize task generator."""
        self.config = config
        self.task_history: List[Dict[str, Any]] = []
        self.current_difficulty = 0.0
        
    async def generate_task(
        self,
        difficulty: Optional[float] = None,
        constraints: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate a new task."""
        if difficulty is None:
            difficulty = self.current_difficulty
            
        # Generate task parameters
        params = self._generate_parameters(difficulty, constraints)
        
        # Create task specification
        task = {
            "id": str(uuid4()),
            "difficulty": difficulty,
            "parameters": params,
            "constraints": constraints or {},
            "timestamp": datetime.now().isoformat()
        }
        
        # Store in history
        self.task_history.append(task)
        
        return task
        
    def _generate_parameters(
        self,
        difficulty: float,
        constraints: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate task parameters based on difficulty and constraints."""
        params = {}
        
        # Scale parameter complexity with difficulty
        num_objectives = max(1, int(difficulty * 3))  # 1-3 objectives
        num_constraints = max(0, int(difficulty * 5))  # 0-5 constraints
        
        # Generate objectives
        params["objectives"] = [
            self._generate_objective(difficulty)
            for _ in range(num_objectives)
        ]
        
        # Generate additional constraints
        if constraints:
            params["constraints"] = constraints
        else:
            params["constraints"] = [
                self._generate_constraint(difficulty)
                for _ in range(num_constraints)
            ]
            
        return params
        
    def _generate_objective(self, difficulty: float) -> Dict[str, Any]:
        """Generate a single objective."""
        objective_types = ["maximize", "minimize", "achieve", "maintain"]
        
        return {
            "type": np.random.choice(objective_types),
            "target": np.random.uniform(0, 1) * difficulty,
            "weight": np.random.uniform(0.5, 1.0)
        }
        
    def _generate_constraint(self, difficulty: float) -> Dict[str, Any]:
        """Generate a single constraint."""
        constraint_types = ["range", "threshold", "temporal", "resource"]
        
        return {
            "type": np.random.choice(constraint_types),
            "value": np.random.uniform(0, 1) * difficulty,
            "tolerance": np.random.uniform(0, 0.2) * (1 - difficulty)
        }

class RewardShaper:
    """Shapes rewards based on task objectives and constraints."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize reward shaper."""
        self.config = config
        self.reward_history: List[Dict[str, Any]] = []
        
    def shape_reward(
        self,
        base_reward: float,
        state: Dict[str, Any],
        action: Dict[str, Any],
        task: Dict[str, Any]
    ) -> float:
        """Shape the reward based on objectives and constraints."""
        shaped_reward = base_reward * self.config.get("base_scale", 1.0)
        
        # Apply objective-based shaping
        if "objectives" in task["parameters"]:
            for objective in task["parameters"]["objectives"]:
                shaped_reward += self._evaluate_objective(
                    objective,
                    state,
                    action
                )
                
        # Apply constraint penalties
        if "constraints" in task["parameters"]:
            for constraint in task["parameters"]["constraints"]:
                shaped_reward += self._evaluate_constraint(
                    constraint,
                    state,
                    action
                )
                
        # Record reward
        self.reward_history.append({
            "timestamp": datetime.now().isoformat(),
            "base_reward": base_reward,
            "shaped_reward": shaped_reward,
            "state": state,
            "action": action,
            "task": task
        })
        
        return shaped_reward
        
    def _evaluate_objective(
        self,
        objective: Dict[str, Any],
        state: Dict[str, Any],
        action: Dict[str, Any]
    ) -> float:
        """Evaluate reward contribution from an objective."""
        if objective["type"] == "maximize":
            return objective["weight"] * (
                state.get(objective.get("key", "value"), 0) -
                objective["target"]
            )
        elif objective["type"] == "minimize":
            return objective["weight"] * (
                objective["target"] -
                state.get(objective.get("key", "value"), 0)
            )
        elif objective["type"] == "achieve":
            return objective["weight"] * (
                1.0 if state.get(objective.get("key", "value"), 0) >= objective["target"]
                else -0.1
            )
        elif objective["type"] == "maintain":
            return objective["weight"] * (
                1.0 if abs(
                    state.get(objective.get("key", "value"), 0) -
                    objective["target"]
                ) <= objective.get("tolerance", 0.1)
                else -0.2
            )
        return 0.0
        
    def _evaluate_constraint(
        self,
        constraint: Dict[str, Any],
        state: Dict[str, Any],
        action: Dict[str, Any]
    ) -> float:
        """Evaluate penalty from a constraint."""
        if constraint["type"] == "range":
            value = state.get(constraint.get("key", "value"), 0)
            if value < constraint["min"] or value > constraint["max"]:
                return -constraint.get("penalty", 1.0)
        elif constraint["type"] == "threshold":
            value = state.get(constraint.get("key", "value"), 0)
            if value > constraint["value"]:
                return -constraint.get("penalty", 1.0)
        elif constraint["type"] == "temporal":
            if state.get("time_step", 0) > constraint["value"]:
                return -constraint.get("penalty", 1.0)
        elif constraint["type"] == "resource":
            usage = state.get(constraint.get("key", "resource"), 0)
            if usage > constraint["value"]:
                return -constraint.get("penalty", 1.0)
        return 0.0

class BaseEnvironment:
    """Base environment class for H-MAS."""
    
    def __init__(self, config: EnvironmentConfig):
        """Initialize environment."""
        self.config = config
        self.logger = logging.getLogger(f"env_{config.name}")
        
        # Create save directory
        Path(config.save_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize spaces
        self.observation_space = self._create_space(config.observation_space)
        self.action_space = self._create_space(config.action_space)
        
        # Initialize components
        self.task_generator = TaskGenerator({"difficulty_range": (0.0, 1.0)})
        self.reward_shaper = RewardShaper({"base_scale": config.reward_scale})
        
        # Initialize state
        self.current_task: Optional[Dict[str, Any]] = None
        self.current_state: Dict[str, Any] = {}
        self.step_count: int = 0
        self.episode_count: int = 0
        self.episode_history: List[Dict[str, Any]] = []
        
    def _create_space(self, space_config: Dict[str, Any]) -> gym.Space:
        """Create a gym space from configuration."""
        if space_config["type"] == "box":
            return spaces.Box(
                low=np.array(space_config["low"]),
                high=np.array(space_config["high"]),
                dtype=space_config.get("dtype", np.float32)
            )
        elif space_config["type"] == "discrete":
            return spaces.Discrete(space_config["n"])
        elif space_config["type"] == "dict":
            return spaces.Dict({
                key: self._create_space(subspace)
                for key, subspace in space_config["spaces"].items()
            })
        else:
            raise ValueError(f"Unsupported space type: {space_config['type']}")
            
    async def reset(
        self,
        task: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None
    ) -> Dict[str, Any]:
        """Reset environment to initial state."""
        if seed is not None:
            np.random.seed(seed)
            
        # Generate or use provided task
        self.current_task = (
            task if task is not None
            else await self.task_generator.generate_task()
        )
        
        # Reset state
        self.current_state = self._create_initial_state()
        self.step_count = 0
        
        # Start new episode record if enabled
        if self.config.record_episodes:
            self._start_episode_recording()
            
        return self.current_state
        
    async def step(
        self,
        action: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Execute action and return new state."""
        if not self.action_space.contains(action):
            self.logger.warning(f"Invalid action: {action}")
            action = self.action_space.sample()
            
        # Update state
        next_state = await self._compute_next_state(action)
        
        # Compute reward
        reward = self._compute_base_reward(
            self.current_state,
            action,
            next_state
        )
        
        # Shape reward if task exists
        if self.current_task:
            reward = self.reward_shaper.shape_reward(
                reward,
                next_state,
                action,
                self.current_task
            )
            
        # Check termination
        done = self._check_termination(next_state)
        
        # Update step count
        self.step_count += 1
        
        # Record step if enabled
        if self.config.record_episodes:
            self._record_step(
                self.current_state,
                action,
                next_state,
                reward,
                done
            )
            
        # Update current state
        self.current_state = next_state
        
        # Return step results
        return next_state, reward, done, {
            "task": self.current_task,
            "step_count": self.step_count
        }
        
    async def _compute_next_state(
        self,
        action: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compute next state based on action."""
        # This should be implemented by specific environments
        raise NotImplementedError
        
    def _compute_base_reward(
        self,
        state: Dict[str, Any],
        action: Dict[str, Any],
        next_state: Dict[str, Any]
    ) -> float:
        """Compute base reward for transition."""
        # This should be implemented by specific environments
        raise NotImplementedError
        
    def _check_termination(self, state: Dict[str, Any]) -> bool:
        """Check if episode should terminate."""
        # Check step limit
        if self.step_count >= self.config.max_steps:
            return True
            
        # Check time limit if set
        if (
            self.config.time_limit and
            state.get("time", 0) >= self.config.time_limit
        ):
            return True
            
        # Check task-specific termination
        if self.current_task and "termination" in self.current_task:
            return self._evaluate_termination(
                state,
                self.current_task["termination"]
            )
            
        return False
        
    def _evaluate_termination(
        self,
        state: Dict[str, Any],
        conditions: Dict[str, Any]
    ) -> bool:
        """Evaluate task-specific termination conditions."""
        for key, condition in conditions.items():
            if key not in state:
                continue
                
            if isinstance(condition, dict):
                if condition["type"] == "threshold":
                    if state[key] >= condition["value"]:
                        return True
                elif condition["type"] == "range":
                    if (
                        state[key] < condition["min"] or
                        state[key] > condition["max"]
                    ):
                        return True
            else:
                if state[key] == condition:
                    return True
                    
        return False
        
    def _create_initial_state(self) -> Dict[str, Any]:
        """Create initial state."""
        # This should be implemented by specific environments
        raise NotImplementedError
        
    def _start_episode_recording(self) -> None:
        """Start recording a new episode."""
        self.episode_count += 1
        self.current_episode = {
            "id": str(uuid4()),
            "task": self.current_task,
            "start_time": datetime.now().isoformat(),
            "steps": []
        }
        
    def _record_step(
        self,
        state: Dict[str, Any],
        action: Dict[str, Any],
        next_state: Dict[str, Any],
        reward: float,
        done: bool
    ) -> None:
        """Record a step in the current episode."""
        if not hasattr(self, "current_episode"):
            return
            
        self.current_episode["steps"].append({
            "step": self.step_count,
            "state": state,
            "action": action,
            "next_state": next_state,
            "reward": reward,
            "done": done,
            "timestamp": datetime.now().isoformat()
        })
        
        if done:
            self.current_episode["end_time"] = datetime.now().isoformat()
            self.episode_history.append(self.current_episode)
            self._save_episode(self.current_episode)
            
    def _save_episode(self, episode: Dict[str, Any]) -> None:
        """Save episode recording to file."""
        episode_dir = Path(self.config.save_dir) / "episodes"
        episode_dir.mkdir(exist_ok=True)
        
        episode_path = episode_dir / f"episode_{episode['id']}.json"
        with open(episode_path, "w") as f:
            json.dump(episode, f, indent=2)
            
    async def close(self) -> None:
        """Clean up environment resources."""
        # Save any pending episode data
        if (
            hasattr(self, "current_episode") and
            self.current_episode["steps"]
        ):
            self.current_episode["end_time"] = datetime.now().isoformat()
            self.episode_history.append(self.current_episode)
            self._save_episode(self.current_episode)
            
        # Clear state
        self.current_task = None
        self.current_state = {} 