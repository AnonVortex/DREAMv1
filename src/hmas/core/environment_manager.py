"""Environment management and integration module for H-MAS."""

from typing import Dict, List, Optional, Any, Tuple, Callable, Union
from dataclasses import dataclass
import numpy as np
import gym
import torch
from pathlib import Path
import json
import logging
from datetime import datetime

from .environments import EnvironmentType
from .curriculum import CurriculumManager, CurriculumConfig
from .replay import ExperienceReplay, Experience

@dataclass
class EnvironmentConfig:
    """Configuration for environment management."""
    base_env_configs: Dict[EnvironmentType, Dict[str, Any]] = None
    max_episode_steps: int = 1000
    parallel_envs: int = 4
    record_video: bool = False
    video_dir: str = "videos"
    save_dir: str = "environment_data"
    enable_rendering: bool = False
    
    def __post_init__(self):
        """Initialize default configurations if none provided."""
        if self.base_env_configs is None:
            self.base_env_configs = {
                EnvironmentType.PERCEPTION: {
                    "observation_size": (84, 84, 3),
                    "num_objects": 5,
                    "max_steps": 100
                },
                EnvironmentType.REASONING: {
                    "problem_size": 10,
                    "max_steps": 50,
                    "reward_type": "sparse"
                },
                EnvironmentType.COMMUNICATION: {
                    "vocab_size": 100,
                    "max_message_length": 10,
                    "num_agents": 2
                },
                EnvironmentType.PLANNING: {
                    "horizon": 10,
                    "num_goals": 3,
                    "dynamic_obstacles": True
                }
            }

class EnvironmentManager:
    """Manages environment creation and interaction."""
    
    def __init__(
        self,
        config: EnvironmentConfig,
        curriculum_manager: CurriculumManager,
        experience_replay: ExperienceReplay
    ):
        """Initialize environment manager."""
        self.config = config
        self.curriculum_manager = curriculum_manager
        self.experience_replay = experience_replay
        self.logger = logging.getLogger("environment_manager")
        
        # Create save directories
        Path(config.save_dir).mkdir(parents=True, exist_ok=True)
        if config.record_video:
            Path(config.video_dir).mkdir(parents=True, exist_ok=True)
            
        self.active_envs: Dict[str, Any] = {}
        self.env_stats: Dict[str, Dict[str, Any]] = {}
        
    def create_environment(
        self,
        task_id: str,
        env_type: EnvironmentType,
        difficulty: float
    ) -> gym.Env:
        """Create a new environment instance."""
        base_config = self.config.base_env_configs[env_type].copy()
        
        # Get curriculum-adjusted configuration
        env_config = self.curriculum_manager.generate_curriculum(
            task_id,
            env_type,
            base_config
        )[int(difficulty * 10)]  # Convert difficulty to curriculum index
        
        # Create environment based on type
        if env_type == EnvironmentType.PERCEPTION:
            env = self._create_perception_env(env_config)
        elif env_type == EnvironmentType.REASONING:
            env = self._create_reasoning_env(env_config)
        elif env_type == EnvironmentType.COMMUNICATION:
            env = self._create_communication_env(env_config)
        elif env_type == EnvironmentType.PLANNING:
            env = self._create_planning_env(env_config)
        else:
            raise ValueError(f"Unsupported environment type: {env_type}")
            
        # Wrap environment with monitoring and recording
        env = self._wrap_environment(env, task_id)
        
        # Store environment instance
        self.active_envs[task_id] = env
        self.env_stats[task_id] = {
            "episodes_completed": 0,
            "total_steps": 0,
            "total_reward": 0.0,
            "success_rate": 0.0
        }
        
        return env
        
    def step(
        self,
        task_id: str,
        action: Union[int, np.ndarray]
    ) -> Tuple[Any, float, bool, Dict[str, Any]]:
        """Take a step in the environment."""
        if task_id not in self.active_envs:
            raise ValueError(f"No active environment for task {task_id}")
            
        env = self.active_envs[task_id]
        
        # Take step in environment
        next_state, reward, done, info = env.step(action)
        
        # Shape reward using curriculum manager
        shaped_reward = self.curriculum_manager.shape_reward(
            task_id,
            env.env_type,
            env.state,  # Current state before step
            action,
            reward,
            next_state,
            done,
            info
        )
        
        # Update statistics
        self.env_stats[task_id]["total_steps"] += 1
        self.env_stats[task_id]["total_reward"] += shaped_reward
        
        if done:
            self.env_stats[task_id]["episodes_completed"] += 1
            success = info.get("success", False)
            self._update_success_rate(task_id, success)
            
            # Update curriculum progress
            self.curriculum_manager.update_progress(
                task_id,
                env.difficulty,
                success,
                self.env_stats[task_id]["total_reward"]
            )
            
        return next_state, shaped_reward, done, info
        
    def reset(self, task_id: str) -> Any:
        """Reset environment for new episode."""
        if task_id not in self.active_envs:
            raise ValueError(f"No active environment for task {task_id}")
            
        env = self.active_envs[task_id]
        state = env.reset()
        
        # Reset episode statistics
        self.env_stats[task_id]["total_reward"] = 0.0
        
        return state
        
    def close(self, task_id: str) -> None:
        """Close environment and clean up resources."""
        if task_id in self.active_envs:
            self.active_envs[task_id].close()
            del self.active_envs[task_id]
            
    def get_env_info(self, task_id: str) -> Dict[str, Any]:
        """Get environment information and statistics."""
        if task_id not in self.active_envs:
            raise ValueError(f"No active environment for task {task_id}")
            
        env = self.active_envs[task_id]
        stats = self.env_stats[task_id]
        
        return {
            "env_type": env.env_type,
            "difficulty": env.difficulty,
            "observation_space": env.observation_space,
            "action_space": env.action_space,
            "episodes_completed": stats["episodes_completed"],
            "total_steps": stats["total_steps"],
            "average_reward": stats["total_reward"] / max(stats["total_steps"], 1),
            "success_rate": stats["success_rate"]
        }
        
    def _create_perception_env(self, config: Dict[str, Any]) -> gym.Env:
        """Create perception environment."""
        # TODO: Implement actual perception environment
        # This is a placeholder that should be replaced with actual implementation
        env = gym.make("CarRacing-v0")  # Placeholder
        env.env_type = EnvironmentType.PERCEPTION
        env.difficulty = config.get("difficulty", 0.0)
        return env
        
    def _create_reasoning_env(self, config: Dict[str, Any]) -> gym.Env:
        """Create reasoning environment."""
        # TODO: Implement actual reasoning environment
        env = gym.make("FrozenLake-v1")  # Placeholder
        env.env_type = EnvironmentType.REASONING
        env.difficulty = config.get("difficulty", 0.0)
        return env
        
    def _create_communication_env(self, config: Dict[str, Any]) -> gym.Env:
        """Create communication environment."""
        # TODO: Implement actual communication environment
        env = gym.make("CartPole-v1")  # Placeholder
        env.env_type = EnvironmentType.COMMUNICATION
        env.difficulty = config.get("difficulty", 0.0)
        return env
        
    def _create_planning_env(self, config: Dict[str, Any]) -> gym.Env:
        """Create planning environment."""
        # TODO: Implement actual planning environment
        env = gym.make("Acrobot-v1")  # Placeholder
        env.env_type = EnvironmentType.PLANNING
        env.difficulty = config.get("difficulty", 0.0)
        return env
        
    def _wrap_environment(self, env: gym.Env, task_id: str) -> gym.Env:
        """Wrap environment with monitoring and recording."""
        if self.config.max_episode_steps:
            env = gym.wrappers.TimeLimit(env, self.config.max_episode_steps)
            
        if self.config.record_video:
            env = gym.wrappers.RecordVideo(
                env,
                Path(self.config.video_dir) / task_id,
                episode_trigger=lambda x: x % 100 == 0  # Record every 100th episode
            )
            
        if self.config.enable_rendering:
            env = gym.wrappers.Monitor(
                env,
                Path(self.config.save_dir) / task_id,
                force=True,
                video_callable=False
            )
            
        return env
        
    def _update_success_rate(self, task_id: str, success: bool) -> None:
        """Update running success rate."""
        stats = self.env_stats[task_id]
        alpha = 0.1  # Exponential moving average factor
        current_rate = stats["success_rate"]
        stats["success_rate"] = current_rate * (1 - alpha) + success * alpha
        
    def save_state(self, save_path: str) -> None:
        """Save environment manager state."""
        state = {
            "env_stats": self.env_stats,
            "config": {
                "max_episode_steps": self.config.max_episode_steps,
                "parallel_envs": self.config.parallel_envs,
                "record_video": self.config.record_video,
                "video_dir": self.config.video_dir,
                "save_dir": self.config.save_dir,
                "enable_rendering": self.config.enable_rendering
            }
        }
        
        with open(save_path, "w") as f:
            json.dump(state, f, indent=2)
            
    def load_state(self, load_path: str) -> None:
        """Load environment manager state."""
        with open(load_path, "r") as f:
            state = json.load(f)
            
        self.env_stats = state["env_stats"]
        
        # Update config with loaded values
        for key, value in state["config"].items():
            setattr(self.config, key, value)
            
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all environments."""
        return {
            task_id: {
                "env_info": self.get_env_info(task_id),
                "curriculum_progress": self.curriculum_manager.get_progress_summary(task_id)
            }
            for task_id in self.active_envs.keys()
        } 