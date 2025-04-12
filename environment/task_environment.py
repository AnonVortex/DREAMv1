from typing import Dict, List, Optional, Any, Tuple, Union
import numpy as np
import logging
from dataclasses import dataclass, field
from enum import Enum, auto
import gym
from gym import spaces

from .environment_manager import EnvironmentManager, EnvironmentConfig

logger = logging.getLogger(__name__)

class TaskType(Enum):
    """Types of tasks that can be performed in the environment."""
    NAVIGATION = auto()
    MANIPULATION = auto()
    COOPERATION = auto()
    EXPLORATION = auto()
    CUSTOM = auto()

@dataclass
class TaskConfig:
    """Configuration for a task environment."""
    task_type: TaskType = TaskType.NAVIGATION
    time_limit: float = 300.0  # seconds
    success_threshold: float = 0.95
    reward_shaping: bool = True
    sparse_rewards: bool = False
    observation_noise: float = 0.0
    action_noise: float = 0.0
    
    # Task-specific parameters
    goal_distance_threshold: float = 0.5
    min_agents_for_cooperation: int = 2
    max_agents_for_cooperation: int = 5
    object_interaction_radius: float = 1.0
    exploration_area_size: float = 10.0

class TaskEnvironment(gym.Env):
    """Task-specific environment that implements the gym interface."""
    
    def __init__(
        self,
        env_config: Optional[EnvironmentConfig] = None,
        task_config: Optional[TaskConfig] = None
    ):
        """Initialize the task environment."""
        super().__init__()
        
        self.env_config = env_config or EnvironmentConfig()
        self.task_config = task_config or TaskConfig()
        
        # Initialize environment manager
        self.env_manager = EnvironmentManager(self.env_config)
        
        # Initialize task-specific state
        self.current_step = 0
        self.episode_reward = 0.0
        self.task_completion = 0.0
        self.active_agents: List[str] = []
        self.goals: Dict[str, Tuple[float, float, float]] = {}
        
        # Define action and observation spaces
        self._setup_spaces()
        
        logger.info("Task environment initialized with type: %s", 
                   self.task_config.task_type.name)
    
    def _setup_spaces(self) -> None:
        """Set up the action and observation spaces."""
        # Action space: [force_x, force_y, force_z, torque_x, torque_y, torque_z]
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(6,),
            dtype=np.float32
        )
        
        # Observation space depends on task type
        if self.task_config.task_type == TaskType.NAVIGATION:
            # [position_x, position_y, position_z, velocity_x, velocity_y, velocity_z,
            #  goal_x, goal_y, goal_z, distance_to_goal]
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(10,),
                dtype=np.float32
            )
        
        elif self.task_config.task_type == TaskType.MANIPULATION:
            # Add object state to observation
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(16,),  # Additional object position and orientation
                dtype=np.float32
            )
        
        elif self.task_config.task_type == TaskType.COOPERATION:
            # Include other agents' states
            max_agents = self.task_config.max_agents_for_cooperation
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(10 * max_agents,),  # State for each potential agent
                dtype=np.float32
            )
        
        elif self.task_config.task_type == TaskType.EXPLORATION:
            # Include exploration map
            map_size = int(self.task_config.exploration_area_size)
            self.observation_space = spaces.Dict({
                "agent_state": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(6,),
                    dtype=np.float32
                ),
                "exploration_map": spaces.Box(
                    low=0,
                    high=1,
                    shape=(map_size, map_size),
                    dtype=np.float32
                )
            })
    
    def _get_observation(self) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """Get the current observation based on task type."""
        try:
            env_state = self.env_manager.get_observation()
            
            if self.task_config.task_type == TaskType.NAVIGATION:
                # Get the first agent's state (for single-agent navigation)
                agent_id = self.active_agents[0]
                agent_state = env_state["agents"][agent_id]
                goal = self.goals[agent_id]
                
                # Calculate distance to goal
                position = np.array(agent_state["position"])
                goal_pos = np.array(goal)
                distance = np.linalg.norm(position - goal_pos)
                
                obs = np.concatenate([
                    position,
                    agent_state["velocity"],
                    goal_pos,
                    [distance]
                ])
                
                # Add observation noise
                if self.task_config.observation_noise > 0:
                    obs += np.random.normal(
                        0,
                        self.task_config.observation_noise,
                        obs.shape
                    )
                
                return obs
            
            elif self.task_config.task_type == TaskType.MANIPULATION:
                # Include object state in observation
                agent_id = self.active_agents[0]
                agent_state = env_state["agents"][agent_id]
                
                # Get the first object's state
                object_id = list(env_state["objects"].keys())[0]
                object_state = env_state["objects"][object_id]
                
                obs = np.concatenate([
                    agent_state["position"],
                    agent_state["velocity"],
                    agent_state["orientation"],
                    object_state["position"],
                    object_state["orientation"]
                ])
                
                return obs
            
            elif self.task_config.task_type == TaskType.COOPERATION:
                # Combine all agents' states
                all_states = []
                for agent_id in self.active_agents:
                    agent_state = env_state["agents"][agent_id]
                    goal = self.goals[agent_id]
                    
                    position = np.array(agent_state["position"])
                    goal_pos = np.array(goal)
                    distance = np.linalg.norm(position - goal_pos)
                    
                    agent_obs = np.concatenate([
                        position,
                        agent_state["velocity"],
                        goal_pos,
                        [distance]
                    ])
                    all_states.append(agent_obs)
                
                # Pad with zeros if fewer than max agents
                while len(all_states) < self.task_config.max_agents_for_cooperation:
                    all_states.append(np.zeros(10))
                
                return np.concatenate(all_states)
            
            elif self.task_config.task_type == TaskType.EXPLORATION:
                agent_id = self.active_agents[0]
                agent_state = env_state["agents"][agent_id]
                
                # Create exploration map (simplified version)
                map_size = int(self.task_config.exploration_area_size)
                exploration_map = np.zeros((map_size, map_size))
                
                # Mark explored areas around agent position
                pos = agent_state["position"]
                x, y = int(pos[0] + map_size/2), int(pos[1] + map_size/2)
                radius = 2
                for i in range(max(0, x-radius), min(map_size, x+radius+1)):
                    for j in range(max(0, y-radius), min(map_size, y+radius+1)):
                        exploration_map[i, j] = 1
                
                return {
                    "agent_state": np.concatenate([
                        agent_state["position"],
                        agent_state["velocity"]
                    ]),
                    "exploration_map": exploration_map
                }
            
            else:
                raise ValueError(f"Unsupported task type: {self.task_config.task_type}")
            
        except Exception as e:
            logger.error("Failed to get observation: %s", str(e))
            raise
    
    def _compute_reward(self) -> float:
        """Compute the reward based on task type and current state."""
        try:
            env_state = self.env_manager.get_observation()
            reward = 0.0
            
            if self.task_config.task_type == TaskType.NAVIGATION:
                agent_id = self.active_agents[0]
                agent_state = env_state["agents"][agent_id]
                goal = self.goals[agent_id]
                
                # Calculate distance to goal
                position = np.array(agent_state["position"])
                goal_pos = np.array(goal)
                distance = np.linalg.norm(position - goal_pos)
                
                if distance < self.task_config.goal_distance_threshold:
                    # Goal reached
                    reward = 1.0
                    self.task_completion = 1.0
                elif self.task_config.reward_shaping:
                    # Shaped reward based on distance
                    reward = -distance / self.task_config.exploration_area_size
                else:
                    # Sparse reward
                    reward = 0.0
            
            elif self.task_config.task_type == TaskType.MANIPULATION:
                agent_id = self.active_agents[0]
                agent_state = env_state["agents"][agent_id]
                object_id = list(env_state["objects"].keys())[0]
                object_state = env_state["objects"][object_id]
                
                # Calculate distance to object
                agent_pos = np.array(agent_state["position"])
                object_pos = np.array(object_state["position"])
                distance = np.linalg.norm(agent_pos - object_pos)
                
                if distance < self.task_config.object_interaction_radius:
                    reward = 1.0
                    self.task_completion = 1.0
                elif self.task_config.reward_shaping:
                    reward = -distance / self.task_config.exploration_area_size
            
            elif self.task_config.task_type == TaskType.COOPERATION:
                # Reward based on average progress of all agents
                distances = []
                for agent_id in self.active_agents:
                    agent_state = env_state["agents"][agent_id]
                    goal = self.goals[agent_id]
                    
                    position = np.array(agent_state["position"])
                    goal_pos = np.array(goal)
                    distance = np.linalg.norm(position - goal_pos)
                    distances.append(distance)
                
                avg_distance = np.mean(distances)
                if avg_distance < self.task_config.goal_distance_threshold:
                    reward = 1.0
                    self.task_completion = 1.0
                elif self.task_config.reward_shaping:
                    reward = -avg_distance / self.task_config.exploration_area_size
            
            elif self.task_config.task_type == TaskType.EXPLORATION:
                # Reward based on area explored
                obs = self._get_observation()
                exploration_map = obs["exploration_map"]
                explored_ratio = np.sum(exploration_map) / exploration_map.size
                
                if explored_ratio > self.task_config.success_threshold:
                    reward = 1.0
                    self.task_completion = 1.0
                elif self.task_config.reward_shaping:
                    reward = explored_ratio
            
            return reward
            
        except Exception as e:
            logger.error("Failed to compute reward: %s", str(e))
            raise
    
    def step(self, action: np.ndarray) -> Tuple[Any, float, bool, Dict[str, Any]]:
        """Step the environment forward."""
        try:
            # Add action noise
            if self.task_config.action_noise > 0:
                action += np.random.normal(
                    0,
                    self.task_config.action_noise,
                    action.shape
                )
            
            # Apply action to first agent (or distribute among agents for cooperation)
            if self.task_config.task_type == TaskType.COOPERATION:
                # Split action among agents (simplified)
                action_per_agent = action.reshape(-1, 6)
                for i, agent_id in enumerate(self.active_agents):
                    if i < len(action_per_agent):
                        self.env_manager.apply_action(agent_id, action_per_agent[i])
            else:
                self.env_manager.apply_action(self.active_agents[0], action)
            
            # Step physics simulation
            self.env_manager.step()
            self.current_step += 1
            
            # Get observation and compute reward
            observation = self._get_observation()
            reward = self._compute_reward()
            self.episode_reward += reward
            
            # Check termination conditions
            done = (
                self.current_step >= self.env_config.max_steps or
                self.task_completion >= self.task_config.success_threshold
            )
            
            # Additional info
            info = {
                "current_step": self.current_step,
                "episode_reward": self.episode_reward,
                "task_completion": self.task_completion
            }
            
            return observation, reward, done, info
            
        except Exception as e:
            logger.error("Failed to step environment: %s", str(e))
            raise
    
    def reset(self) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """Reset the environment to initial state."""
        try:
            # Reset physics simulation
            self.env_manager.reset()
            
            # Reset internal state
            self.current_step = 0
            self.episode_reward = 0.0
            self.task_completion = 0.0
            self.active_agents.clear()
            self.goals.clear()
            
            # Set up task-specific initial state
            if self.task_config.task_type == TaskType.NAVIGATION:
                # Add single agent with random position
                agent_id = "agent_0"
                agent_pos = (
                    np.random.uniform(-5, 5),
                    np.random.uniform(-5, 5),
                    0.5
                )
                self.env_manager.add_agent(agent_id, agent_pos)
                self.active_agents.append(agent_id)
                
                # Set random goal
                goal_pos = (
                    np.random.uniform(-5, 5),
                    np.random.uniform(-5, 5),
                    0.5
                )
                self.goals[agent_id] = goal_pos
            
            elif self.task_config.task_type == TaskType.MANIPULATION:
                # Add agent and object
                agent_id = "agent_0"
                agent_pos = (0, 0, 0.5)
                self.env_manager.add_agent(agent_id, agent_pos)
                self.active_agents.append(agent_id)
                
                # Add object to manipulate
                object_pos = (
                    np.random.uniform(-2, 2),
                    np.random.uniform(-2, 2),
                    0.5
                )
                self.env_manager.add_object(
                    "target_object",
                    "cube.urdf",
                    object_pos
                )
            
            elif self.task_config.task_type == TaskType.COOPERATION:
                # Add multiple agents
                num_agents = np.random.randint(
                    self.task_config.min_agents_for_cooperation,
                    self.task_config.max_agents_for_cooperation + 1
                )
                
                for i in range(num_agents):
                    agent_id = f"agent_{i}"
                    agent_pos = (
                        np.random.uniform(-5, 5),
                        np.random.uniform(-5, 5),
                        0.5
                    )
                    self.env_manager.add_agent(agent_id, agent_pos)
                    self.active_agents.append(agent_id)
                    
                    # Set random goal for each agent
                    goal_pos = (
                        np.random.uniform(-5, 5),
                        np.random.uniform(-5, 5),
                        0.5
                    )
                    self.goals[agent_id] = goal_pos
            
            elif self.task_config.task_type == TaskType.EXPLORATION:
                # Add single agent for exploration
                agent_id = "agent_0"
                agent_pos = (0, 0, 0.5)  # Start at center
                self.env_manager.add_agent(agent_id, agent_pos)
                self.active_agents.append(agent_id)
            
            return self._get_observation()
            
        except Exception as e:
            logger.error("Failed to reset environment: %s", str(e))
            raise
    
    def render(self, mode: str = "rgb_array") -> np.ndarray:
        """Render the environment."""
        return self.env_manager.render(mode)
    
    def close(self) -> None:
        """Clean up resources."""
        self.env_manager.close() 