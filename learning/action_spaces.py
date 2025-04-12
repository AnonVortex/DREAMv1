from enum import Enum
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from dataclasses import dataclass

class ActionType(Enum):
    CONTINUOUS = "continuous"
    DISCRETE = "discrete"
    HYBRID = "hybrid"

@dataclass
class ActionBounds:
    low: np.ndarray
    high: np.ndarray
    
    def validate(self, action: np.ndarray) -> bool:
        return np.all(action >= self.low) and np.all(action <= self.high)

class ActionSpace:
    def __init__(
        self,
        action_type: ActionType,
        dimensions: int,
        bounds: Optional[ActionBounds] = None,
        discrete_values: Optional[List[int]] = None
    ):
        self.action_type = action_type
        self.dimensions = dimensions
        self.bounds = bounds
        self.discrete_values = discrete_values
        self._validate_initialization()
    
    def _validate_initialization(self):
        if self.action_type == ActionType.CONTINUOUS and self.bounds is None:
            raise ValueError("Continuous action space requires bounds")
        if self.action_type == ActionType.DISCRETE and self.discrete_values is None:
            raise ValueError("Discrete action space requires discrete values")
    
    def validate_action(self, action: np.ndarray) -> bool:
        """Validate if an action is within the defined space."""
        if self.action_type == ActionType.CONTINUOUS:
            return self.bounds.validate(action)
        elif self.action_type == ActionType.DISCRETE:
            return all(a in self.discrete_values for a in action)
        return False
    
    def normalize_action(self, action: np.ndarray) -> np.ndarray:
        """Normalize action to [-1, 1] range."""
        if self.action_type == ActionType.CONTINUOUS:
            return 2.0 * (action - self.bounds.low) / (self.bounds.high - self.bounds.low) - 1.0
        return action
    
    def denormalize_action(self, normalized_action: np.ndarray) -> np.ndarray:
        """Convert normalized action back to original space."""
        if self.action_type == ActionType.CONTINUOUS:
            return 0.5 * (normalized_action + 1.0) * (self.bounds.high - self.bounds.low) + self.bounds.low
        return normalized_action
    
    def sample(self) -> np.ndarray:
        """Sample a random action from the space."""
        if self.action_type == ActionType.CONTINUOUS:
            return np.random.uniform(self.bounds.low, self.bounds.high)
        elif self.action_type == ActionType.DISCRETE:
            return np.random.choice(self.discrete_values, size=self.dimensions)
        raise NotImplementedError("Sampling not implemented for this action type")

class NavigationActionSpace(ActionSpace):
    def __init__(self):
        super().__init__(
            action_type=ActionType.CONTINUOUS,
            dimensions=2,  # velocity_x, velocity_y
            bounds=ActionBounds(
                low=np.array([-1.0, -1.0]),
                high=np.array([1.0, 1.0])
            )
        )
    
    def add_safety_constraints(self, current_state: Dict) -> np.ndarray:
        """Apply safety constraints based on current state."""
        max_velocity = 0.5 if current_state.get("near_obstacle", False) else 1.0
        self.bounds = ActionBounds(
            low=np.array([-max_velocity, -max_velocity]),
            high=np.array([max_velocity, max_velocity])
        )

class ManipulationActionSpace(ActionSpace):
    def __init__(self):
        super().__init__(
            action_type=ActionType.CONTINUOUS,
            dimensions=7,  # joint angles or end-effector pose
            bounds=ActionBounds(
                low=np.array([-np.pi] * 7),
                high=np.array([np.pi] * 7)
            )
        )
    
    def validate_joint_limits(self, action: np.ndarray, current_joints: np.ndarray) -> bool:
        """Additional validation for joint limits and velocities."""
        max_joint_velocity = 0.5  # radians per second
        joint_velocity = action - current_joints
        return np.all(np.abs(joint_velocity) <= max_joint_velocity)

class CooperativeActionSpace(ActionSpace):
    def __init__(self, num_agents: int):
        super().__init__(
            action_type=ActionType.HYBRID,
            dimensions=num_agents * 3,  # position_x, position_y, communication_signal
            bounds=ActionBounds(
                low=np.array([-1.0, -1.0, 0.0] * num_agents),
                high=np.array([1.0, 1.0, 1.0] * num_agents)
            )
        )
        self.num_agents = num_agents
    
    def decompose_action(self, action: np.ndarray) -> List[Tuple[np.ndarray, float]]:
        """Decompose joint action into individual agent actions and communication signals."""
        actions = []
        for i in range(self.num_agents):
            start_idx = i * 3
            movement = action[start_idx:start_idx + 2]
            comm_signal = action[start_idx + 2]
            actions.append((movement, comm_signal))
        return actions

class ActionSpaceFactory:
    @staticmethod
    def create_action_space(task_type: str, **kwargs) -> ActionSpace:
        """Create appropriate action space based on task type."""
        if task_type == "navigation":
            return NavigationActionSpace()
        elif task_type == "manipulation":
            return ManipulationActionSpace()
        elif task_type == "cooperative":
            num_agents = kwargs.get("num_agents", 2)
            return CooperativeActionSpace(num_agents)
        else:
            raise ValueError(f"Unknown task type: {task_type}") 