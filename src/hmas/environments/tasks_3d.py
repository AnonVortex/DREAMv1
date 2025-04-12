"""Task configurations for 3D Environment.

This module provides predefined tasks and task generators for the 3D environment.
Tasks include navigation, manipulation, balancing, and multi-agent scenarios.
"""

from typing import Dict, List, Any, Optional
import numpy as np
from dataclasses import dataclass
from enum import Enum, auto
import json
from pathlib import Path
import random
from uuid import uuid4

class TaskType(Enum):
    """Types of 3D environment tasks."""
    NAVIGATION = auto()       # Path finding and obstacle avoidance
    MANIPULATION = auto()     # Object grasping and manipulation
    BALANCE = auto()         # Object balancing
    CONSTRUCTION = auto()    # Building structures
    MULTI_AGENT = auto()     # Cooperative tasks
    EXPLORATION = auto()     # Environment exploration

@dataclass
class TaskParameters:
    """Common parameters for task generation."""
    difficulty: float = 0.5  # 0.0 to 1.0
    workspace_size: float = 10.0  # Size of task workspace
    num_obstacles: int = 5
    time_limit: Optional[float] = None
    require_vision: bool = True
    allow_tool_use: bool = False
    cooperative: bool = False

class NavigationTaskGenerator:
    """Generator for navigation tasks."""
    
    def __init__(self, params: TaskParameters):
        self.params = params
        
    def generate(self) -> Dict[str, Any]:
        """Generate a navigation task."""
        # Scale complexity with difficulty
        num_waypoints = max(1, int(self.params.difficulty * 5))  # 1-5 waypoints
        num_obstacles = max(
            0,
            int(self.params.num_obstacles * self.params.difficulty)
        )
        
        # Generate waypoints
        waypoints = self._generate_waypoints(num_waypoints)
        
        # Generate obstacles
        obstacles = self._generate_obstacles(num_obstacles, waypoints)
        
        # Create task specification
        task = {
            "id": str(uuid4()),
            "type": TaskType.NAVIGATION.name,
            "parameters": {
                "difficulty": self.params.difficulty,
                "time_limit": self.params.time_limit,
                "workspace": [-self.params.workspace_size/2, self.params.workspace_size/2],
                "objects": [
                    # Agent
                    {
                        "id": "agent",
                        "type": "AGENT",
                        "position": waypoints[0].tolist(),
                        "rotation": [1, 0, 0, 0],
                        "scale": [0.5, 0.5, 1.0],
                        "mass": 1.0
                    },
                    # Target
                    {
                        "id": "target",
                        "type": "TARGET",
                        "position": waypoints[-1].tolist(),
                        "rotation": [1, 0, 0, 0],
                        "scale": [0.3, 0.3, 0.3]
                    }
                ],
                "objectives": [
                    {
                        "type": "reach_target",
                        "agent_id": "agent",
                        "target_id": "target",
                        "weight": 1.0
                    }
                ],
                "waypoints": [w.tolist() for w in waypoints],
                "vision_required": self.params.require_vision
            }
        }
        
        # Add obstacles to objects list
        task["parameters"]["objects"].extend(obstacles)
        
        return task
        
    def _generate_waypoints(self, num_points: int) -> List[np.ndarray]:
        """Generate waypoints for navigation."""
        waypoints = []
        workspace = self.params.workspace_size
        
        for i in range(num_points):
            # Generate point with increasing height complexity
            point = np.array([
                random.uniform(-workspace/2, workspace/2),
                random.uniform(-workspace/2, workspace/2),
                random.uniform(0, self.params.difficulty * workspace/4)
            ])
            waypoints.append(point)
            
        return waypoints
        
    def _generate_obstacles(
        self,
        num_obstacles: int,
        waypoints: List[np.ndarray]
    ) -> List[Dict[str, Any]]:
        """Generate obstacles avoiding waypoints."""
        obstacles = []
        workspace = self.params.workspace_size
        min_distance = workspace * 0.1  # Minimum distance from waypoints
        
        for i in range(num_obstacles):
            while True:
                # Generate potential obstacle position
                position = np.array([
                    random.uniform(-workspace/2, workspace/2),
                    random.uniform(-workspace/2, workspace/2),
                    random.uniform(0, workspace/2)
                ])
                
                # Check distance from waypoints
                valid = True
                for waypoint in waypoints:
                    if np.linalg.norm(position - waypoint) < min_distance:
                        valid = False
                        break
                        
                if valid:
                    break
                    
            # Create obstacle specification
            obstacle = {
                "id": f"obstacle_{i}",
                "type": "OBSTACLE",
                "position": position.tolist(),
                "rotation": [1, 0, 0, 0],
                "scale": [
                    random.uniform(0.5, 2.0),
                    random.uniform(0.5, 2.0),
                    random.uniform(1.0, 3.0)
                ]
            }
            obstacles.append(obstacle)
            
        return obstacles

class ManipulationTaskGenerator:
    """Generator for object manipulation tasks."""
    
    def __init__(self, params: TaskParameters):
        self.params = params
        
    def generate(self) -> Dict[str, Any]:
        """Generate a manipulation task."""
        # Scale complexity with difficulty
        num_objects = max(1, int(self.params.difficulty * 3))  # 1-3 objects
        require_tools = self.params.allow_tool_use and self.params.difficulty > 0.7
        
        # Generate target positions
        target_positions = self._generate_target_positions(num_objects)
        
        # Create task specification
        task = {
            "id": str(uuid4()),
            "type": TaskType.MANIPULATION.name,
            "parameters": {
                "difficulty": self.params.difficulty,
                "time_limit": self.params.time_limit,
                "workspace": [-self.params.workspace_size/2, self.params.workspace_size/2],
                "objects": [
                    # Agent
                    {
                        "id": "agent",
                        "type": "AGENT",
                        "position": [0, -self.params.workspace_size/3, 1.0],
                        "rotation": [1, 0, 0, 0],
                        "scale": [0.5, 0.5, 1.0],
                        "mass": 1.0
                    }
                ],
                "objectives": [],
                "vision_required": self.params.require_vision,
                "tool_use_required": require_tools
            }
        }
        
        # Add manipulable objects and targets
        for i in range(num_objects):
            # Object to manipulate
            object_pos = np.array([
                random.uniform(-self.params.workspace_size/4, self.params.workspace_size/4),
                -self.params.workspace_size/4,
                0.5
            ])
            
            task["parameters"]["objects"].extend([
                {
                    "id": f"object_{i}",
                    "type": "DYNAMIC",
                    "position": object_pos.tolist(),
                    "rotation": [1, 0, 0, 0],
                    "scale": [0.3, 0.3, 0.3],
                    "mass": random.uniform(0.5, 2.0),
                    "friction": random.uniform(0.4, 0.6)
                },
                {
                    "id": f"target_{i}",
                    "type": "TARGET",
                    "position": target_positions[i].tolist(),
                    "rotation": [1, 0, 0, 0],
                    "scale": [0.3, 0.3, 0.3]
                }
            ])
            
            # Add objective
            task["parameters"]["objectives"].append({
                "type": "reach_target",
                "agent_id": f"object_{i}",
                "target_id": f"target_{i}",
                "weight": 1.0/num_objects
            })
            
        # Add tools if required
        if require_tools:
            task["parameters"]["objects"].append({
                "id": "tool",
                "type": "TOOL",
                "position": [
                    self.params.workspace_size/4,
                    -self.params.workspace_size/4,
                    0.5
                ],
                "rotation": [1, 0, 0, 0],
                "scale": [0.2, 0.2, 1.0],
                "mass": 0.5,
                "friction": 0.7
            })
            
        return task
        
    def _generate_target_positions(self, num_positions: int) -> List[np.ndarray]:
        """Generate target positions for objects."""
        positions = []
        workspace = self.params.workspace_size
        min_distance = workspace * 0.2  # Minimum distance between targets
        
        for i in range(num_positions):
            while True:
                # Generate potential position
                position = np.array([
                    random.uniform(-workspace/4, workspace/4),
                    workspace/4,  # Targets on opposite side
                    random.uniform(0, workspace/4)
                ])
                
                # Check distance from other targets
                valid = True
                for other_pos in positions:
                    if np.linalg.norm(position - other_pos) < min_distance:
                        valid = False
                        break
                        
                if valid:
                    break
                    
            positions.append(position)
            
        return positions

class BalanceTaskGenerator:
    """Generator for balance-based tasks."""
    
    def __init__(self, params: TaskParameters):
        self.params = params
        
    def generate(self) -> Dict[str, Any]:
        """Generate a balance task."""
        # Scale complexity with difficulty
        num_objects = max(1, int(self.params.difficulty * 3))  # 1-3 objects
        platform_tilt = self.params.difficulty * 30  # 0-30 degrees
        
        # Create task specification
        task = {
            "id": str(uuid4()),
            "type": TaskType.BALANCE.name,
            "parameters": {
                "difficulty": self.params.difficulty,
                "time_limit": self.params.time_limit,
                "workspace": [-self.params.workspace_size/2, self.params.workspace_size/2],
                "objects": [
                    # Agent
                    {
                        "id": "agent",
                        "type": "AGENT",
                        "position": [0, -self.params.workspace_size/3, 1.0],
                        "rotation": [1, 0, 0, 0],
                        "scale": [0.5, 0.5, 1.0],
                        "mass": 1.0
                    },
                    # Platform
                    {
                        "id": "platform",
                        "type": "DYNAMIC",
                        "position": [0, 0, 1.0],
                        "rotation": self._euler_to_quaternion(0, platform_tilt, 0),
                        "scale": [2.0, 2.0, 0.1],
                        "mass": 5.0,
                        "friction": 0.8
                    }
                ],
                "objectives": [
                    {
                        "type": "maintain_balance",
                        "object_id": "platform",
                        "weight": 1.0,
                        "tolerance": 0.1
                    }
                ],
                "vision_required": self.params.require_vision
            }
        }
        
        # Add objects to balance
        for i in range(num_objects):
            position = np.array([
                random.uniform(-0.5, 0.5),
                random.uniform(-0.5, 0.5),
                1.5  # Above platform
            ])
            
            task["parameters"]["objects"].append({
                "id": f"object_{i}",
                "type": "DYNAMIC",
                "position": position.tolist(),
                "rotation": [1, 0, 0, 0],
                "scale": [0.3, 0.3, 0.3],
                "mass": random.uniform(0.5, 2.0),
                "friction": random.uniform(0.4, 0.6)
            })
            
            # Add balance objective for object
            task["parameters"]["objectives"].append({
                "type": "maintain_balance",
                "object_id": f"object_{i}",
                "weight": 0.5/num_objects,
                "tolerance": 0.2
            })
            
        return task
        
    def _euler_to_quaternion(
        self,
        roll: float,
        pitch: float,
        yaw: float
    ) -> List[float]:
        """Convert Euler angles to quaternion."""
        # Convert to radians
        roll = np.radians(roll)
        pitch = np.radians(pitch)
        yaw = np.radians(yaw)
        
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)
        
        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy
        
        return [w, x, y, z]

class ConstructionTaskGenerator:
    """Generator for construction and assembly tasks."""
    
    def __init__(self, params: TaskParameters):
        self.params = params
        
    def generate(self) -> Dict[str, Any]:
        """Generate a construction task."""
        # Scale complexity with difficulty
        num_pieces = max(2, int(self.params.difficulty * 5))  # 2-5 pieces
        structure_height = max(2, int(self.params.difficulty * 4))  # 2-4 levels
        require_tools = self.params.allow_tool_use and self.params.difficulty > 0.6
        
        # Generate target structure
        structure_spec = self._generate_structure_spec(num_pieces, structure_height)
        
        # Create task specification
        task = {
            "id": str(uuid4()),
            "type": TaskType.CONSTRUCTION.name,
            "parameters": {
                "difficulty": self.params.difficulty,
                "time_limit": self.params.time_limit,
                "workspace": [-self.params.workspace_size/2, self.params.workspace_size/2],
                "objects": [
                    # Agent
                    {
                        "id": "agent",
                        "type": "AGENT",
                        "position": [0, -self.params.workspace_size/3, 1.0],
                        "rotation": [1, 0, 0, 0],
                        "scale": [0.5, 0.5, 1.0],
                        "mass": 1.0
                    }
                ],
                "objectives": [],
                "vision_required": self.params.require_vision,
                "tool_use_required": require_tools,
                "target_structure": structure_spec
            }
        }
        
        # Add construction pieces
        for i, piece_spec in enumerate(structure_spec["pieces"]):
            # Initial position of piece (scattered around workspace)
            initial_pos = np.array([
                random.uniform(-self.params.workspace_size/4, self.params.workspace_size/4),
                -self.params.workspace_size/4,
                0.5
            ])
            
            task["parameters"]["objects"].append({
                "id": f"piece_{i}",
                "type": "DYNAMIC",
                "position": initial_pos.tolist(),
                "rotation": [1, 0, 0, 0],
                "scale": piece_spec["scale"],
                "mass": piece_spec.get("mass", 1.0),
                "friction": piece_spec.get("friction", 0.7),
                "color": piece_spec.get("color", [0.8, 0.8, 0.8]),
                "mesh_path": piece_spec.get("mesh_path")
            })
            
            # Add placement objective
            task["parameters"]["objectives"].append({
                "type": "place_piece",
                "piece_id": f"piece_{i}",
                "target_position": piece_spec["target_position"],
                "target_rotation": piece_spec["target_rotation"],
                "weight": 1.0/num_pieces,
                "tolerance_position": 0.1,
                "tolerance_rotation": 0.2
            })
        
        # Add tools if required
        if require_tools:
            task["parameters"]["objects"].extend([
                {
                    "id": "gripper",
                    "type": "TOOL",
                    "position": [
                        self.params.workspace_size/4,
                        -self.params.workspace_size/4,
                        0.5
                    ],
                    "rotation": [1, 0, 0, 0],
                    "scale": [0.2, 0.2, 0.4],
                    "mass": 0.5,
                    "friction": 0.9
                },
                {
                    "id": "support_tool",
                    "type": "TOOL",
                    "position": [
                        -self.params.workspace_size/4,
                        -self.params.workspace_size/4,
                        0.5
                    ],
                    "rotation": [1, 0, 0, 0],
                    "scale": [0.3, 0.3, 1.0],
                    "mass": 0.8,
                    "friction": 0.7
                }
            ])
            
        return task
        
    def _generate_structure_spec(
        self,
        num_pieces: int,
        height: int
    ) -> Dict[str, Any]:
        """Generate specification for target structure."""
        pieces = []
        workspace = self.params.workspace_size
        base_size = 0.5  # Base size for pieces
        
        # Define piece types
        piece_types = [
            {
                "name": "block",
                "scale": [base_size, base_size, base_size],
                "mass": 1.0,
                "friction": 0.7
            },
            {
                "name": "platform",
                "scale": [base_size * 2, base_size * 2, base_size * 0.2],
                "mass": 2.0,
                "friction": 0.8
            },
            {
                "name": "beam",
                "scale": [base_size * 0.2, base_size * 2, base_size * 0.2],
                "mass": 0.8,
                "friction": 0.6
            }
        ]
        
        # Generate structure layout
        current_height = 0.0
        level_pieces = max(1, num_pieces // height)
        
        for level in range(height):
            level_offset = np.array([0, 0, current_height])
            
            # Place pieces in current level
            for i in range(level_pieces):
                # Select piece type based on position and difficulty
                piece_type = random.choice(piece_types)
                
                # Calculate target position
                angle = (i / level_pieces) * 2 * np.pi
                radius = base_size * (1 + level * 0.5)  # Increasing radius with height
                position = np.array([
                    np.cos(angle) * radius,
                    np.sin(angle) * radius,
                    0
                ]) + level_offset
                
                # Calculate target rotation
                rotation = self._euler_to_quaternion(0, 0, np.degrees(angle))
                
                # Create piece specification
                piece = {
                    "type": piece_type["name"],
                    "scale": piece_type["scale"],
                    "mass": piece_type["mass"],
                    "friction": piece_type["friction"],
                    "target_position": position.tolist(),
                    "target_rotation": rotation,
                    "color": [
                        random.uniform(0.4, 0.8),
                        random.uniform(0.4, 0.8),
                        random.uniform(0.4, 0.8)
                    ]
                }
                pieces.append(piece)
                
            # Update height for next level
            current_height += max(p["scale"][2] for p in piece_types) * 1.2
            
        return {
            "pieces": pieces,
            "height": height,
            "base_size": base_size,
            "complexity": self.params.difficulty
        }
        
    def _euler_to_quaternion(
        self,
        roll: float,
        pitch: float,
        yaw: float
    ) -> List[float]:
        """Convert Euler angles to quaternion."""
        # Convert to radians
        roll = np.radians(roll)
        pitch = np.radians(pitch)
        yaw = np.radians(yaw)
        
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)
        
        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy
        
        return [w, x, y, z]

class MultiAgentTaskGenerator:
    """Generator for multi-agent cooperative tasks."""
    
    def __init__(self, params: TaskParameters):
        self.params = params
        if not self.params.cooperative:
            self.params.cooperative = True  # Force cooperative mode
        
    def generate(self) -> Dict[str, Any]:
        """Generate a multi-agent task."""
        # Scale complexity with difficulty
        num_agents = max(2, int(self.params.difficulty * 4))  # 2-4 agents
        num_objectives = max(2, int(self.params.difficulty * 3))  # 2-3 objectives
        require_tools = self.params.allow_tool_use and self.params.difficulty > 0.6
        
        # Generate team objectives
        objectives = self._generate_team_objectives(num_objectives)
        
        # Create task specification
        task = {
            "id": str(uuid4()),
            "type": TaskType.MULTI_AGENT.name,
            "parameters": {
                "difficulty": self.params.difficulty,
                "time_limit": self.params.time_limit,
                "workspace": [-self.params.workspace_size/2, self.params.workspace_size/2],
                "objects": [],
                "objectives": objectives,
                "vision_required": self.params.require_vision,
                "tool_use_required": require_tools,
                "communication_required": True,
                "roles": self._generate_agent_roles(num_agents)
            }
        }
        
        # Add agents with different roles
        for i in range(num_agents):
            position = np.array([
                random.uniform(-self.params.workspace_size/4, self.params.workspace_size/4),
                -self.params.workspace_size/3,
                1.0
            ])
            
            task["parameters"]["objects"].append({
                "id": f"agent_{i}",
                "type": "AGENT",
                "position": position.tolist(),
                "rotation": [1, 0, 0, 0],
                "scale": [0.5, 0.5, 1.0],
                "mass": 1.0,
                "role": task["parameters"]["roles"][i]
            })
        
        # Add shared resources and tools
        self._add_shared_resources(task)
        if require_tools:
            self._add_tools(task)
            
        return task
        
    def _generate_team_objectives(self, num_objectives: int) -> List[Dict[str, Any]]:
        """Generate cooperative team objectives."""
        objectives = []
        objective_types = ["transport", "assembly", "coordination"]
        
        for i in range(num_objectives):
            obj_type = random.choice(objective_types)
            
            if obj_type == "transport":
                # Cooperative object transportation
                target_pos = np.array([
                    random.uniform(-self.params.workspace_size/4, self.params.workspace_size/4),
                    self.params.workspace_size/4,
                    random.uniform(0.5, 1.5)
                ])
                
                objectives.append({
                    "type": "cooperative_transport",
                    "object_id": f"heavy_object_{i}",
                    "target_position": target_pos.tolist(),
                    "min_agents_required": 2,
                    "weight": 1.0/num_objectives,
                    "tolerance": 0.2
                })
                
            elif obj_type == "assembly":
                # Cooperative assembly task
                objectives.append({
                    "type": "cooperative_assembly",
                    "structure_id": f"structure_{i}",
                    "roles_required": ["assembler", "supporter"],
                    "weight": 1.0/num_objectives,
                    "tolerance": 0.15
                })
                
            else:  # coordination
                # Position coordination task
                objectives.append({
                    "type": "coordinate_positions",
                    "formation_shape": random.choice(["line", "triangle", "square"]),
                    "spacing": random.uniform(1.0, 2.0),
                    "weight": 1.0/num_objectives,
                    "tolerance": 0.2
                })
                
        return objectives
        
    def _generate_agent_roles(self, num_agents: int) -> List[Dict[str, Any]]:
        """Generate role specifications for agents."""
        roles = []
        available_roles = [
            {
                "name": "transporter",
                "capabilities": ["lift", "carry", "push"],
                "strength": random.uniform(0.7, 1.0)
            },
            {
                "name": "assembler",
                "capabilities": ["grasp", "place", "align"],
                "precision": random.uniform(0.7, 1.0)
            },
            {
                "name": "supporter",
                "capabilities": ["brace", "stabilize", "guide"],
                "stability": random.uniform(0.7, 1.0)
            },
            {
                "name": "coordinator",
                "capabilities": ["scan", "plan", "direct"],
                "range": random.uniform(0.7, 1.0)
            }
        ]
        
        # Ensure at least one of each required role
        min_roles = min(num_agents, len(available_roles))
        roles.extend(random.sample(available_roles, min_roles))
        
        # Fill remaining slots randomly
        while len(roles) < num_agents:
            roles.append(random.choice(available_roles))
            
        return roles
        
    def _add_shared_resources(self, task: Dict[str, Any]) -> None:
        """Add shared resources to the task."""
        # Add heavy objects for transport
        for obj in task["parameters"]["objectives"]:
            if obj["type"] == "cooperative_transport":
                position = np.array([
                    random.uniform(-self.params.workspace_size/4, self.params.workspace_size/4),
                    -self.params.workspace_size/4,
                    0.5
                ])
                
                task["parameters"]["objects"].append({
                    "id": obj["object_id"],
                    "type": "DYNAMIC",
                    "position": position.tolist(),
                    "rotation": [1, 0, 0, 0],
                    "scale": [1.0, 1.0, 0.5],
                    "mass": 5.0,  # Heavy object requiring multiple agents
                    "friction": 0.7,
                    "color": [0.8, 0.2, 0.2]
                })
                
        # Add assembly structures
        for obj in task["parameters"]["objectives"]:
            if obj["type"] == "cooperative_assembly":
                # Add base platform
                task["parameters"]["objects"].append({
                    "id": f"{obj['structure_id']}_base",
                    "type": "STATIC",
                    "position": [0, 0, 0],
                    "rotation": [1, 0, 0, 0],
                    "scale": [2.0, 2.0, 0.1],
                    "color": [0.2, 0.8, 0.2]
                })
                
                # Add assembly pieces
                num_pieces = max(2, int(self.params.difficulty * 4))
                for i in range(num_pieces):
                    position = np.array([
                        random.uniform(-self.params.workspace_size/4, self.params.workspace_size/4),
                        -self.params.workspace_size/4,
                        0.5
                    ])
                    
                    task["parameters"]["objects"].append({
                        "id": f"{obj['structure_id']}_piece_{i}",
                        "type": "DYNAMIC",
                        "position": position.tolist(),
                        "rotation": [1, 0, 0, 0],
                        "scale": [0.4, 0.4, 0.4],
                        "mass": 2.0,
                        "friction": 0.7,
                        "color": [0.2, 0.8, 0.2]
                    })
    
    def _add_tools(self, task: Dict[str, Any]) -> None:
        """Add shared tools to the task."""
        tools = [
            {
                "id": "lifting_tool",
                "type": "TOOL",
                "position": [
                    self.params.workspace_size/4,
                    -self.params.workspace_size/4,
                    0.5
                ],
                "rotation": [1, 0, 0, 0],
                "scale": [0.3, 0.3, 1.0],
                "mass": 0.8,
                "friction": 0.9,
                "color": [0.8, 0.8, 0.2]
            },
            {
                "id": "assembly_tool",
                "type": "TOOL",
                "position": [
                    0,
                    -self.params.workspace_size/4,
                    0.5
                ],
                "rotation": [1, 0, 0, 0],
                "scale": [0.2, 0.2, 0.8],
                "mass": 0.5,
                "friction": 0.8,
                "color": [0.2, 0.8, 0.8]
            },
            {
                "id": "support_tool",
                "type": "TOOL",
                "position": [
                    -self.params.workspace_size/4,
                    -self.params.workspace_size/4,
                    0.5
                ],
                "rotation": [1, 0, 0, 0],
                "scale": [0.4, 0.4, 0.6],
                "mass": 1.0,
                "friction": 0.7,
                "color": [0.8, 0.2, 0.8]
            }
        ]
        
        task["parameters"]["objects"].extend(tools)

class ExplorationTaskGenerator:
    """Generator for environment exploration tasks."""
    
    def __init__(self, params: TaskParameters):
        self.params = params
        
    def generate(self) -> Dict[str, Any]:
        """Generate an exploration task."""
        # Scale complexity with difficulty
        num_regions = max(2, int(self.params.difficulty * 5))  # 2-5 regions
        num_resources = max(2, int(self.params.difficulty * 4))  # 2-4 resources
        num_obstacles = max(3, int(self.params.difficulty * 6))  # 3-6 obstacles
        fog_of_war = self.params.difficulty > 0.4  # Enable fog of war at higher difficulties
        
        # Generate regions and resources
        regions = self._generate_regions(num_regions)
        resources = self._generate_resources(num_resources, regions)
        obstacles = self._generate_obstacles(num_obstacles, regions, resources)
        
        # Create task specification
        task = {
            "id": str(uuid4()),
            "type": TaskType.EXPLORATION.name,
            "parameters": {
                "difficulty": self.params.difficulty,
                "time_limit": self.params.time_limit,
                "workspace": [-self.params.workspace_size/2, self.params.workspace_size/2],
                "objects": [
                    # Agent with exploration capabilities
                    {
                        "id": "explorer",
                        "type": "AGENT",
                        "position": [0, -self.params.workspace_size/3, 1.0],
                        "rotation": [1, 0, 0, 0],
                        "scale": [0.5, 0.5, 1.0],
                        "mass": 1.0,
                        "sensors": {
                            "range_sensor": {
                                "range": 5.0,
                                "angle": 120.0,
                                "noise": 0.1
                            },
                            "camera": {
                                "fov": 90.0,
                                "resolution": [64, 64],
                                "max_depth": 10.0
                            }
                        }
                    }
                ],
                "objectives": [
                    {
                        "type": "explore_regions",
                        "regions": regions,
                        "coverage_threshold": 0.8,
                        "weight": 0.4
                    },
                    {
                        "type": "find_resources",
                        "num_resources": num_resources,
                        "weight": 0.6
                    }
                ],
                "vision_required": True,
                "fog_of_war": fog_of_war,
                "regions": regions
            }
        }
        
        # Add resources and obstacles
        task["parameters"]["objects"].extend(resources)
        task["parameters"]["objects"].extend(obstacles)
        
        return task
        
    def _generate_regions(self, num_regions: int) -> List[Dict[str, Any]]:
        """Generate exploration regions."""
        regions = []
        workspace = self.params.workspace_size
        min_size = workspace * 0.2
        max_size = workspace * 0.4
        
        for i in range(num_regions):
            # Generate region with random position and size
            position = np.array([
                random.uniform(-workspace/3, workspace/3),
                random.uniform(-workspace/3, workspace/3),
                0
            ])
            
            size = np.array([
                random.uniform(min_size, max_size),
                random.uniform(min_size, max_size),
                random.uniform(1.0, 3.0)
            ])
            
            regions.append({
                "id": f"region_{i}",
                "position": position.tolist(),
                "size": size.tolist(),
                "importance": random.uniform(0.5, 1.0),
                "explored": False,
                "points_of_interest": []
            })
            
        return regions
        
    def _generate_resources(
        self,
        num_resources: int,
        regions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate resources within regions."""
        resources = []
        resource_types = ["energy", "data", "material"]
        
        for i in range(num_resources):
            # Select random region
            region = random.choice(regions)
            region_pos = np.array(region["position"])
            region_size = np.array(region["size"])
            
            # Generate position within region bounds
            position = np.array([
                random.uniform(
                    region_pos[0] - region_size[0]/2,
                    region_pos[0] + region_size[0]/2
                ),
                random.uniform(
                    region_pos[1] - region_size[1]/2,
                    region_pos[1] + region_size[1]/2
                ),
                0.5
            ])
            
            # Create resource
            resource_type = random.choice(resource_types)
            resources.append({
                "id": f"resource_{i}",
                "type": "RESOURCE",
                "resource_type": resource_type,
                "position": position.tolist(),
                "rotation": [1, 0, 0, 0],
                "scale": [0.3, 0.3, 0.3],
                "value": random.uniform(0.5, 1.0),
                "discovered": False,
                "color": {
                    "energy": [1.0, 0.8, 0.0],
                    "data": [0.0, 0.8, 1.0],
                    "material": [0.8, 0.4, 0.0]
                }[resource_type]
            })
            
            # Add as point of interest to region
            region["points_of_interest"].append({
                "id": f"resource_{i}",
                "type": "resource",
                "discovered": False
            })
            
        return resources
        
    def _generate_obstacles(
        self,
        num_obstacles: int,
        regions: List[Dict[str, Any]],
        resources: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate obstacles avoiding regions and resources."""
        obstacles = []
        workspace = self.params.workspace_size
        min_distance = workspace * 0.1
        
        for i in range(num_obstacles):
            while True:
                # Generate potential obstacle position
                position = np.array([
                    random.uniform(-workspace/2, workspace/2),
                    random.uniform(-workspace/2, workspace/2),
                    random.uniform(0, workspace/4)
                ])
                
                # Check distance from regions and resources
                valid = True
                
                # Check regions
                for region in regions:
                    region_pos = np.array(region["position"])
                    if np.linalg.norm(position[:2] - region_pos[:2]) < min_distance:
                        valid = False
                        break
                
                # Check resources
                if valid:
                    for resource in resources:
                        resource_pos = np.array(resource["position"])
                        if np.linalg.norm(position[:2] - resource_pos[:2]) < min_distance:
                            valid = False
                            break
                
                if valid:
                    break
            
            # Create obstacle with random shape
            shape = random.choice(["cube", "cylinder", "sphere"])
            if shape == "cube":
                scale = [
                    random.uniform(0.5, 2.0),
                    random.uniform(0.5, 2.0),
                    random.uniform(1.0, 3.0)
                ]
            elif shape == "cylinder":
                radius = random.uniform(0.3, 1.0)
                scale = [radius, radius, random.uniform(1.0, 3.0)]
            else:  # sphere
                radius = random.uniform(0.5, 1.5)
                scale = [radius, radius, radius]
            
            obstacles.append({
                "id": f"obstacle_{i}",
                "type": "OBSTACLE",
                "shape": shape,
                "position": position.tolist(),
                "rotation": [1, 0, 0, 0],
                "scale": scale,
                "color": [0.5, 0.5, 0.5]
            })
            
        return obstacles

def create_task(
    task_type: TaskType,
    params: Optional[TaskParameters] = None
) -> Dict[str, Any]:
    """Create a task of specified type with given parameters."""
    if params is None:
        params = TaskParameters()
        
    generators = {
        TaskType.NAVIGATION: NavigationTaskGenerator,
        TaskType.MANIPULATION: ManipulationTaskGenerator,
        TaskType.BALANCE: BalanceTaskGenerator,
        TaskType.CONSTRUCTION: ConstructionTaskGenerator,
        TaskType.MULTI_AGENT: MultiAgentTaskGenerator,
        TaskType.EXPLORATION: ExplorationTaskGenerator
    }
    
    if task_type not in generators:
        raise ValueError(f"Unsupported task type: {task_type}")
        
    generator = generators[task_type](params)
    return generator.generate()

def load_task(file_path: str) -> Dict[str, Any]:
    """Load task specification from file."""
    with open(file_path, 'r') as f:
        return json.load(f)
        
def save_task(task: Dict[str, Any], file_path: str) -> None:
    """Save task specification to file."""
    with open(file_path, 'w') as f:
        json.dump(task, f, indent=2) 