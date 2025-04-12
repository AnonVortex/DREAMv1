from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import logging
from dataclasses import dataclass
from enum import Enum, auto
import gym
import pybullet as p
import pybullet_data

logger = logging.getLogger(__name__)

class PhysicsBackend(Enum):
    """Supported physics simulation backends."""
    PYBULLET = auto()
    MUJOCO = auto()
    CUSTOM = auto()

@dataclass
class EnvironmentConfig:
    """Configuration for the environment."""
    physics_backend: PhysicsBackend = PhysicsBackend.PYBULLET
    render_mode: str = "rgb_array"
    max_steps: int = 1000
    time_step: float = 1/240.0  # Physics simulation timestep
    gravity: Tuple[float, float, float] = (0, 0, -9.81)
    enable_gui: bool = False
    camera_distance: float = 2.0
    camera_yaw: float = 45.0
    camera_pitch: float = -30.0
    
    # Multi-agent settings
    max_agents: int = 10
    agent_collision_radius: float = 0.5
    communication_range: float = 5.0
    
    # Task settings
    task_difficulty: float = 0.5
    dynamic_difficulty: bool = True
    reward_scale: float = 1.0

class EnvironmentManager:
    """Manages the simulation environment and physics."""
    
    def __init__(self, config: Optional[EnvironmentConfig] = None):
        """Initialize the environment manager."""
        self.config = config or EnvironmentConfig()
        self.physics_client = None
        self.agent_bodies: Dict[str, int] = {}  # Maps agent IDs to physics body IDs
        self.objects: Dict[str, int] = {}  # Maps object IDs to physics body IDs
        self.step_counter = 0
        self.is_initialized = False
        
        # Initialize physics simulation
        self._initialize_physics()
        logger.info("Environment manager initialized with %s backend", 
                   self.config.physics_backend.name)
    
    def _initialize_physics(self) -> None:
        """Initialize the physics simulation backend."""
        try:
            if self.config.physics_backend == PhysicsBackend.PYBULLET:
                connection_mode = p.GUI if self.config.enable_gui else p.DIRECT
                self.physics_client = p.connect(connection_mode)
                p.setAdditionalSearchPath(pybullet_data.getDataPath())
                p.setGravity(*self.config.gravity)
                p.setTimeStep(self.config.time_step)
                
                # Load ground plane
                p.loadURDF("plane.urdf")
                
                # Set camera position
                p.resetDebugVisualizerCamera(
                    self.config.camera_distance,
                    self.config.camera_yaw,
                    self.config.camera_pitch,
                    [0, 0, 0]
                )
                
            elif self.config.physics_backend == PhysicsBackend.MUJOCO:
                # TODO: Implement MuJoCo backend
                raise NotImplementedError("MuJoCo backend not yet implemented")
            
            elif self.config.physics_backend == PhysicsBackend.CUSTOM:
                # TODO: Implement custom physics backend
                raise NotImplementedError("Custom physics backend not yet implemented")
            
            self.is_initialized = True
            
        except Exception as e:
            logger.error("Failed to initialize physics backend: %s", str(e))
            raise
    
    def add_agent(self, agent_id: str, position: Tuple[float, float, float]) -> None:
        """Add an agent to the environment."""
        try:
            if agent_id in self.agent_bodies:
                logger.warning("Agent %s already exists in environment", agent_id)
                return
            
            # Create visual shape for the agent
            visual_shape_id = p.createVisualShape(
                shapeType=p.GEOM_SPHERE,
                radius=self.config.agent_collision_radius,
                rgbaColor=[0.8, 0.3, 0.3, 1]
            )
            
            # Create collision shape
            collision_shape_id = p.createCollisionShape(
                shapeType=p.GEOM_SPHERE,
                radius=self.config.agent_collision_radius
            )
            
            # Create multi-body
            body_id = p.createMultiBody(
                baseMass=1.0,
                baseCollisionShapeIndex=collision_shape_id,
                baseVisualShapeIndex=visual_shape_id,
                basePosition=position
            )
            
            self.agent_bodies[agent_id] = body_id
            logger.info("Added agent %s to environment at position %s", 
                       agent_id, position)
            
        except Exception as e:
            logger.error("Failed to add agent %s: %s", agent_id, str(e))
            raise
    
    def remove_agent(self, agent_id: str) -> None:
        """Remove an agent from the environment."""
        try:
            if agent_id not in self.agent_bodies:
                logger.warning("Agent %s does not exist in environment", agent_id)
                return
            
            p.removeBody(self.agent_bodies[agent_id])
            del self.agent_bodies[agent_id]
            logger.info("Removed agent %s from environment", agent_id)
            
        except Exception as e:
            logger.error("Failed to remove agent %s: %s", agent_id, str(e))
            raise
    
    def add_object(
        self,
        object_id: str,
        urdf_path: str,
        position: Tuple[float, float, float],
        orientation: Optional[Tuple[float, float, float, float]] = None
    ) -> None:
        """Add an object to the environment using its URDF description."""
        try:
            if object_id in self.objects:
                logger.warning("Object %s already exists in environment", object_id)
                return
            
            orientation = orientation or (0, 0, 0, 1)  # Default to no rotation
            body_id = p.loadURDF(
                urdf_path,
                position,
                orientation,
                useFixedBase=True
            )
            
            self.objects[object_id] = body_id
            logger.info("Added object %s to environment at position %s", 
                       object_id, position)
            
        except Exception as e:
            logger.error("Failed to add object %s: %s", object_id, str(e))
            raise
    
    def remove_object(self, object_id: str) -> None:
        """Remove an object from the environment."""
        try:
            if object_id not in self.objects:
                logger.warning("Object %s does not exist in environment", object_id)
                return
            
            p.removeBody(self.objects[object_id])
            del self.objects[object_id]
            logger.info("Removed object %s from environment", object_id)
            
        except Exception as e:
            logger.error("Failed to remove object %s: %s", object_id, str(e))
            raise
    
    def get_agent_state(self, agent_id: str) -> Dict[str, Any]:
        """Get the current state of an agent."""
        try:
            if agent_id not in self.agent_bodies:
                raise ValueError(f"Agent {agent_id} does not exist in environment")
            
            body_id = self.agent_bodies[agent_id]
            position, orientation = p.getBasePositionAndOrientation(body_id)
            velocity, angular_velocity = p.getBaseVelocity(body_id)
            
            return {
                "position": position,
                "orientation": orientation,
                "velocity": velocity,
                "angular_velocity": angular_velocity
            }
            
        except Exception as e:
            logger.error("Failed to get state for agent %s: %s", agent_id, str(e))
            raise
    
    def apply_action(self, agent_id: str, action: np.ndarray) -> None:
        """Apply an action to an agent."""
        try:
            if agent_id not in self.agent_bodies:
                raise ValueError(f"Agent {agent_id} does not exist in environment")
            
            body_id = self.agent_bodies[agent_id]
            
            # Assuming action is [force_x, force_y, force_z, torque_x, torque_y, torque_z]
            force = action[:3]
            torque = action[3:] if len(action) > 3 else [0, 0, 0]
            
            p.applyExternalForce(
                body_id,
                -1,  # Apply to base
                force,
                [0, 0, 0],  # Apply at center of mass
                p.WORLD_FRAME
            )
            
            p.applyExternalTorque(
                body_id,
                -1,  # Apply to base
                torque,
                p.WORLD_FRAME
            )
            
        except Exception as e:
            logger.error("Failed to apply action for agent %s: %s", agent_id, str(e))
            raise
    
    def get_observation(self) -> Dict[str, Any]:
        """Get the current observation of the environment."""
        try:
            observation = {
                "agents": {},
                "objects": {},
                "time": self.step_counter * self.config.time_step
            }
            
            # Get agent states
            for agent_id in self.agent_bodies:
                observation["agents"][agent_id] = self.get_agent_state(agent_id)
            
            # Get object states
            for object_id, body_id in self.objects.items():
                position, orientation = p.getBasePositionAndOrientation(body_id)
                observation["objects"][object_id] = {
                    "position": position,
                    "orientation": orientation
                }
            
            return observation
            
        except Exception as e:
            logger.error("Failed to get observation: %s", str(e))
            raise
    
    def step(self) -> Dict[str, Any]:
        """Step the physics simulation forward."""
        try:
            p.stepSimulation()
            self.step_counter += 1
            
            # Get current observation
            observation = self.get_observation()
            
            # Check for collisions
            collision_pairs = []
            for i in range(p.getNumBodies()):
                for contact in p.getContactPoints(i):
                    body1, body2 = contact[1], contact[2]
                    collision_pairs.append((body1, body2))
            
            # Update observation with collision information
            observation["collisions"] = collision_pairs
            
            return observation
            
        except Exception as e:
            logger.error("Failed to step environment: %s", str(e))
            raise
    
    def reset(self) -> Dict[str, Any]:
        """Reset the environment to initial state."""
        try:
            # Reset simulation
            p.resetSimulation()
            
            # Re-initialize physics
            self._initialize_physics()
            
            # Reset internal state
            self.step_counter = 0
            self.agent_bodies.clear()
            self.objects.clear()
            
            return self.get_observation()
            
        except Exception as e:
            logger.error("Failed to reset environment: %s", str(e))
            raise
    
    def render(self, mode: str = "rgb_array") -> np.ndarray:
        """Render the environment."""
        try:
            if mode == "rgb_array":
                view_matrix = p.computeViewMatrixFromYawPitchRoll(
                    cameraTargetPosition=[0, 0, 0],
                    distance=self.config.camera_distance,
                    yaw=self.config.camera_yaw,
                    pitch=self.config.camera_pitch,
                    roll=0,
                    upAxisIndex=2
                )
                
                proj_matrix = p.computeProjectionMatrixFOV(
                    fov=60,
                    aspect=1.0,
                    nearVal=0.1,
                    farVal=100.0
                )
                
                width, height = 320, 240
                _, _, rgba, _, _ = p.getCameraImage(
                    width,
                    height,
                    view_matrix,
                    proj_matrix
                )
                
                return np.array(rgba)
            
            else:
                raise ValueError(f"Unsupported render mode: {mode}")
            
        except Exception as e:
            logger.error("Failed to render environment: %s", str(e))
            raise
    
    def close(self) -> None:
        """Clean up resources."""
        try:
            if self.physics_client is not None:
                p.disconnect(self.physics_client)
                self.physics_client = None
                self.is_initialized = False
            
        except Exception as e:
            logger.error("Failed to close environment: %s", str(e))
            raise 