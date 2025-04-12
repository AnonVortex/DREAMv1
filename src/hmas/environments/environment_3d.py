"""3D Environment implementation for H-MAS.

This module implements a 3D environment for spatial reasoning and physical interaction tasks.
It supports physics simulation, collision detection, and various 3D object interactions.
"""

from typing import Dict, List, Any, Optional, Tuple, Union, Set
import numpy as np
from dataclasses import dataclass
from pathlib import Path
import json
import logging
from enum import Enum, auto
import quaternion  # For 3D rotations
import trimesh  # For 3D mesh handling
import pybullet as p  # For physics simulation
import pybullet_data

from ..environments.base_environment import BaseEnvironment, EnvironmentConfig

class ObjectType(Enum):
    """Types of 3D objects in the environment."""
    STATIC = auto()      # Immovable objects like walls
    DYNAMIC = auto()     # Movable objects with physics
    AGENT = auto()       # Agent bodies
    TOOL = auto()        # Interactive tools
    TARGET = auto()      # Goal objects
    OBSTACLE = auto()    # Blocking objects

@dataclass
class Object3D:
    """Representation of a 3D object in the environment."""
    id: str
    type: ObjectType
    position: np.ndarray  # [x, y, z]
    rotation: np.ndarray  # Quaternion [w, x, y, z]
    scale: np.ndarray    # [sx, sy, sz]
    mesh_path: Optional[str] = None
    mass: float = 1.0
    friction: float = 0.5
    restitution: float = 0.5
    bullet_id: Optional[int] = None  # PyBullet body ID

@dataclass
class Environment3DConfig(EnvironmentConfig):
    """Configuration for 3D environment."""
    gravity: np.ndarray = np.array([0, 0, -9.81])
    physics_timestep: float = 1/240.0
    max_objects: int = 100
    render_width: int = 640
    render_height: int = 480
    camera_position: np.ndarray = np.array([2, 2, 2])
    camera_target: np.ndarray = np.array([0, 0, 0])
    enable_shadows: bool = True
    mesh_directory: str = "assets/meshes"

class Environment3D(BaseEnvironment):
    """3D environment with physics simulation."""
    
    def __init__(self, config: Environment3DConfig):
        """Initialize 3D environment."""
        super().__init__(config)
        self.config: Environment3DConfig = config
        
        # Initialize physics
        self._init_physics_engine()
        
        # Object management
        self.objects: Dict[str, Object3D] = {}
        self.agent_ids: Set[str] = set()
        
        # Camera setup
        self.camera_position = config.camera_position.copy()
        self.camera_target = config.camera_target.copy()
        self._update_camera()
        
    def _init_physics_engine(self) -> None:
        """Initialize PyBullet physics engine."""
        # Start physics client
        p.connect(p.DIRECT)  # or p.GUI for visualization
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        # Set physics parameters
        p.setGravity(*self.config.gravity)
        p.setTimeStep(self.config.physics_timestep)
        p.setRealTimeSimulation(0)  # We'll step manually
        
        if self.config.enable_shadows:
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
            
    def _create_initial_state(self) -> Dict[str, Any]:
        """Create initial environment state."""
        # Reset physics engine
        p.resetSimulation()
        p.setGravity(*self.config.gravity)
        
        # Load ground plane
        p.loadURDF("plane.urdf")
        
        # Clear object tracking
        self.objects.clear()
        self.agent_ids.clear()
        
        # Create initial state dict
        state = {
            "objects": {},
            "agents": {},
            "time": 0.0,
            "physics_steps": 0
        }
        
        # Add initial objects based on task
        if self.current_task:
            self._setup_task_objects(self.current_task)
            
        # Update state with object information
        for obj_id, obj in self.objects.items():
            state["objects"][obj_id] = {
                "position": obj.position.tolist(),
                "rotation": obj.rotation.tolist(),
                "type": obj.type.name
            }
            
        return state
        
    async def _compute_next_state(
        self,
        action: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compute next state based on action."""
        # Apply agent actions
        for agent_id, agent_action in action.items():
            if agent_id in self.agent_ids:
                self._apply_agent_action(agent_id, agent_action)
                
        # Step physics simulation
        p.stepSimulation()
        
        # Update object states
        next_state = {
            "objects": {},
            "agents": {},
            "time": self.current_state["time"] + self.config.physics_timestep,
            "physics_steps": self.current_state["physics_steps"] + 1
        }
        
        for obj_id, obj in self.objects.items():
            if obj.bullet_id is not None:
                # Get updated position and rotation from physics
                pos, rot = p.getBasePositionAndOrientation(obj.bullet_id)
                obj.position = np.array(pos)
                obj.rotation = np.array(rot)
                
            next_state["objects"][obj_id] = {
                "position": obj.position.tolist(),
                "rotation": obj.rotation.tolist(),
                "type": obj.type.name
            }
            
            if obj_id in self.agent_ids:
                next_state["agents"][obj_id] = next_state["objects"][obj_id]
                
        return next_state
        
    def _compute_base_reward(
        self,
        state: Dict[str, Any],
        action: Dict[str, Any],
        next_state: Dict[str, Any]
    ) -> float:
        """Compute base reward for transition."""
        reward = 0.0
        
        if self.current_task and "objectives" in self.current_task["parameters"]:
            for objective in self.current_task["parameters"]["objectives"]:
                if objective["type"] == "reach_target":
                    # Check distance to target
                    agent_pos = np.array(
                        next_state["agents"][objective["agent_id"]]["position"]
                    )
                    target_pos = np.array(
                        next_state["objects"][objective["target_id"]]["position"]
                    )
                    distance = np.linalg.norm(agent_pos - target_pos)
                    reward -= distance * objective.get("weight", 1.0)
                    
                elif objective["type"] == "maintain_balance":
                    # Check object orientation
                    obj = self.objects[objective["object_id"]]
                    up_vector = quaternion.rotate_vectors(
                        obj.rotation,
                        np.array([0, 0, 1])
                    )
                    tilt = np.arccos(np.dot(up_vector, [0, 0, 1]))
                    reward -= tilt * objective.get("weight", 1.0)
                    
        return reward
        
    def _apply_agent_action(
        self,
        agent_id: str,
        action: Dict[str, Any]
    ) -> None:
        """Apply agent action to physics simulation."""
        if agent_id not in self.objects:
            return
            
        obj = self.objects[agent_id]
        if obj.bullet_id is None:
            return
            
        if "force" in action:
            force = np.array(action["force"])
            p.applyExternalForce(
                obj.bullet_id,
                -1,  # Link ID (-1 for base)
                force,
                obj.position,  # Point of application
                p.WORLD_FRAME
            )
            
        if "torque" in action:
            torque = np.array(action["torque"])
            p.applyExternalTorque(
                obj.bullet_id,
                -1,  # Link ID
                torque,
                p.WORLD_FRAME
            )
            
    def _setup_task_objects(self, task: Dict[str, Any]) -> None:
        """Setup objects for the current task."""
        if "objects" in task["parameters"]:
            for obj_spec in task["parameters"]["objects"]:
                self.spawn_object(obj_spec)
                
    def spawn_object(
        self,
        spec: Dict[str, Any]
    ) -> Optional[str]:
        """Spawn a new object in the environment."""
        if len(self.objects) >= self.config.max_objects:
            self.logger.warning("Maximum object limit reached")
            return None
            
        obj_id = spec.get("id", str(len(self.objects)))
        obj_type = ObjectType[spec["type"]]
        position = np.array(spec.get("position", [0, 0, 0]))
        rotation = np.array(spec.get("rotation", [1, 0, 0, 0]))
        scale = np.array(spec.get("scale", [1, 1, 1]))
        
        # Create object
        obj = Object3D(
            id=obj_id,
            type=obj_type,
            position=position,
            rotation=rotation,
            scale=scale,
            mesh_path=spec.get("mesh_path"),
            mass=spec.get("mass", 1.0),
            friction=spec.get("friction", 0.5),
            restitution=spec.get("restitution", 0.5)
        )
        
        # Load into physics engine
        if obj.mesh_path:
            mesh_path = Path(self.config.mesh_directory) / obj.mesh_path
            if mesh_path.exists():
                # Load mesh
                mesh = trimesh.load(str(mesh_path))
                # Convert to convex hull for physics
                vertices = mesh.vertices * scale
                indices = mesh.faces
                
                # Create collision shape
                collision_shape = p.createCollisionShape(
                    p.GEOM_MESH,
                    vertices=vertices,
                    indices=indices,
                    meshScale=[1, 1, 1]
                )
            else:
                # Fallback to basic shape
                collision_shape = p.createCollisionShape(
                    p.GEOM_BOX,
                    halfExtents=scale/2
                )
        else:
            # Use basic shape
            collision_shape = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=scale/2
            )
            
        # Create body
        obj.bullet_id = p.createMultiBody(
            baseMass=obj.mass if obj_type != ObjectType.STATIC else 0,
            baseCollisionShapeIndex=collision_shape,
            basePosition=position,
            baseOrientation=rotation,
            baseInertialFramePosition=[0, 0, 0],
            baseInertialFrameOrientation=[0, 0, 0, 1]
        )
        
        # Set dynamics properties
        p.changeDynamics(
            obj.bullet_id,
            -1,  # Link ID
            lateralFriction=obj.friction,
            restitution=obj.restitution
        )
        
        # Store object
        self.objects[obj_id] = obj
        if obj_type == ObjectType.AGENT:
            self.agent_ids.add(obj_id)
            
        return obj_id
        
    def remove_object(self, obj_id: str) -> bool:
        """Remove an object from the environment."""
        if obj_id not in self.objects:
            return False
            
        obj = self.objects[obj_id]
        if obj.bullet_id is not None:
            p.removeBody(obj.bullet_id)
            
        del self.objects[obj_id]
        self.agent_ids.discard(obj_id)
        return True
        
    def _update_camera(self) -> None:
        """Update camera view matrix."""
        p.resetDebugVisualizerCamera(
            cameraDistance=np.linalg.norm(
                self.camera_position - self.camera_target
            ),
            cameraYaw=0,
            cameraPitch=-45,
            cameraTargetPosition=self.camera_target
        )
        
    def get_camera_image(
        self,
        width: Optional[int] = None,
        height: Optional[int] = None
    ) -> np.ndarray:
        """Get RGB image from camera."""
        width = width or self.config.render_width
        height = height or self.config.render_height
        
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=self.camera_position,
            cameraTargetPosition=self.camera_target,
            cameraUpVector=[0, 0, 1]
        )
        
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=60,
            aspect=width/height,
            nearVal=0.1,
            farVal=100.0
        )
        
        # Get image from physics engine
        _, _, rgba, _, _ = p.getCameraImage(
            width=width,
            height=height,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL
        )
        
        return rgba[..., :3]  # Return RGB only
        
    async def close(self) -> None:
        """Clean up environment resources."""
        await super().close()
        p.disconnect() 