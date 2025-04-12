"""
Dynamic Task Generation System for Coalition-based Training
"""

from enum import Enum, auto
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Any
import numpy as np
import logging
from datetime import datetime
import uuid

from .coalition_manager import Coalition, CoalitionStatus
from .role_manager import RoleType

logger = logging.getLogger(__name__)

class TaskDifficulty(Enum):
    """Difficulty levels for generated tasks."""
    BEGINNER = auto()
    INTERMEDIATE = auto()
    ADVANCED = auto()
    EXPERT = auto()

@dataclass
class TaskParameters:
    """Parameters defining a task's characteristics."""
    difficulty: TaskDifficulty
    min_agents: int
    max_agents: int
    required_roles: Set[RoleType]
    required_capabilities: Dict[str, float]
    time_limit: float
    resource_requirements: Dict[str, float]
    success_criteria: Dict[str, float]

class TaskGenerator:
    """Generates tasks suitable for coalition-based training."""
    
    def __init__(
        self,
        base_difficulty: TaskDifficulty = TaskDifficulty.BEGINNER,
        difficulty_scaling_factor: float = 1.2,
        min_task_duration: float = 100.0,
        max_task_duration: float = 1000.0
    ):
        self.logger = logging.getLogger(__name__)
        self.base_difficulty = base_difficulty
        self.difficulty_scaling_factor = difficulty_scaling_factor
        self.min_task_duration = min_task_duration
        self.max_task_duration = max_task_duration
        
        # Task history and metrics
        self.task_history: List[Dict[str, Any]] = []
        self.coalition_performance: Dict[str, List[float]] = {}
        
    def generate_task(
        self,
        coalition: Optional[Coalition] = None,
        performance_history: Optional[List[Dict[str, float]]] = None
    ) -> Dict[str, Any]:
        """Generate a task suitable for the given coalition."""
        try:
            # Determine task difficulty
            difficulty = self._determine_task_difficulty(
                coalition,
                performance_history
            )
            
            # Calculate base parameters
            base_params = self._calculate_base_parameters(difficulty)
            
            # Adjust for coalition if provided
            if coalition:
                base_params = self._adjust_for_coalition(
                    base_params,
                    coalition,
                    performance_history
                )
            
            # Generate task configuration
            task_config = {
                "id": str(uuid.uuid4()),
                "creation_time": datetime.now().timestamp(),
                "difficulty": difficulty.name,
                "parameters": base_params,
                "coalition_id": coalition.id if coalition else None,
                "status": "created"
            }
            
            # Add to history
            self.task_history.append(task_config)
            
            return task_config
            
        except Exception as e:
            self.logger.error(f"Error generating task: {str(e)}")
            return self._generate_fallback_task()
    
    def _determine_task_difficulty(
        self,
        coalition: Optional[Coalition],
        performance_history: Optional[List[Dict[str, float]]]
    ) -> TaskDifficulty:
        """Determine appropriate task difficulty based on coalition performance."""
        if not coalition or not performance_history:
            return self.base_difficulty
            
        # Calculate recent performance metrics
        recent_performance = performance_history[-5:] if len(performance_history) > 5 else performance_history
        avg_success_rate = np.mean([p.get("success_rate", 0.0) for p in recent_performance])
        avg_completion_time = np.mean([p.get("completion_time", float("inf")) for p in recent_performance])
        
        # Adjust difficulty based on performance
        if avg_success_rate > 0.8 and avg_completion_time < self.min_task_duration:
            # Increase difficulty
            difficulties = list(TaskDifficulty)
            current_idx = difficulties.index(self.base_difficulty)
            if current_idx < len(difficulties) - 1:
                return difficulties[current_idx + 1]
        elif avg_success_rate < 0.3:
            # Decrease difficulty
            difficulties = list(TaskDifficulty)
            current_idx = difficulties.index(self.base_difficulty)
            if current_idx > 0:
                return difficulties[current_idx - 1]
        
        return self.base_difficulty
    
    def _calculate_base_parameters(
        self,
        difficulty: TaskDifficulty
    ) -> Dict[str, Any]:
        """Calculate base task parameters for given difficulty."""
        # Scale factors based on difficulty
        difficulty_scales = {
            TaskDifficulty.BEGINNER: 1.0,
            TaskDifficulty.INTERMEDIATE: 1.5,
            TaskDifficulty.ADVANCED: 2.0,
            TaskDifficulty.EXPERT: 3.0
        }
        scale = difficulty_scales[difficulty]
        
        # Generate base parameters
        return {
            "min_agents": max(1, int(2 * scale)),
            "max_agents": max(2, int(4 * scale)),
            "required_roles": {
                RoleType.LEARNER.name,
                RoleType.EXPLORER.name if scale > 1.0 else None,
                RoleType.SPECIALIST.name if scale > 2.0 else None
            } - {None},
            "required_capabilities": {
                "learning_rate": 0.3 * scale,
                "adaptability": 0.2 * scale,
                "skill_mastery": 0.4 * scale,
                "coordination": 0.3 * scale if scale > 1.0 else 0.0
            },
            "time_limit": min(
                self.max_task_duration,
                self.min_task_duration * scale
            ),
            "resource_requirements": {
                "computation": 100 * scale,
                "memory": 50 * scale,
                "bandwidth": 30 * scale if scale > 1.0 else 10
            },
            "success_criteria": {
                "min_reward": 10 * scale,
                "min_success_rate": max(0.6, 0.8 - 0.1 * scale),
                "max_completion_time": self.max_task_duration
            }
        }
    
    def _adjust_for_coalition(
        self,
        base_params: Dict[str, Any],
        coalition: Coalition,
        performance_history: Optional[List[Dict[str, float]]]
    ) -> Dict[str, Any]:
        """Adjust task parameters based on coalition characteristics."""
        if not performance_history:
            return base_params
            
        # Calculate coalition performance metrics
        recent_performance = performance_history[-5:] if len(performance_history) > 5 else performance_history
        avg_performance = np.mean([
            list(p.values())
            for p in recent_performance
        ], axis=0)
        
        # Adjust agent requirements
        base_params["min_agents"] = min(
            base_params["min_agents"],
            len(coalition.members)
        )
        base_params["max_agents"] = min(
            base_params["max_agents"],
            len(coalition.members)
        )
        
        # Adjust time limit based on past performance
        avg_completion_time = np.mean([
            p.get("completion_time", self.max_task_duration)
            for p in recent_performance
        ])
        base_params["time_limit"] = min(
            self.max_task_duration,
            avg_completion_time * self.difficulty_scaling_factor
        )
        
        # Adjust success criteria
        avg_reward = np.mean([
            np.mean(list(p.get("agent_rewards", {0: 0.0}).values()))
            for p in recent_performance
        ])
        base_params["success_criteria"]["min_reward"] = max(
            1.0,
            avg_reward * self.difficulty_scaling_factor
        )
        
        return base_params
    
    def _generate_fallback_task(self) -> Dict[str, Any]:
        """Generate a simple fallback task when normal generation fails."""
        return {
            "id": str(uuid.uuid4()),
            "creation_time": datetime.now().timestamp(),
            "difficulty": TaskDifficulty.BEGINNER.name,
            "parameters": {
                "min_agents": 1,
                "max_agents": 2,
                "required_roles": {RoleType.LEARNER.name},
                "required_capabilities": {
                    "learning_rate": 0.2,
                    "adaptability": 0.2,
                    "skill_mastery": 0.2
                },
                "time_limit": self.min_task_duration,
                "resource_requirements": {
                    "computation": 50,
                    "memory": 25,
                    "bandwidth": 10
                },
                "success_criteria": {
                    "min_reward": 5.0,
                    "min_success_rate": 0.6,
                    "max_completion_time": self.max_task_duration
                }
            },
            "coalition_id": None,
            "status": "created"
        }
    
    def update_task_metrics(
        self,
        task_id: str,
        coalition_id: str,
        metrics: Dict[str, float]
    ) -> None:
        """Update task completion metrics."""
        # Store coalition performance
        if coalition_id not in self.coalition_performance:
            self.coalition_performance[coalition_id] = []
        self.coalition_performance[coalition_id].append(metrics)
        
        # Update task history
        for task in self.task_history:
            if task["id"] == task_id:
                task["completion_metrics"] = metrics
                task["status"] = "completed"
                break
    
    def get_coalition_difficulty_level(
        self,
        coalition_id: str
    ) -> TaskDifficulty:
        """Get the current difficulty level for a coalition."""
        if coalition_id not in self.coalition_performance:
            return self.base_difficulty
            
        recent_performance = self.coalition_performance[coalition_id][-5:]
        avg_success_rate = np.mean([
            p.get("success_rate", 0.0)
            for p in recent_performance
        ])
        
        # Determine difficulty based on success rate
        if avg_success_rate > 0.8:
            return TaskDifficulty.EXPERT
        elif avg_success_rate > 0.6:
            return TaskDifficulty.ADVANCED
        elif avg_success_rate > 0.4:
            return TaskDifficulty.INTERMEDIATE
        else:
            return TaskDifficulty.BEGINNER
    
    def get_task_history(
        self,
        coalition_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get task history, optionally filtered by coalition."""
        if not coalition_id:
            return self.task_history
            
        return [
            task for task in self.task_history
            if task.get("coalition_id") == coalition_id
        ] 