from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

class DifficultyLevel(Enum):
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"

@dataclass
class TaskParameters:
    workspace_size: Tuple[float, float, float]
    num_obstacles: int
    time_limit: float
    vision_required: bool
    tool_use_allowed: bool
    cooperation_required: bool
    min_success_rate: float
    min_episodes: int

class CurriculumStage:
    def __init__(
        self,
        difficulty: DifficultyLevel,
        task_params: TaskParameters,
        completion_criteria: Dict[str, float]
    ):
        self.difficulty = difficulty
        self.task_params = task_params
        self.completion_criteria = completion_criteria
        self.episodes_completed = 0
        self.success_rate = 0.0
        self.metrics_history: List[Dict[str, float]] = []
    
    def update_metrics(self, metrics: Dict[str, float]):
        """Update stage metrics with new episode results."""
        self.episodes_completed += 1
        self.metrics_history.append(metrics)
        
        # Calculate success rate over last N episodes
        recent_successes = [m.get("task_success", 0) for m in self.metrics_history[-50:]]
        self.success_rate = np.mean(recent_successes) if recent_successes else 0.0
    
    def is_completed(self) -> bool:
        """Check if stage completion criteria are met."""
        if self.episodes_completed < self.task_params.min_episodes:
            return False
            
        if self.success_rate < self.task_params.min_success_rate:
            return False
            
        # Check additional completion criteria
        for metric, threshold in self.completion_criteria.items():
            recent_values = [m.get(metric, 0) for m in self.metrics_history[-50:]]
            if not recent_values or np.mean(recent_values) < threshold:
                return False
        
        return True

class CurriculumManager:
    """Manages curriculum learning with progressive difficulty adjustment."""
    
    def __init__(self):
        self.stages: Dict[DifficultyLevel, CurriculumStage] = {}
        self.current_difficulty = DifficultyLevel.BEGINNER
        self.initialize_curriculum()
    
    def initialize_curriculum(self):
        """Initialize curriculum stages with progressive difficulty."""
        # Beginner stage
        self.stages[DifficultyLevel.BEGINNER] = CurriculumStage(
            difficulty=DifficultyLevel.BEGINNER,
            task_params=TaskParameters(
                workspace_size=(5.0, 5.0, 3.0),
                num_obstacles=3,
                time_limit=300,
                vision_required=False,
                tool_use_allowed=False,
                cooperation_required=False,
                min_success_rate=0.7,
                min_episodes=100
            ),
            completion_criteria={
                "average_reward": 50.0,
                "exploration_coverage": 0.6,
                "collision_rate": 0.2
            }
        )
        
        # Intermediate stage
        self.stages[DifficultyLevel.INTERMEDIATE] = CurriculumStage(
            difficulty=DifficultyLevel.INTERMEDIATE,
            task_params=TaskParameters(
                workspace_size=(8.0, 8.0, 4.0),
                num_obstacles=5,
                time_limit=250,
                vision_required=True,
                tool_use_allowed=False,
                cooperation_required=False,
                min_success_rate=0.6,
                min_episodes=150
            ),
            completion_criteria={
                "average_reward": 75.0,
                "exploration_coverage": 0.7,
                "collision_rate": 0.15
            }
        )
        
        # Advanced stage
        self.stages[DifficultyLevel.ADVANCED] = CurriculumStage(
            difficulty=DifficultyLevel.ADVANCED,
            task_params=TaskParameters(
                workspace_size=(10.0, 10.0, 5.0),
                num_obstacles=8,
                time_limit=200,
                vision_required=True,
                tool_use_allowed=True,
                cooperation_required=False,
                min_success_rate=0.5,
                min_episodes=200
            ),
            completion_criteria={
                "average_reward": 100.0,
                "exploration_coverage": 0.8,
                "collision_rate": 0.1,
                "tool_use_efficiency": 0.6
            }
        )
        
        # Expert stage
        self.stages[DifficultyLevel.EXPERT] = CurriculumStage(
            difficulty=DifficultyLevel.EXPERT,
            task_params=TaskParameters(
                workspace_size=(15.0, 15.0, 6.0),
                num_obstacles=12,
                time_limit=180,
                vision_required=True,
                tool_use_allowed=True,
                cooperation_required=True,
                min_success_rate=0.4,
                min_episodes=250
            ),
            completion_criteria={
                "average_reward": 150.0,
                "exploration_coverage": 0.9,
                "collision_rate": 0.05,
                "tool_use_efficiency": 0.8,
                "cooperation_score": 0.7
            }
        )
    
    def get_current_parameters(self) -> TaskParameters:
        """Get parameters for current difficulty level."""
        return self.stages[self.current_difficulty].task_params
    
    def update_progress(self, metrics: Dict[str, float]) -> bool:
        """Update progress and check for difficulty advancement."""
        current_stage = self.stages[self.current_difficulty]
        current_stage.update_metrics(metrics)
        
        # Check if current stage is completed
        if current_stage.is_completed():
            # Try to advance to next difficulty
            if self._advance_difficulty():
                logger.info(f"Advanced to difficulty level: {self.current_difficulty.value}")
                return True
        
        return False
    
    def _advance_difficulty(self) -> bool:
        """Attempt to advance to next difficulty level."""
        difficulty_order = list(DifficultyLevel)
        current_index = difficulty_order.index(self.current_difficulty)
        
        # Check if we're already at maximum difficulty
        if current_index == len(difficulty_order) - 1:
            return False
        
        # Advance to next difficulty
        self.current_difficulty = difficulty_order[current_index + 1]
        return True
    
    def get_progress_metrics(self) -> Dict[str, Any]:
        """Get detailed progress metrics for current stage."""
        current_stage = self.stages[self.current_difficulty]
        return {
            "current_difficulty": self.current_difficulty.value,
            "episodes_completed": current_stage.episodes_completed,
            "success_rate": current_stage.success_rate,
            "completion_criteria": current_stage.completion_criteria,
            "recent_metrics": current_stage.metrics_history[-10:] if current_stage.metrics_history else []
        }
    
    def save_state(self) -> Dict[str, Any]:
        """Save curriculum state for checkpointing."""
        return {
            "current_difficulty": self.current_difficulty.value,
            "stages": {
                level.value: {
                    "episodes_completed": stage.episodes_completed,
                    "success_rate": stage.success_rate,
                    "metrics_history": stage.metrics_history
                }
                for level, stage in self.stages.items()
            }
        }
    
    def load_state(self, state: Dict[str, Any]):
        """Load curriculum state from checkpoint."""
        self.current_difficulty = DifficultyLevel(state["current_difficulty"])
        for level_str, stage_data in state["stages"].items():
            level = DifficultyLevel(level_str)
            if level in self.stages:
                stage = self.stages[level]
                stage.episodes_completed = stage_data["episodes_completed"]
                stage.success_rate = stage_data["success_rate"]
                stage.metrics_history = stage_data["metrics_history"]
        
        logger.info(f"Loaded curriculum state: {self.current_difficulty.value}") 