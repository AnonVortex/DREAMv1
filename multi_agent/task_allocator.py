"""
Task Allocation System for Multi-Agent Coordination
"""

from enum import Enum, auto
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple, Any
import numpy as np
from datetime import datetime
import logging
from .role_manager import RoleType

class TaskPriority(Enum):
    """Priority levels for tasks."""
    LOW = auto()
    MEDIUM = auto()
    HIGH = auto()
    CRITICAL = auto()

class TaskStatus(Enum):
    """Status of a task in the system."""
    PENDING = auto()    # Not yet assigned
    ASSIGNED = auto()   # Assigned but not started
    IN_PROGRESS = auto() # Currently being executed
    COMPLETED = auto()   # Successfully completed
    FAILED = auto()      # Failed to complete
    BLOCKED = auto()     # Waiting for dependencies

@dataclass
class TaskRequirements:
    """Requirements for task execution."""
    min_skill_level: float
    required_capabilities: Dict[str, float]
    preferred_role: Optional[RoleType]
    min_agents: int
    max_agents: int
    estimated_duration: float
    resource_requirements: Dict[str, float]

@dataclass
class Task:
    """Represents a task in the system."""
    task_id: str
    parent_id: Optional[str]
    priority: TaskPriority
    requirements: TaskRequirements
    dependencies: Set[str]
    subtasks: List[str]
    assigned_agents: Set[int]
    status: TaskStatus
    progress: float
    start_time: Optional[datetime]
    completion_time: Optional[datetime]
    metrics: Dict[str, float]

class TaskAllocator:
    """Manages task decomposition and allocation."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Task management
        self.tasks: Dict[str, Task] = {}
        self.task_queue: List[str] = []
        self.agent_assignments: Dict[int, Set[str]] = {}
        
        # Performance tracking
        self.completion_history: List[Tuple[str, datetime, float]] = []
        self.failure_history: List[Tuple[str, datetime, str]] = []
        
    async def create_task(
        self,
        task_id: str,
        priority: TaskPriority,
        requirements: TaskRequirements,
        parent_id: Optional[str] = None,
        dependencies: Optional[Set[str]] = None
    ) -> Task:
        """Create a new task in the system."""
        task = Task(
            task_id=task_id,
            parent_id=parent_id,
            priority=priority,
            requirements=requirements,
            dependencies=dependencies or set(),
            subtasks=[],
            assigned_agents=set(),
            status=TaskStatus.PENDING,
            progress=0.0,
            start_time=None,
            completion_time=None,
            metrics={}
        )
        
        self.tasks[task_id] = task
        self._update_task_queue()
        
        return task

    async def decompose_task(
        self,
        task_id: str,
        subtask_specs: List[Dict[str, Any]]
    ) -> List[str]:
        """Decompose a task into subtasks."""
        if task_id not in self.tasks:
            raise ValueError(f"Task {task_id} not found")
            
        parent_task = self.tasks[task_id]
        subtask_ids = []
        
        for spec in subtask_specs:
            subtask_id = f"{task_id}_sub_{len(parent_task.subtasks)}"
            
            # Create subtask with inherited properties
            requirements = TaskRequirements(
                min_skill_level=spec.get("min_skill_level", parent_task.requirements.min_skill_level),
                required_capabilities=spec.get("required_capabilities", {}),
                preferred_role=spec.get("preferred_role", parent_task.requirements.preferred_role),
                min_agents=spec.get("min_agents", 1),
                max_agents=spec.get("max_agents", parent_task.requirements.max_agents),
                estimated_duration=spec.get("estimated_duration", parent_task.requirements.estimated_duration / len(subtask_specs)),
                resource_requirements=spec.get("resource_requirements", {})
            )
            
            await self.create_task(
                task_id=subtask_id,
                priority=parent_task.priority,
                requirements=requirements,
                parent_id=task_id,
                dependencies=set(spec.get("dependencies", []))
            )
            
            subtask_ids.append(subtask_id)
            parent_task.subtasks.append(subtask_id)
        
        return subtask_ids

    async def assign_task(
        self,
        task_id: str,
        agent_ids: Set[int],
        agent_capabilities: Dict[int, Dict[str, float]],
        agent_roles: Dict[int, RoleType]
    ) -> bool:
        """Assign a task to a set of agents."""
        if task_id not in self.tasks:
            return False
            
        task = self.tasks[task_id]
        
        # Check if task can be assigned
        if task.status != TaskStatus.PENDING:
            return False
            
        if len(agent_ids) < task.requirements.min_agents:
            return False
            
        # Verify agent capabilities
        for agent_id in agent_ids:
            if not self._verify_agent_capabilities(
                agent_id,
                agent_capabilities[agent_id],
                agent_roles[agent_id],
                task.requirements
            ):
                return False
        
        # Check task dependencies
        if not self._are_dependencies_met(task_id):
            task.status = TaskStatus.BLOCKED
            return False
        
        # Assign task
        task.assigned_agents = agent_ids
        task.status = TaskStatus.ASSIGNED
        task.start_time = datetime.now()
        
        # Update agent assignments
        for agent_id in agent_ids:
            if agent_id not in self.agent_assignments:
                self.agent_assignments[agent_id] = set()
            self.agent_assignments[agent_id].add(task_id)
        
        self._update_task_queue()
        return True

    def _verify_agent_capabilities(
        self,
        agent_id: int,
        capabilities: Dict[str, float],
        role: RoleType,
        requirements: TaskRequirements
    ) -> bool:
        """Verify if an agent meets task requirements."""
        # Check skill level
        if sum(capabilities.values()) / len(capabilities) < requirements.min_skill_level:
            return False
        
        # Check required capabilities
        for cap, required_level in requirements.required_capabilities.items():
            if cap not in capabilities or capabilities[cap] < required_level:
                return False
        
        # Check preferred role
        if requirements.preferred_role and role != requirements.preferred_role:
            return False
        
        return True

    def _are_dependencies_met(self, task_id: str) -> bool:
        """Check if all dependencies of a task are completed."""
        task = self.tasks[task_id]
        return all(
            self.tasks[dep_id].status == TaskStatus.COMPLETED
            for dep_id in task.dependencies
        )

    async def update_task_progress(
        self,
        task_id: str,
        progress: float,
        metrics: Optional[Dict[str, float]] = None
    ) -> None:
        """Update task progress and metrics."""
        if task_id not in self.tasks:
            return
            
        task = self.tasks[task_id]
        task.progress = progress
        
        if metrics:
            task.metrics.update(metrics)
        
        # Check for task completion
        if progress >= 1.0:
            await self.complete_task(task_id)
        elif progress < 0:
            await self.fail_task(task_id, "Negative progress reported")

    async def complete_task(self, task_id: str) -> None:
        """Mark a task as completed."""
        if task_id not in self.tasks:
            return
            
        task = self.tasks[task_id]
        task.status = TaskStatus.COMPLETED
        task.completion_time = datetime.now()
        
        # Record completion
        self.completion_history.append((
            task_id,
            task.completion_time,
            (task.completion_time - task.start_time).total_seconds()
            if task.start_time else 0
        ))
        
        # Release assigned agents
        for agent_id in task.assigned_agents:
            if agent_id in self.agent_assignments:
                self.agent_assignments[agent_id].remove(task_id)
        
        # Update parent task progress
        if task.parent_id:
            parent = self.tasks[task.parent_id]
            completed_subtasks = sum(
                1 for subtask_id in parent.subtasks
                if self.tasks[subtask_id].status == TaskStatus.COMPLETED
            )
            parent_progress = completed_subtasks / len(parent.subtasks)
            await self.update_task_progress(task.parent_id, parent_progress)
        
        self._update_task_queue()

    async def fail_task(self, task_id: str, reason: str) -> None:
        """Mark a task as failed."""
        if task_id not in self.tasks:
            return
            
        task = self.tasks[task_id]
        task.status = TaskStatus.FAILED
        
        # Record failure
        self.failure_history.append((
            task_id,
            datetime.now(),
            reason
        ))
        
        # Release assigned agents
        for agent_id in task.assigned_agents:
            if agent_id in self.agent_assignments:
                self.agent_assignments[agent_id].remove(task_id)
        
        self._update_task_queue()

    def _update_task_queue(self) -> None:
        """Update the priority queue of pending tasks."""
        self.task_queue = [
            task_id for task_id, task in self.tasks.items()
            if task.status == TaskStatus.PENDING
        ]
        
        # Sort by priority and dependencies
        self.task_queue.sort(
            key=lambda x: (
                -self.tasks[x].priority.value,  # Higher priority first
                len(self.tasks[x].dependencies),  # Fewer dependencies first
                self.tasks[x].requirements.estimated_duration  # Shorter tasks first
            )
        )

    def get_available_tasks(self, agent_id: int) -> List[str]:
        """Get list of tasks available for an agent."""
        return [
            task_id for task_id in self.task_queue
            if len(self.tasks[task_id].assigned_agents) < self.tasks[task_id].requirements.max_agents
            and self._are_dependencies_met(task_id)
        ]

    def get_task_metrics(self) -> Dict[str, float]:
        """Get overall task allocation metrics."""
        total_tasks = len(self.tasks)
        if total_tasks == 0:
            return {}
            
        completed_tasks = sum(
            1 for task in self.tasks.values()
            if task.status == TaskStatus.COMPLETED
        )
        
        failed_tasks = sum(
            1 for task in self.tasks.values()
            if task.status == TaskStatus.FAILED
        )
        
        avg_completion_time = (
            np.mean([duration for _, _, duration in self.completion_history])
            if self.completion_history else 0
        )
        
        return {
            "total_tasks": total_tasks,
            "completion_rate": completed_tasks / total_tasks,
            "failure_rate": failed_tasks / total_tasks,
            "avg_completion_time": avg_completion_time,
            "pending_tasks": len(self.task_queue),
            "blocked_tasks": sum(
                1 for task in self.tasks.values()
                if task.status == TaskStatus.BLOCKED
            )
        } 