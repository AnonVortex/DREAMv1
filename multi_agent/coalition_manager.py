"""
Coalition Formation and Management System for Multi-Agent Coordination
"""

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any
import numpy as np
from datetime import datetime
import logging
import torch
import torch.nn.functional as F
from .role_manager import RoleType
from .task_allocator import TaskRequirements, Task
from ..utils.metrics import MetricsTracker
import uuid

class CoalitionStatus(Enum):
    """Status of a coalition in the system."""
    FORMING = auto()    # Coalition is being formed
    ACTIVE = auto()     # Coalition is actively working
    DISSOLVING = auto() # Coalition is being dissolved
    DISSOLVED = auto()  # Coalition has been disbanded

@dataclass
class CoalitionMetrics:
    """Metrics for coalition performance."""
    task_completion_rate: float
    communication_efficiency: float
    resource_utilization: float
    coordination_score: float
    avg_reward: float
    stability_score: float

@dataclass
class Coalition:
    """Represents a coalition of agents."""
    id: str
    members: Set[str]
    task_id: str
    formation_time: float
    status: CoalitionStatus = CoalitionStatus.FORMING
    performance_history: List[Dict[str, float]] = field(default_factory=list)
    communication_history: List[Dict[str, float]] = field(default_factory=list)
    reward_history: List[Dict[str, float]] = field(default_factory=list)

class CoalitionManager:
    """Manages coalition formation and coordination."""
    
    def __init__(self, num_agents: int, min_coalition_size: int = 2):
        self.logger = logging.getLogger(__name__)
        self.num_agents = num_agents
        self.min_coalition_size = min_coalition_size
        
        # Coalition management
        self.coalitions: Dict[str, Coalition] = {}
        self.agent_coalitions: Dict[str, str] = {}
        self.coalition_history: List[Dict[str, Any]] = []
        
        # Performance tracking
        self.performance_metrics = MetricsTracker()
        
    async def form_coalition(
        self,
        task: Dict[str, Any],
        available_agents: Dict[str, Dict[str, Any]],
        agent_roles: Dict[str, str]
    ) -> Optional[str]:
        """Form a coalition for a given task."""
        try:
            # Generate coalition ID
            coalition_id = str(uuid.uuid4())
            
            # Select agents based on task requirements and agent capabilities
            selected_agents = self._select_agents_for_task(
                task=task,
                available_agents=available_agents,
                agent_roles=agent_roles
            )
            
            if not selected_agents:
                logger.warning(f"No suitable agents found for task {task['id']}")
                return None
            
            # Create coalition
            coalition = Coalition(
                id=coalition_id,
                members=selected_agents,
                task_id=task["id"],
                formation_time=task.get("creation_time", 0.0)
            )
            
            # Update coalition and agent mappings
            self.coalitions[coalition_id] = coalition
            for agent_id in selected_agents:
                self.agent_coalitions[agent_id] = coalition_id
            
            # Record formation in history
            self.coalition_history.append({
                "event": "formation",
                "coalition_id": coalition_id,
                "task_id": task["id"],
                "members": list(selected_agents),
                "time": coalition.formation_time
            })
            
            # Update metrics
            self.performance_metrics.update({
                "coalitions_formed": 1,
                "avg_coalition_size": len(selected_agents)
            })
            
            logger.info(
                f"Formed coalition {coalition_id} for task {task['id']} "
                f"with {len(selected_agents)} agents"
            )
            
            return coalition_id
            
        except Exception as e:
            logger.error(f"Error forming coalition: {str(e)}")
            return None
    
    def _select_agents_for_task(
        self,
        task: Dict[str, Any],
        available_agents: Dict[str, Dict[str, Any]],
        agent_roles: Dict[str, str]
    ) -> Set[str]:
        """Select agents for a task based on requirements and capabilities."""
        selected_agents = set()
        
        # Get task requirements
        required_capabilities = task.get("required_capabilities", set())
        min_agents = task.get("min_agents", 1)
        max_agents = task.get("max_agents", len(available_agents))
        
        # Score each available agent
        agent_scores = {}
        for agent_id, capabilities in available_agents.items():
            # Skip agents already in coalitions
            if agent_id in self.agent_coalitions:
                continue
                
            # Calculate capability match score
            capability_score = len(
                set(capabilities.keys()) & required_capabilities
            ) / len(required_capabilities) if required_capabilities else 1.0
            
            # Consider agent role
            role_score = 1.0
            if task.get("preferred_roles"):
                role_score = 1.5 if agent_roles[agent_id] in task["preferred_roles"] else 0.5
            
            # Calculate final score
            agent_scores[agent_id] = capability_score * role_score
        
        # Select top scoring agents within min/max bounds
        sorted_agents = sorted(
            agent_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        for agent_id, score in sorted_agents:
            if len(selected_agents) >= max_agents:
                break
            if score > 0:
                selected_agents.add(agent_id)
        
        # Ensure minimum number of agents
        if len(selected_agents) < min_agents:
            logger.warning(
                f"Could not find enough qualified agents for task {task['id']}. "
                f"Required: {min_agents}, Found: {len(selected_agents)}"
            )
            return set()
        
        return selected_agents
    
    async def update_coalition_performance(
        self,
        coalition_id: str,
        task_metrics: Dict[str, float],
        communication_stats: Dict[str, float],
        agent_rewards: Dict[str, float]
    ) -> None:
        """Update performance metrics for a coalition."""
        try:
            coalition = self.coalitions.get(coalition_id)
            if not coalition:
                logger.warning(f"Coalition {coalition_id} not found")
                return
            
            # Update histories
            coalition.performance_history.append(task_metrics)
            coalition.communication_history.append(communication_stats)
            coalition.reward_history.append(agent_rewards)
            
            # Calculate aggregate metrics
            avg_reward = np.mean(list(agent_rewards.values()))
            reward_std = np.std(list(agent_rewards.values()))
            
            # Update performance metrics
            self.performance_metrics.update({
                "avg_coalition_reward": avg_reward,
                "reward_std": reward_std,
                "task_success_rate": task_metrics.get("success_rate", 0.0),
                "communication_efficiency": communication_stats.get("efficiency", 1.0)
            })
            
        except Exception as e:
            logger.error(f"Error updating coalition performance: {str(e)}")
    
    async def dissolve_coalition(
        self,
        coalition_id: str,
        reason: str = "completed"
    ) -> bool:
        """Dissolve a coalition and free its agents."""
        try:
            coalition = self.coalitions.get(coalition_id)
            if not coalition:
                logger.warning(f"Coalition {coalition_id} not found")
                return False
            
            # Update coalition status
            coalition.status = CoalitionStatus.DISSOLVING
            
            # Remove agent mappings
            for agent_id in coalition.members:
                self.agent_coalitions.pop(agent_id, None)
            
            # Update coalition status and history
            coalition.status = CoalitionStatus.DISSOLVED
            self.coalition_history.append({
                "event": "dissolution",
                "coalition_id": coalition_id,
                "reason": reason,
                "members": list(coalition.members),
                "time": coalition.formation_time  # TODO: Add actual dissolution time
            })
            
            # Remove coalition
            self.coalitions.pop(coalition_id, None)
            
            # Update metrics
            self.performance_metrics.update({
                "coalitions_dissolved": 1
            })
            
            logger.info(f"Dissolved coalition {coalition_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error dissolving coalition: {str(e)}")
            return False
    
    def get_coalition_metrics(self) -> Dict[str, Dict[str, float]]:
        """Get performance metrics for all active coalitions."""
        metrics = {}
        
        for coalition_id, coalition in self.coalitions.items():
            if coalition.status == CoalitionStatus.ACTIVE:
                # Calculate coalition-specific metrics
                avg_performance = np.mean([
                    list(m.values())
                    for m in coalition.performance_history[-5:]  # Last 5 updates
                ], axis=0)
                
                avg_reward = np.mean([
                    np.mean(list(r.values()))
                    for r in coalition.reward_history[-5:]
                ]) if coalition.reward_history else 0.0
                
                metrics[coalition_id] = {
                    "avg_reward": avg_reward,
                    "num_members": len(coalition.members),
                    "avg_performance": float(avg_performance),
                    "lifetime": len(coalition.performance_history)
                }
        
        return metrics
    
    def get_agent_coalition(self, agent_id: str) -> Optional[str]:
        """Get the coalition ID for an agent."""
        return self.agent_coalitions.get(agent_id)
    
    def get_coalition_members(self, coalition_id: str) -> Optional[Set[str]]:
        """Get the members of a coalition."""
        coalition = self.coalitions.get(coalition_id)
        return coalition.members if coalition else None 