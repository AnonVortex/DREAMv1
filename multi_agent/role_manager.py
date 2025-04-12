"""
Role Management System for Multi-Agent Coordination
"""

from enum import Enum, auto
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import torch
from datetime import datetime
import logging

class RoleType(Enum):
    """Available agent roles in the system."""
    COORDINATOR = auto()  # Manages team coordination
    EXPLORER = auto()     # Explores environment
    SPECIALIST = auto()   # Handles specialized tasks
    SUPPORTER = auto()    # Supports other agents
    LEARNER = auto()      # Focuses on learning new skills

@dataclass
class RoleSpec:
    """Specification for a role."""
    type: RoleType
    requirements: Dict[str, float]  # Skill requirements
    priorities: Dict[str, float]    # Task type priorities
    communication_priority: float    # Priority in communication
    max_agents: int                 # Maximum agents in this role

@dataclass
class RoleAssignment:
    """Represents a role assignment to an agent."""
    agent_id: int
    role_type: RoleType
    start_time: datetime
    performance_metrics: Dict[str, float]
    confidence: float  # Confidence in role suitability

class RoleManager:
    """Manages dynamic role assignment and transitions."""
    
    def __init__(self, num_agents: int):
        self.num_agents = num_agents
        self.logger = logging.getLogger(__name__)
        
        # Initialize role specifications
        self.role_specs: Dict[RoleType, RoleSpec] = self._initialize_role_specs()
        
        # Current role assignments
        self.role_assignments: Dict[int, RoleAssignment] = {}
        
        # Role transition history
        self.transition_history: List[Tuple[int, RoleType, RoleType, datetime]] = []
        
        # Performance tracking
        self.role_performance: Dict[RoleType, List[float]] = {
            role: [] for role in RoleType
        }
        
    def _initialize_role_specs(self) -> Dict[RoleType, RoleSpec]:
        """Initialize specifications for each role."""
        return {
            RoleType.COORDINATOR: RoleSpec(
                type=RoleType.COORDINATOR,
                requirements={"leadership": 0.7, "communication": 0.8},
                priorities={"coordination": 1.0, "planning": 0.9},
                communication_priority=1.0,
                max_agents=max(1, self.num_agents // 5)
            ),
            RoleType.EXPLORER: RoleSpec(
                type=RoleType.EXPLORER,
                requirements={"mobility": 0.7, "perception": 0.8},
                priorities={"exploration": 1.0, "mapping": 0.8},
                communication_priority=0.7,
                max_agents=max(1, self.num_agents // 3)
            ),
            RoleType.SPECIALIST: RoleSpec(
                type=RoleType.SPECIALIST,
                requirements={"skill_mastery": 0.8, "adaptability": 0.7},
                priorities={"specialized_tasks": 1.0, "skill_sharing": 0.7},
                communication_priority=0.8,
                max_agents=max(1, self.num_agents // 4)
            ),
            RoleType.SUPPORTER: RoleSpec(
                type=RoleType.SUPPORTER,
                requirements={"cooperation": 0.8, "resource_management": 0.7},
                priorities={"support": 1.0, "resource_sharing": 0.9},
                communication_priority=0.9,
                max_agents=max(1, self.num_agents // 4)
            ),
            RoleType.LEARNER: RoleSpec(
                type=RoleType.LEARNER,
                requirements={"learning_rate": 0.7, "exploration": 0.8},
                priorities={"skill_acquisition": 1.0, "knowledge_sharing": 0.8},
                communication_priority=0.6,
                max_agents=max(1, self.num_agents // 5)
            )
        }

    async def assign_initial_roles(self, agent_capabilities: Dict[int, Dict[str, float]]) -> Dict[int, RoleType]:
        """Assign initial roles to agents based on their capabilities."""
        assignments = {}
        role_counts = {role: 0 for role in RoleType}
        
        # Sort agents by their average capability
        sorted_agents = sorted(
            agent_capabilities.items(),
            key=lambda x: sum(x[1].values()) / len(x[1]),
            reverse=True
        )
        
        for agent_id, capabilities in sorted_agents:
            best_role = None
            best_score = float('-inf')
            
            for role_type, spec in self.role_specs.items():
                if role_counts[role_type] >= spec.max_agents:
                    continue
                    
                # Calculate role suitability score
                score = self._calculate_role_suitability(capabilities, spec)
                
                if score > best_score:
                    best_score = score
                    best_role = role_type
            
            if best_role is not None:
                assignments[agent_id] = best_role
                role_counts[best_role] += 1
                
                # Record assignment
                self.role_assignments[agent_id] = RoleAssignment(
                    agent_id=agent_id,
                    role_type=best_role,
                    start_time=datetime.now(),
                    performance_metrics={},
                    confidence=best_score
                )
        
        return assignments

    def _calculate_role_suitability(self, capabilities: Dict[str, float], role_spec: RoleSpec) -> float:
        """Calculate how suitable an agent is for a role."""
        score = 0.0
        total_weight = 0.0
        
        # Check requirements
        for req, threshold in role_spec.requirements.items():
            if req in capabilities:
                weight = 1.0
                total_weight += weight
                score += weight * max(0, capabilities[req] - threshold + 0.2)
        
        # Check priorities alignment
        for priority, importance in role_spec.priorities.items():
            if priority in capabilities:
                weight = importance
                total_weight += weight
                score += weight * capabilities[priority]
        
        return score / total_weight if total_weight > 0 else 0.0

    async def update_role_performance(
        self,
        agent_id: int,
        metrics: Dict[str, float]
    ) -> Optional[RoleType]:
        """Update role performance and potentially trigger role transition."""
        if agent_id not in self.role_assignments:
            return None
            
        current_role = self.role_assignments[agent_id]
        
        # Update performance metrics
        current_role.performance_metrics.update(metrics)
        
        # Calculate overall performance score
        performance_score = sum(metrics.values()) / len(metrics)
        self.role_performance[current_role.role_type].append(performance_score)
        
        # Check for role transition
        if len(self.role_performance[current_role.role_type]) >= 10:
            avg_performance = np.mean(self.role_performance[current_role.role_type][-10:])
            
            # Consider role transition if performance is poor
            if avg_performance < 0.5 and current_role.confidence < 0.7:
                return await self.transition_role(agent_id, metrics)
        
        return None

    async def transition_role(
        self,
        agent_id: int,
        current_metrics: Dict[str, float]
    ) -> Optional[RoleType]:
        """Transition an agent to a new role based on performance."""
        if agent_id not in self.role_assignments:
            return None
            
        current_role = self.role_assignments[agent_id]
        role_counts = self._get_role_counts()
        
        best_new_role = None
        best_score = float('-inf')
        
        # Evaluate each possible role
        for role_type, spec in self.role_specs.items():
            if role_type == current_role.role_type:
                continue
                
            if role_counts[role_type] >= spec.max_agents:
                continue
                
            # Calculate suitability for new role
            score = self._calculate_role_suitability(current_metrics, spec)
            
            # Add transition bonus for complementary roles
            score += self._calculate_transition_bonus(current_role.role_type, role_type)
            
            if score > best_score:
                best_score = score
                best_new_role = role_type
        
        if best_new_role and best_score > current_role.confidence:
            # Record transition
            self.transition_history.append((
                agent_id,
                current_role.role_type,
                best_new_role,
                datetime.now()
            ))
            
            # Update assignment
            self.role_assignments[agent_id] = RoleAssignment(
                agent_id=agent_id,
                role_type=best_new_role,
                start_time=datetime.now(),
                performance_metrics=current_metrics,
                confidence=best_score
            )
            
            return best_new_role
        
        return None

    def _calculate_transition_bonus(self, current_role: RoleType, new_role: RoleType) -> float:
        """Calculate bonus score for role transitions that benefit the team."""
        transition_bonuses = {
            (RoleType.EXPLORER, RoleType.SPECIALIST): 0.2,  # Explorer gained specific knowledge
            (RoleType.LEARNER, RoleType.SPECIALIST): 0.3,   # Learner mastered skills
            (RoleType.SUPPORTER, RoleType.COORDINATOR): 0.2, # Supporter gained leadership
            (RoleType.SPECIALIST, RoleType.COORDINATOR): 0.1 # Specialist can guide others
        }
        return transition_bonuses.get((current_role, new_role), 0.0)

    def _get_role_counts(self) -> Dict[RoleType, int]:
        """Get current count of agents in each role."""
        counts = {role: 0 for role in RoleType}
        for assignment in self.role_assignments.values():
            counts[assignment.role_type] += 1
        return counts

    def get_role_metrics(self) -> Dict[str, Dict[str, float]]:
        """Get performance metrics for each role."""
        metrics = {}
        for role_type in RoleType:
            if self.role_performance[role_type]:
                metrics[role_type.name] = {
                    "avg_performance": np.mean(self.role_performance[role_type]),
                    "recent_performance": np.mean(self.role_performance[role_type][-10:]),
                    "num_agents": self._get_role_counts()[role_type],
                    "transition_rate": len([t for t in self.transition_history 
                                         if t[1] == role_type or t[2] == role_type]) / max(1, len(self.transition_history))
                }
        return metrics

    def get_agent_role_info(self, agent_id: int) -> Optional[Dict[str, Any]]:
        """Get detailed role information for an agent."""
        if agent_id not in self.role_assignments:
            return None
            
        assignment = self.role_assignments[agent_id]
        spec = self.role_specs[assignment.role_type]
        
        return {
            "role": assignment.role_type.name,
            "duration": (datetime.now() - assignment.start_time).total_seconds(),
            "confidence": assignment.confidence,
            "performance_metrics": assignment.performance_metrics,
            "requirements": spec.requirements,
            "priorities": spec.priorities,
            "communication_priority": spec.communication_priority
        } 