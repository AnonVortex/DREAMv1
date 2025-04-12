"""
Coalition Learning Manager for Multi-Agent Systems
Handles coalition-specific learning strategies and knowledge transfer
"""

from enum import Enum, auto
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
import logging
import json

from ..multi_agent.coalition_manager import Coalition, CoalitionStatus
from .learning_agent import LearningAgent
from ..utils.metrics import MetricsTracker

logger = logging.getLogger(__name__)

class LearningStrategy(Enum):
    """Types of learning strategies for coalitions."""
    INDIVIDUAL = auto()      # Each agent learns independently
    CENTRALIZED = auto()     # Centralized learning with shared policy
    HIERARCHICAL = auto()    # Hierarchical learning with role-based policies
    FEDERATED = auto()       # Federated learning across coalition members
    ENSEMBLE = auto()        # Ensemble of policies

@dataclass
class LearningConfig:
    """Configuration for coalition learning."""
    strategy: LearningStrategy = LearningStrategy.CENTRALIZED
    learning_rate: float = 0.001
    batch_size: int = 64
    update_frequency: int = 100
    knowledge_share_threshold: float = 0.7
    adaptation_rate: float = 0.1
    max_policy_cache: int = 10
    performance_window: int = 100

class PolicyCache:
    """Cache for storing and managing coalition policies."""
    def __init__(self, max_size: int = 10):
        self.max_size = max_size
        self.policies: Dict[str, Dict[str, Any]] = {}
        self.performance_history: Dict[str, List[float]] = {}
        
    def add_policy(
        self,
        policy_id: str,
        state_dict: Dict[str, torch.Tensor],
        performance: float,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Add a policy to the cache."""
        if len(self.policies) >= self.max_size:
            # Remove worst performing policy
            worst_policy = min(
                self.performance_history.items(),
                key=lambda x: np.mean(x[1])
            )[0]
            del self.policies[worst_policy]
            del self.performance_history[worst_policy]
        
        self.policies[policy_id] = {
            "state_dict": state_dict,
            "timestamp": datetime.now().timestamp(),
            "metadata": metadata or {}
        }
        
        if policy_id not in self.performance_history:
            self.performance_history[policy_id] = []
        self.performance_history[policy_id].append(performance)
    
    def get_best_policy(self) -> Optional[Tuple[str, Dict[str, Any]]]:
        """Get the best performing policy."""
        if not self.policies:
            return None
            
        best_policy = max(
            self.performance_history.items(),
            key=lambda x: np.mean(x[1])
        )[0]
        
        return best_policy, self.policies[best_policy]

class CoalitionLearningManager:
    """Manages learning strategies and knowledge transfer for coalitions."""
    
    def __init__(
        self,
        config: Optional[LearningConfig] = None
    ):
        self.config = config or LearningConfig()
        self.metrics = MetricsTracker()
        
        # Learning state
        self.coalition_policies: Dict[str, PolicyCache] = {}
        self.learning_progress: Dict[str, Dict[str, Any]] = {}
        self.knowledge_base: Dict[str, Dict[str, Any]] = {}
        
        # Performance tracking
        self.performance_history: Dict[str, List[Dict[str, float]]] = {}
        self.adaptation_history: Dict[str, List[Dict[str, Any]]] = {}
    
    async def initialize_coalition_learning(
        self,
        coalition: Coalition,
        learning_agents: Dict[str, LearningAgent],
        initial_policy: Optional[Dict[str, Any]] = None
    ) -> None:
        """Initialize learning for a new coalition."""
        try:
            coalition_id = coalition.id
            
            # Create policy cache for coalition
            self.coalition_policies[coalition_id] = PolicyCache(
                max_size=self.config.max_policy_cache
            )
            
            # Initialize learning progress tracking
            self.learning_progress[coalition_id] = {
                "episodes_trained": 0,
                "current_performance": 0.0,
                "best_performance": float('-inf'),
                "last_update": datetime.now().timestamp()
            }
            
            # Initialize knowledge base
            self.knowledge_base[coalition_id] = {
                "task_experiences": {},
                "successful_strategies": {},
                "failure_cases": {},
                "adaptation_history": []
            }
            
            # Set initial policy if provided
            if initial_policy:
                await self.update_coalition_policy(
                    coalition_id=coalition_id,
                    policy_state=initial_policy,
                    performance=0.0,
                    metadata={"source": "initialization"}
                )
            
            # Initialize performance history
            self.performance_history[coalition_id] = []
            
            logger.info(f"Initialized learning for coalition {coalition_id}")
            
        except Exception as e:
            logger.error(f"Error initializing coalition learning: {str(e)}")
            raise
    
    async def update_coalition_policy(
        self,
        coalition_id: str,
        policy_state: Dict[str, Any],
        performance: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Update the policy for a coalition."""
        try:
            # Add policy to cache
            policy_id = f"policy_{datetime.now().timestamp()}"
            self.coalition_policies[coalition_id].add_policy(
                policy_id=policy_id,
                state_dict=policy_state,
                performance=performance,
                metadata=metadata
            )
            
            # Update learning progress
            self.learning_progress[coalition_id]["episodes_trained"] += 1
            self.learning_progress[coalition_id]["current_performance"] = performance
            self.learning_progress[coalition_id]["last_update"] = datetime.now().timestamp()
            
            if performance > self.learning_progress[coalition_id]["best_performance"]:
                self.learning_progress[coalition_id]["best_performance"] = performance
            
            # Update metrics
            self.metrics.update({
                f"coalition_{coalition_id}_performance": performance,
                f"coalition_{coalition_id}_policies": len(self.coalition_policies[coalition_id].policies)
            })
            
        except Exception as e:
            logger.error(f"Error updating coalition policy: {str(e)}")
            raise
    
    async def transfer_knowledge(
        self,
        source_coalition: str,
        target_coalition: str,
        task_type: str,
        adaptation_steps: int = 100
    ) -> bool:
        """Transfer knowledge between coalitions."""
        try:
            # Get source coalition's best policy
            source_policy = self.coalition_policies[source_coalition].get_best_policy()
            if not source_policy:
                logger.warning(f"No policy found for source coalition {source_coalition}")
                return False
            
            policy_id, policy_data = source_policy
            
            # Check if knowledge transfer is beneficial
            source_performance = np.mean(
                self.coalition_policies[source_coalition].performance_history[policy_id]
            )
            if source_performance < self.config.knowledge_share_threshold:
                logger.info(
                    f"Source coalition {source_coalition} performance below threshold "
                    f"for knowledge transfer"
                )
                return False
            
            # Adapt policy for target coalition
            adapted_policy = await self._adapt_policy(
                source_policy=policy_data["state_dict"],
                target_coalition=target_coalition,
                adaptation_steps=adaptation_steps
            )
            
            # Update target coalition's policy
            await self.update_coalition_policy(
                coalition_id=target_coalition,
                policy_state=adapted_policy,
                performance=source_performance * self.config.adaptation_rate,
                metadata={
                    "source_coalition": source_coalition,
                    "source_policy": policy_id,
                    "task_type": task_type
                }
            )
            
            # Record adaptation
            self.adaptation_history.setdefault(target_coalition, []).append({
                "timestamp": datetime.now().timestamp(),
                "source_coalition": source_coalition,
                "task_type": task_type,
                "source_performance": source_performance
            })
            
            logger.info(
                f"Successfully transferred knowledge from coalition {source_coalition} "
                f"to coalition {target_coalition}"
            )
            return True
            
        except Exception as e:
            logger.error(f"Error transferring knowledge: {str(e)}")
            return False
    
    async def _adapt_policy(
        self,
        source_policy: Dict[str, torch.Tensor],
        target_coalition: str,
        adaptation_steps: int
    ) -> Dict[str, torch.Tensor]:
        """Adapt a policy for a target coalition."""
        try:
            adapted_policy = {}
            
            # Get target coalition's characteristics
            target_info = self.knowledge_base[target_coalition]
            
            for param_name, param_tensor in source_policy.items():
                # Create adaptation mask based on layer importance
                adaptation_mask = self._create_adaptation_mask(
                    param_tensor.shape,
                    target_info
                )
                
                # Apply progressive adaptation
                adapted_tensor = param_tensor.clone()
                for step in range(adaptation_steps):
                    adaptation_rate = self.config.adaptation_rate * (
                        1 - step / adaptation_steps
                    )
                    noise = torch.randn_like(param_tensor) * adaptation_rate
                    adapted_tensor = adapted_tensor * (1 - adaptation_mask) + (
                        adapted_tensor + noise
                    ) * adaptation_mask
                
                adapted_policy[param_name] = adapted_tensor
            
            return adapted_policy
            
        except Exception as e:
            logger.error(f"Error adapting policy: {str(e)}")
            raise
    
    def _create_adaptation_mask(
        self,
        tensor_shape: Tuple[int, ...],
        target_info: Dict[str, Any]
    ) -> torch.Tensor:
        """Create an adaptation mask for policy transfer."""
        # Create base mask
        mask = torch.ones(tensor_shape)
        
        # Adjust mask based on target coalition's characteristics
        if "task_experiences" in target_info:
            experience_factor = len(target_info["task_experiences"]) / 100
            mask *= experience_factor
        
        if "successful_strategies" in target_info:
            strategy_factor = len(target_info["successful_strategies"]) / 50
            mask *= strategy_factor
        
        # Ensure mask values are between 0 and 1
        mask = torch.clamp(mask, 0, 1)
        
        return mask
    
    def get_coalition_learning_status(
        self,
        coalition_id: str
    ) -> Dict[str, Any]:
        """Get learning status for a coalition."""
        try:
            if coalition_id not in self.learning_progress:
                return {}
            
            status = self.learning_progress[coalition_id].copy()
            
            # Add policy information
            if coalition_id in self.coalition_policies:
                policy_cache = self.coalition_policies[coalition_id]
                best_policy = policy_cache.get_best_policy()
                if best_policy:
                    policy_id, policy_data = best_policy
                    status["best_policy"] = {
                        "id": policy_id,
                        "performance": np.mean(policy_cache.performance_history[policy_id]),
                        "timestamp": policy_data["timestamp"],
                        "metadata": policy_data["metadata"]
                    }
            
            # Add knowledge base statistics
            if coalition_id in self.knowledge_base:
                kb = self.knowledge_base[coalition_id]
                status["knowledge_base"] = {
                    "num_experiences": len(kb["task_experiences"]),
                    "num_strategies": len(kb["successful_strategies"]),
                    "num_failures": len(kb["failure_cases"]),
                    "num_adaptations": len(kb["adaptation_history"])
                }
            
            # Add recent performance
            if coalition_id in self.performance_history:
                recent_performance = self.performance_history[coalition_id][-self.config.performance_window:]
                status["recent_performance"] = {
                    "mean": np.mean([p["performance"] for p in recent_performance]),
                    "std": np.std([p["performance"] for p in recent_performance]),
                    "trend": self._calculate_performance_trend(recent_performance)
                }
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting coalition learning status: {str(e)}")
            return {}
    
    def _calculate_performance_trend(
        self,
        performance_history: List[Dict[str, float]]
    ) -> float:
        """Calculate the trend in performance history."""
        if len(performance_history) < 2:
            return 0.0
            
        # Use linear regression to calculate trend
        x = np.arange(len(performance_history))
        y = np.array([p["performance"] for p in performance_history])
        
        A = np.vstack([x, np.ones(len(x))]).T
        m, c = np.linalg.lstsq(A, y, rcond=None)[0]
        
        return m  # Return slope as trend indicator
    
    async def optimize_learning_strategy(
        self,
        coalition_id: str,
        performance_metrics: Dict[str, float]
    ) -> None:
        """Optimize learning strategy based on performance."""
        try:
            current_strategy = self.config.strategy
            
            # Calculate strategy effectiveness
            effectiveness = self._evaluate_strategy_effectiveness(
                coalition_id,
                current_strategy,
                performance_metrics
            )
            
            # Consider strategy change if effectiveness is low
            if effectiveness < 0.5:
                new_strategy = self._select_best_strategy(
                    coalition_id,
                    performance_metrics
                )
                
                if new_strategy != current_strategy:
                    logger.info(
                        f"Changing learning strategy for coalition {coalition_id} "
                        f"from {current_strategy.name} to {new_strategy.name}"
                    )
                    self.config.strategy = new_strategy
                    
                    # Record adaptation
                    self.adaptation_history.setdefault(coalition_id, []).append({
                        "timestamp": datetime.now().timestamp(),
                        "old_strategy": current_strategy.name,
                        "new_strategy": new_strategy.name,
                        "reason": "low_effectiveness",
                        "metrics": performance_metrics
                    })
            
        except Exception as e:
            logger.error(f"Error optimizing learning strategy: {str(e)}")
    
    def _evaluate_strategy_effectiveness(
        self,
        coalition_id: str,
        strategy: LearningStrategy,
        metrics: Dict[str, float]
    ) -> float:
        """Evaluate the effectiveness of a learning strategy."""
        try:
            # Base effectiveness on multiple factors
            effectiveness_scores = []
            
            # Performance improvement
            if coalition_id in self.performance_history:
                recent_performance = self.performance_history[coalition_id][-self.config.performance_window:]
                if recent_performance:
                    trend = self._calculate_performance_trend(recent_performance)
                    effectiveness_scores.append(max(0, min(1, trend + 0.5)))
            
            # Learning efficiency
            learning_progress = self.learning_progress.get(coalition_id, {})
            if learning_progress:
                episodes_trained = learning_progress.get("episodes_trained", 0)
                best_performance = learning_progress.get("best_performance", 0)
                efficiency = best_performance / max(1, episodes_trained)
                effectiveness_scores.append(max(0, min(1, efficiency * 10)))
            
            # Strategy-specific metrics
            if strategy == LearningStrategy.CENTRALIZED:
                # Evaluate coordination effectiveness
                coord_score = metrics.get("coordination_score", 0)
                effectiveness_scores.append(coord_score)
            elif strategy == LearningStrategy.HIERARCHICAL:
                # Evaluate role performance
                role_score = metrics.get("role_performance", 0)
                effectiveness_scores.append(role_score)
            elif strategy == LearningStrategy.FEDERATED:
                # Evaluate knowledge sharing effectiveness
                share_score = metrics.get("knowledge_share_success", 0)
                effectiveness_scores.append(share_score)
            elif strategy == LearningStrategy.ENSEMBLE:
                # Evaluate diversity and combination effectiveness
                diversity_score = metrics.get("policy_diversity", 0)
                effectiveness_scores.append(diversity_score)
            
            # Return average effectiveness if scores exist
            if effectiveness_scores:
                return np.mean(effectiveness_scores)
            return 0.5  # Default to neutral effectiveness
            
        except Exception as e:
            logger.error(f"Error evaluating strategy effectiveness: {str(e)}")
            return 0.0
    
    def _select_best_strategy(
        self,
        coalition_id: str,
        metrics: Dict[str, float]
    ) -> LearningStrategy:
        """Select the best learning strategy based on current conditions."""
        try:
            strategy_scores = {}
            
            # Score each strategy based on current conditions
            for strategy in LearningStrategy:
                score = 0.0
                
                if strategy == LearningStrategy.INDIVIDUAL:
                    # Prefer when coordination is difficult
                    score += (1 - metrics.get("coordination_score", 0)) * 0.5
                    # Prefer when tasks are simple
                    score += (1 - metrics.get("task_complexity", 0)) * 0.5
                
                elif strategy == LearningStrategy.CENTRALIZED:
                    # Prefer when coordination is important
                    score += metrics.get("coordination_score", 0) * 0.4
                    # Prefer when communication is reliable
                    score += metrics.get("communication_reliability", 0) * 0.3
                    # Prefer when tasks require tight coordination
                    score += metrics.get("coordination_requirement", 0) * 0.3
                
                elif strategy == LearningStrategy.HIERARCHICAL:
                    # Prefer when roles are well-defined
                    score += metrics.get("role_clarity", 0) * 0.4
                    # Prefer when task complexity is high
                    score += metrics.get("task_complexity", 0) * 0.3
                    # Prefer when hierarchy is beneficial
                    score += metrics.get("hierarchy_benefit", 0) * 0.3
                
                elif strategy == LearningStrategy.FEDERATED:
                    # Prefer when knowledge sharing is beneficial
                    score += metrics.get("knowledge_share_benefit", 0) * 0.4
                    # Prefer when communication is limited
                    score += (1 - metrics.get("communication_bandwidth", 0)) * 0.3
                    # Prefer when privacy is important
                    score += metrics.get("privacy_requirement", 0) * 0.3
                
                elif strategy == LearningStrategy.ENSEMBLE:
                    # Prefer when diversity is beneficial
                    score += metrics.get("diversity_benefit", 0) * 0.4
                    # Prefer when task has multiple solutions
                    score += metrics.get("solution_space_size", 0) * 0.3
                    # Prefer when computational resources are available
                    score += metrics.get("computational_capacity", 0) * 0.3
                
                strategy_scores[strategy] = score
            
            # Select strategy with highest score
            return max(strategy_scores.items(), key=lambda x: x[1])[0]
            
        except Exception as e:
            logger.error(f"Error selecting best strategy: {str(e)}")
            return LearningStrategy.CENTRALIZED  # Default to centralized strategy 