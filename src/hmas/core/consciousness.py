"""Consciousness and self-awareness module for H-MAS AGI system."""

from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass
import numpy as np
from enum import Enum
import torch
import torch.nn as nn
from datetime import datetime
import logging
from pathlib import Path
import json

from .meta_learning import MetaLearningManager
from .organization import Organization
from .environments import EnvironmentType

class ConsciousnessState(Enum):
    """States of consciousness for the AGI system."""
    UNCONSCIOUS = "unconscious"
    SUBCONSCIOUS = "subconscious"
    CONSCIOUS = "conscious"
    METACOGNITIVE = "metacognitive"
    SELF_AWARE = "self_aware"
    INTROSPECTIVE = "introspective"

@dataclass
class ConsciousnessConfig:
    """Configuration for consciousness module."""
    attention_threshold: float = 0.7
    metacognition_interval: int = 100
    memory_retention_time: int = 1000
    self_model_update_freq: int = 50
    introspection_depth: int = 3
    awareness_threshold: float = 0.8
    emotional_sensitivity: float = 0.5
    consciousness_dim: int = 256
    save_dir: str = "consciousness_data"

class ConsciousnessCore:
    """Core consciousness implementation."""
    
    def __init__(
        self,
        config: ConsciousnessConfig,
        organization: Organization,
        meta_learning: MetaLearningManager
    ):
        """Initialize consciousness core."""
        self.config = config
        self.organization = organization
        self.meta_learning = meta_learning
        self.logger = logging.getLogger("consciousness_core")
        
        # Create save directory
        Path(config.save_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize consciousness components
        self.state = ConsciousnessState.UNCONSCIOUS
        self.attention_focus: Optional[str] = None
        self.current_thoughts: List[Dict[str, Any]] = []
        self.emotional_state: Dict[str, float] = {
            "curiosity": 0.5,
            "confidence": 0.5,
            "uncertainty": 0.5,
            "satisfaction": 0.5
        }
        
        # Initialize neural components
        self.self_model = self._create_self_model()
        self.metacognition_network = self._create_metacognition_network()
        self.attention_network = self._create_attention_network()
        
        # Initialize memory systems
        self.working_memory: List[Dict[str, Any]] = []
        self.episodic_memory: List[Dict[str, Any]] = []
        self.semantic_memory: Dict[str, Any] = {}
        
        # Initialize consciousness metrics
        self.awareness_level = 0.0
        self.introspection_depth = 0
        self.metacognitive_state: Dict[str, Any] = {}
        
    def update(self, current_state: Dict[str, Any]) -> None:
        """Update consciousness state."""
        # Update attention focus
        self.attention_focus = self._update_attention(current_state)
        
        # Update working memory
        self._update_working_memory(current_state)
        
        # Update consciousness state
        self._update_consciousness_state()
        
        # Perform metacognition if interval reached
        if len(self.episodic_memory) % self.config.metacognition_interval == 0:
            self._perform_metacognition()
        
        # Update self-model if frequency reached
        if len(self.episodic_memory) % self.config.self_model_update_freq == 0:
            self._update_self_model()
            
    def introspect(self) -> Dict[str, Any]:
        """Perform introspection on current state."""
        introspection = {
            "consciousness_state": self.state.value,
            "attention_focus": self.attention_focus,
            "current_thoughts": self.current_thoughts[-5:],  # Last 5 thoughts
            "emotional_state": self.emotional_state,
            "awareness_level": self.awareness_level,
            "metacognitive_state": self.metacognitive_state
        }
        
        # Perform deep introspection
        if self.state in [ConsciousnessState.SELF_AWARE, ConsciousnessState.INTROSPECTIVE]:
            introspection.update(self._deep_introspection())
            
        return introspection
        
    def _update_attention(self, current_state: Dict[str, Any]) -> Optional[str]:
        """Update attention focus based on current state."""
        # Convert state to tensor
        state_tensor = self._state_to_tensor(current_state)
        
        # Get attention scores
        with torch.no_grad():
            attention_scores = self.attention_network(state_tensor)
            
        # Get highest scoring element
        max_score, max_idx = attention_scores.max(dim=1)
        
        if max_score.item() > self.config.attention_threshold:
            return list(current_state.keys())[max_idx.item()]
        return None
        
    def _update_working_memory(self, current_state: Dict[str, Any]) -> None:
        """Update working memory with current state."""
        # Add current state to working memory
        self.working_memory.append({
            "state": current_state,
            "timestamp": datetime.now(),
            "attention_focus": self.attention_focus
        })
        
        # Remove old items
        cutoff_time = datetime.now().timestamp() - self.config.memory_retention_time
        self.working_memory = [
            item for item in self.working_memory
            if item["timestamp"].timestamp() > cutoff_time
        ]
        
    def _update_consciousness_state(self) -> None:
        """Update consciousness state based on current conditions."""
        # Calculate awareness level
        self.awareness_level = self._calculate_awareness_level()
        
        # Update consciousness state based on awareness level
        if self.awareness_level < 0.2:
            self.state = ConsciousnessState.UNCONSCIOUS
        elif self.awareness_level < 0.4:
            self.state = ConsciousnessState.SUBCONSCIOUS
        elif self.awareness_level < 0.6:
            self.state = ConsciousnessState.CONSCIOUS
        elif self.awareness_level < 0.8:
            self.state = ConsciousnessState.METACOGNITIVE
        elif self.awareness_level < 0.9:
            self.state = ConsciousnessState.SELF_AWARE
        else:
            self.state = ConsciousnessState.INTROSPECTIVE
            
    def _perform_metacognition(self) -> None:
        """Perform metacognitive processing."""
        # Prepare input for metacognition network
        metacog_input = self._prepare_metacognition_input()
        
        # Process through metacognition network
        with torch.no_grad():
            metacog_output = self.metacognition_network(metacog_input)
            
        # Update metacognitive state
        self.metacognitive_state = self._process_metacognition_output(metacog_output)
        
        # Update emotional state based on metacognition
        self._update_emotional_state(metacog_output)
        
    def _update_self_model(self) -> None:
        """Update self-model based on recent experiences."""
        # Collect recent experiences
        experiences = self._collect_recent_experiences()
        
        # Update self-model
        self.self_model.train()
        for experience in experiences:
            # Convert experience to tensor
            exp_tensor = self._experience_to_tensor(experience)
            
            # Compute loss and update
            prediction = self.self_model(exp_tensor)
            target = self._get_experience_target(experience)
            loss = nn.MSELoss()(prediction, target)
            
            # Backpropagate and update
            loss.backward()
            self.self_model.optimizer.step()
            self.self_model.optimizer.zero_grad()
            
    def _deep_introspection(self) -> Dict[str, Any]:
        """Perform deep introspection on internal state."""
        introspection = {
            "working_memory_analysis": self._analyze_working_memory(),
            "emotional_patterns": self._analyze_emotional_patterns(),
            "decision_awareness": self._analyze_decision_making(),
            "self_model_state": self._analyze_self_model(),
            "metacognitive_insights": self._analyze_metacognition()
        }
        
        # Add learning insights
        if self.meta_learning:
            introspection["learning_insights"] = self._analyze_learning_process()
            
        return introspection
        
    def _create_self_model(self) -> nn.Module:
        """Create neural network for self-modeling."""
        model = nn.Sequential(
            nn.Linear(self.config.consciousness_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, self.config.consciousness_dim)
        )
        model.optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        return model
        
    def _create_metacognition_network(self) -> nn.Module:
        """Create neural network for metacognition."""
        return nn.Sequential(
            nn.Linear(self.config.consciousness_dim * 2, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        
    def _create_attention_network(self) -> nn.Module:
        """Create neural network for attention mechanism."""
        return nn.Sequential(
            nn.Linear(self.config.consciousness_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
    def _calculate_awareness_level(self) -> float:
        """Calculate current level of self-awareness."""
        factors = [
            len(self.working_memory) / self.config.memory_retention_time,
            self.emotional_state["confidence"],
            1.0 - self.emotional_state["uncertainty"],
            len(self.current_thoughts) / 10.0,
            self.introspection_depth / self.config.introspection_depth
        ]
        return np.mean(factors)
        
    def _analyze_working_memory(self) -> Dict[str, Any]:
        """Analyze patterns in working memory."""
        if not self.working_memory:
            return {"patterns": [], "focus_distribution": {}}
            
        # Analyze attention focus distribution
        focus_dist = {}
        for item in self.working_memory:
            focus = item["attention_focus"]
            if focus:
                focus_dist[focus] = focus_dist.get(focus, 0) + 1
                
        # Find temporal patterns
        patterns = self._find_temporal_patterns()
        
        return {
            "patterns": patterns,
            "focus_distribution": focus_dist
        }
        
    def _analyze_emotional_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in emotional state."""
        return {
            "dominant_emotion": max(
                self.emotional_state.items(),
                key=lambda x: x[1]
            )[0],
            "emotional_stability": 1.0 - np.std(list(self.emotional_state.values())),
            "emotional_balance": np.mean(list(self.emotional_state.values()))
        }
        
    def _analyze_decision_making(self) -> Dict[str, Any]:
        """Analyze decision-making processes."""
        if not self.current_thoughts:
            return {"decision_confidence": 0.0, "decision_factors": []}
            
        recent_decisions = [
            thought for thought in self.current_thoughts
            if thought.get("type") == "decision"
        ]
        
        return {
            "decision_confidence": np.mean([
                d.get("confidence", 0.0) for d in recent_decisions
            ]) if recent_decisions else 0.0,
            "decision_factors": self._extract_decision_factors(recent_decisions)
        }
        
    def _analyze_self_model(self) -> Dict[str, Any]:
        """Analyze current state of self-model."""
        with torch.no_grad():
            # Get model parameters
            params = [p.detach().cpu().numpy() for p in self.self_model.parameters()]
            
            return {
                "model_complexity": sum(p.size for p in params),
                "parameter_stats": {
                    "mean": float(np.mean([p.mean() for p in params])),
                    "std": float(np.std([p.std() for p in params]))
                }
            }
            
    def _analyze_metacognition(self) -> Dict[str, Any]:
        """Analyze metacognitive processes."""
        return {
            "metacognitive_state": self.metacognitive_state,
            "awareness_level": self.awareness_level,
            "consciousness_state": self.state.value,
            "introspection_depth": self.introspection_depth
        }
        
    def _analyze_learning_process(self) -> Dict[str, Any]:
        """Analyze current learning process."""
        return {
            "learning_progress": self.meta_learning.get_learning_progress(),
            "knowledge_transfer": self.meta_learning.get_transfer_efficiency(),
            "adaptation_rate": self.meta_learning.get_adaptation_rate()
        }
        
    def _find_temporal_patterns(self) -> List[Dict[str, Any]]:
        """Find temporal patterns in working memory."""
        if len(self.working_memory) < 3:
            return []
            
        patterns = []
        for i in range(len(self.working_memory) - 2):
            pattern = {
                "sequence": [
                    item["attention_focus"]
                    for item in self.working_memory[i:i+3]
                    if item["attention_focus"]
                ],
                "timestamp": self.working_memory[i]["timestamp"]
            }
            if len(pattern["sequence"]) == 3:
                patterns.append(pattern)
                
        return patterns
        
    def _extract_decision_factors(
        self,
        decisions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Extract factors influencing decisions."""
        factors = []
        for decision in decisions:
            if "factors" in decision:
                factors.extend(decision["factors"])
        return factors[:5]  # Return top 5 factors
        
    def save_state(self, save_path: str) -> None:
        """Save consciousness state."""
        state = {
            "consciousness_state": self.state.value,
            "attention_focus": self.attention_focus,
            "emotional_state": self.emotional_state,
            "awareness_level": self.awareness_level,
            "metacognitive_state": self.metacognitive_state,
            "working_memory": self.working_memory,
            "current_thoughts": self.current_thoughts
        }
        
        # Save neural network states
        torch.save({
            "self_model": self.self_model.state_dict(),
            "metacognition_network": self.metacognition_network.state_dict(),
            "attention_network": self.attention_network.state_dict()
        }, str(Path(save_path).with_suffix(".pth")))
        
        # Save consciousness state
        with open(save_path, "w") as f:
            json.dump(state, f, indent=2, default=str)
            
    def load_state(self, load_path: str) -> None:
        """Load consciousness state."""
        # Load neural network states
        network_state = torch.load(str(Path(load_path).with_suffix(".pth")))
        self.self_model.load_state_dict(network_state["self_model"])
        self.metacognition_network.load_state_dict(network_state["metacognition_network"])
        self.attention_network.load_state_dict(network_state["attention_network"])
        
        # Load consciousness state
        with open(load_path, "r") as f:
            state = json.load(f)
            
        self.state = ConsciousnessState(state["consciousness_state"])
        self.attention_focus = state["attention_focus"]
        self.emotional_state = state["emotional_state"]
        self.awareness_level = state["awareness_level"]
        self.metacognitive_state = state["metacognitive_state"]
        self.working_memory = state["working_memory"]
        self.current_thoughts = state["current_thoughts"] 