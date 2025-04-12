"""Perception module for H-MAS AGI system."""

from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import logging
from enum import Enum
import json

from .consciousness import ConsciousnessCore
from .reasoning import ReasoningEngine

class InputModality(Enum):
    """Types of input modalities supported by the perception system."""
    TEXT = "text"
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    IMAGE = "image"
    AUDIO = "audio"
    TEMPORAL = "temporal"
    SPATIAL = "spatial"
    STRUCTURED = "structured"

@dataclass
class PerceptionConfig:
    """Configuration for perception module."""
    feature_dim: int = 512
    num_attention_heads: int = 8
    num_transformer_layers: int = 6
    dropout_rate: float = 0.1
    learning_rate: float = 0.0001
    batch_size: int = 32
    max_sequence_length: int = 1000
    attention_threshold: float = 0.7
    pattern_threshold: float = 0.8
    save_dir: str = "perception_data"

class ModalityEncoder(nn.Module):
    """Neural network for encoding different input modalities."""
    
    def __init__(self, config: PerceptionConfig, modality: InputModality):
        """Initialize modality encoder."""
        super().__init__()
        self.config = config
        self.modality = modality
        
        # Create modality-specific encoding layers
        if modality == InputModality.TEXT:
            self.encoder = self._create_text_encoder()
        elif modality == InputModality.NUMERIC:
            self.encoder = self._create_numeric_encoder()
        elif modality == InputModality.CATEGORICAL:
            self.encoder = self._create_categorical_encoder()
        elif modality == InputModality.IMAGE:
            self.encoder = self._create_image_encoder()
        elif modality == InputModality.AUDIO:
            self.encoder = self._create_audio_encoder()
        elif modality == InputModality.TEMPORAL:
            self.encoder = self._create_temporal_encoder()
        elif modality == InputModality.SPATIAL:
            self.encoder = self._create_spatial_encoder()
        elif modality == InputModality.STRUCTURED:
            self.encoder = self._create_structured_encoder()
            
    def _create_text_encoder(self) -> nn.Module:
        """Create encoder for text input."""
        return nn.Sequential(
            nn.Linear(self.config.feature_dim, self.config.feature_dim),
            nn.ReLU(),
            nn.Linear(self.config.feature_dim, self.config.feature_dim)
        )
        
    def _create_numeric_encoder(self) -> nn.Module:
        """Create encoder for numeric input."""
        return nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, self.config.feature_dim)
        )
        
    def _create_categorical_encoder(self) -> nn.Module:
        """Create encoder for categorical input."""
        return nn.Sequential(
            nn.Embedding(1000, self.config.feature_dim),  # Max 1000 categories
            nn.Linear(self.config.feature_dim, self.config.feature_dim)
        )
        
    def _create_image_encoder(self) -> nn.Module:
        """Create encoder for image input."""
        return nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, self.config.feature_dim)
        )
        
    def _create_audio_encoder(self) -> nn.Module:
        """Create encoder for audio input."""
        return nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=8, stride=4, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=8, stride=4, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=8, stride=4, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(128, self.config.feature_dim)
        )
        
    def _create_temporal_encoder(self) -> nn.Module:
        """Create encoder for temporal sequences."""
        return nn.LSTM(
            input_size=self.config.feature_dim,
            hidden_size=self.config.feature_dim,
            num_layers=2,
            batch_first=True,
            dropout=self.config.dropout_rate
        )
        
    def _create_spatial_encoder(self) -> nn.Module:
        """Create encoder for spatial data."""
        return nn.Sequential(
            nn.Linear(3, 64),  # (x, y, z) coordinates
            nn.ReLU(),
            nn.Linear(64, self.config.feature_dim)
        )
        
    def _create_structured_encoder(self) -> nn.Module:
        """Create encoder for structured data."""
        return nn.Sequential(
            nn.Linear(self.config.feature_dim, self.config.feature_dim),
            nn.ReLU(),
            nn.Linear(self.config.feature_dim, self.config.feature_dim)
        )

class PerceptionCore:
    """Core perception implementation."""
    
    def __init__(
        self,
        config: PerceptionConfig,
        consciousness: ConsciousnessCore,
        reasoning: Optional[ReasoningEngine] = None
    ):
        """Initialize perception core."""
        self.config = config
        self.consciousness = consciousness
        self.reasoning = reasoning
        self.logger = logging.getLogger("perception_core")
        
        # Create save directory
        Path(config.save_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize modality encoders
        self.encoders = {
            modality: ModalityEncoder(config, modality)
            for modality in InputModality
        }
        
        # Initialize integration components
        self.modality_attention = self._create_modality_attention()
        self.feature_integration = self._create_feature_integration()
        self.pattern_detector = self._create_pattern_detector()
        
        # Initialize perception state
        self.current_focus: Optional[Dict[str, Any]] = None
        self.active_patterns: List[Dict[str, Any]] = []
        self.perception_history: List[Dict[str, Any]] = []
        
    def perceive(
        self,
        inputs: Dict[str, Any],
        modalities: Optional[List[InputModality]] = None
    ) -> Dict[str, Any]:
        """Process and integrate multi-modal inputs."""
        # Determine active modalities
        active_modalities = modalities or self._detect_modalities(inputs)
        
        # Encode each modality
        encoded_inputs = {}
        for modality in active_modalities:
            if modality.value in inputs:
                encoded = self._encode_modality(
                    inputs[modality.value],
                    modality
                )
                encoded_inputs[modality] = encoded
                
        if not encoded_inputs:
            return {}
            
        # Apply modality attention
        attended_features = self._apply_modality_attention(encoded_inputs)
        
        # Integrate features
        integrated = self._integrate_features(attended_features)
        
        # Detect patterns
        patterns = self._detect_patterns(integrated)
        
        # Update perception state
        self._update_perception_state(
            inputs,
            encoded_inputs,
            integrated,
            patterns
        )
        
        return {
            "integrated_features": integrated.detach().cpu().numpy(),
            "detected_patterns": patterns,
            "active_modalities": [m.value for m in active_modalities],
            "attention_weights": self._get_attention_weights()
        }
        
    def analyze_scene(
        self,
        scene_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze a complex scene with multiple elements."""
        # Extract scene components
        components = self._extract_scene_components(scene_data)
        
        # Process each component
        component_features = {}
        for comp_id, comp_data in components.items():
            # Determine component modalities
            comp_modalities = self._detect_modalities(comp_data)
            
            # Process component
            processed = self.perceive(comp_data, comp_modalities)
            component_features[comp_id] = processed
            
        # Analyze component relationships
        relationships = self._analyze_component_relationships(
            component_features
        )
        
        # Detect scene-level patterns
        scene_patterns = self._detect_scene_patterns(
            component_features,
            relationships
        )
        
        return {
            "components": component_features,
            "relationships": relationships,
            "patterns": scene_patterns,
            "scene_complexity": self._calculate_scene_complexity(
                component_features
            )
        }
        
    def focus_attention(
        self,
        region: Dict[str, Any],
        modality: Optional[InputModality] = None
    ) -> Dict[str, Any]:
        """Focus attention on specific input region or modality."""
        # Update current focus
        self.current_focus = {
            "region": region,
            "modality": modality,
            "timestamp": datetime.now().isoformat()
        }
        
        # Process focused region
        if modality:
            # Process specific modality
            encoded = self._encode_modality(region, modality)
            attended = self._apply_focused_attention(encoded)
        else:
            # Process all modalities in region
            modalities = self._detect_modalities(region)
            perception = self.perceive(region, modalities)
            attended = perception["integrated_features"]
            
        # Analyze focused features
        analysis = self._analyze_focused_features(attended)
        
        return {
            "focused_features": attended,
            "analysis": analysis,
            "attention_metrics": self._calculate_attention_metrics()
        }
        
    def detect_anomalies(
        self,
        inputs: Dict[str, Any],
        baseline: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Detect anomalies in inputs compared to baseline."""
        # Process inputs
        processed = self.perceive(inputs)
        
        # Get or create baseline
        if baseline is None:
            baseline = self._get_perception_baseline()
            
        # Compare with baseline
        deviations = self._calculate_deviations(
            processed["integrated_features"],
            baseline
        )
        
        # Detect anomalies
        anomalies = []
        for dev in deviations:
            if dev["magnitude"] > self.config.pattern_threshold:
                anomaly = self._analyze_anomaly(dev, inputs)
                anomalies.append(anomaly)
                
        return anomalies
        
    def _create_modality_attention(self) -> nn.Module:
        """Create attention mechanism for modality integration."""
        return nn.MultiheadAttention(
            embed_dim=self.config.feature_dim,
            num_heads=self.config.num_attention_heads,
            dropout=self.config.dropout_rate
        )
        
    def _create_feature_integration(self) -> nn.Module:
        """Create network for feature integration."""
        return nn.Sequential(
            nn.Linear(self.config.feature_dim, self.config.feature_dim),
            nn.ReLU(),
            nn.Linear(self.config.feature_dim, self.config.feature_dim)
        )
        
    def _create_pattern_detector(self) -> nn.Module:
        """Create network for pattern detection."""
        return nn.Sequential(
            nn.Linear(self.config.feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
    def _detect_modalities(
        self,
        inputs: Dict[str, Any]
    ) -> List[InputModality]:
        """Detect input modalities from data."""
        modalities = []
        for modality in InputModality:
            if self._check_modality_presence(inputs, modality):
                modalities.append(modality)
        return modalities
        
    def _encode_modality(
        self,
        data: Any,
        modality: InputModality
    ) -> torch.Tensor:
        """Encode input data for specific modality."""
        # Prepare data for encoding
        prepared = self._prepare_modality_input(data, modality)
        
        # Apply modality-specific encoding
        with torch.no_grad():
            encoded = self.encoders[modality].encoder(prepared)
            
        return encoded
        
    def _apply_modality_attention(
        self,
        encoded_inputs: Dict[InputModality, torch.Tensor]
    ) -> torch.Tensor:
        """Apply attention across modalities."""
        # Stack encoded inputs
        stacked = torch.stack(list(encoded_inputs.values()))
        
        # Apply multi-head attention
        with torch.no_grad():
            attended, _ = self.modality_attention(
                stacked, stacked, stacked
            )
            
        return attended.mean(dim=0)  # Average across modalities
        
    def _integrate_features(
        self,
        features: torch.Tensor
    ) -> torch.Tensor:
        """Integrate features into unified representation."""
        return self.feature_integration(features)
        
    def _detect_patterns(
        self,
        integrated: torch.Tensor
    ) -> List[Dict[str, Any]]:
        """Detect patterns in integrated features."""
        # Apply pattern detection
        with torch.no_grad():
            pattern_features = self.pattern_detector(integrated)
            
        # Find significant patterns
        patterns = []
        for i, feature in enumerate(pattern_features):
            if feature.abs().mean() > self.config.pattern_threshold:
                pattern = {
                    "id": f"pattern_{len(self.active_patterns)}",
                    "features": feature.cpu().numpy(),
                    "strength": float(feature.abs().mean()),
                    "timestamp": datetime.now().isoformat()
                }
                patterns.append(pattern)
                
        return patterns
        
    def _update_perception_state(
        self,
        inputs: Dict[str, Any],
        encoded: Dict[InputModality, torch.Tensor],
        integrated: torch.Tensor,
        patterns: List[Dict[str, Any]]
    ) -> None:
        """Update internal perception state."""
        # Update active patterns
        self.active_patterns.extend(patterns)
        
        # Add to perception history
        self.perception_history.append({
            "inputs": inputs,
            "integrated_features": integrated.cpu().numpy(),
            "patterns": patterns,
            "timestamp": datetime.now().isoformat()
        })
        
        # Maintain history size
        if len(self.perception_history) > self.config.max_sequence_length:
            self.perception_history.pop(0)
            
    def save_state(self, save_path: str) -> None:
        """Save perception state."""
        state = {
            "current_focus": self.current_focus,
            "active_patterns": self.active_patterns,
            "perception_history": self.perception_history
        }
        
        # Save neural network states
        torch.save({
            "encoders": {
                m.value: e.state_dict()
                for m, e in self.encoders.items()
            },
            "modality_attention": self.modality_attention.state_dict(),
            "feature_integration": self.feature_integration.state_dict(),
            "pattern_detector": self.pattern_detector.state_dict()
        }, str(Path(save_path).with_suffix(".pth")))
        
        # Save perception state
        with open(save_path, "w") as f:
            json.dump(state, f, indent=2, default=str)
            
    def load_state(self, load_path: str) -> None:
        """Load perception state."""
        # Load neural network states
        network_state = torch.load(str(Path(load_path).with_suffix(".pth")))
        
        for modality in InputModality:
            self.encoders[modality].load_state_dict(
                network_state["encoders"][modality.value]
            )
            
        self.modality_attention.load_state_dict(
            network_state["modality_attention"]
        )
        self.feature_integration.load_state_dict(
            network_state["feature_integration"]
        )
        self.pattern_detector.load_state_dict(
            network_state["pattern_detector"]
        )
        
        # Load perception state
        with open(load_path, "r") as f:
            state = json.load(f)
            
        self.current_focus = state["current_focus"]
        self.active_patterns = state["active_patterns"]
        self.perception_history = state["perception_history"] 