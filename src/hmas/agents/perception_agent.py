from typing import Dict, Any, List, Optional
from uuid import UUID
import numpy as np
from ..core import Agent

class PerceptionAgent(Agent):
    """Agent specialized in multi-modal perception tasks."""
    
    def __init__(
        self,
        name: str,
        modalities: List[str],
        feature_extractors: Dict[str, Any],
        attention_config: Optional[Dict[str, Any]] = None,
        memory_size: int = 1000,
        team_id: Optional[UUID] = None,
        org_id: Optional[UUID] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            name=name,
            capabilities=["perception"] + modalities,
            memory_size=memory_size,
            team_id=team_id,
            org_id=org_id,
            config=config
        )
        self.modalities = modalities
        self.feature_extractors = feature_extractors
        self.attention_config = attention_config or {
            "attention_type": "multi_head",
            "num_heads": 8,
            "dropout": 0.1
        }
        self.state.update({
            "active_modalities": [],
            "attention_weights": {},
            "feature_cache": {},
            "context_buffer": []
        })
        
    async def initialize(self) -> bool:
        """Initialize perception components."""
        try:
            # Initialize feature extractors for each modality
            for modality in self.modalities:
                if modality not in self.feature_extractors:
                    print(f"Warning: No feature extractor found for {modality}")
                    continue
                    
            self.state["active_modalities"] = self.modalities
            return True
        except Exception as e:
            print(f"Error initializing perception agent {self.name}: {str(e)}")
            return False
            
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process multi-modal input data."""
        try:
            # Extract features for each modality
            features = {}
            for modality in self.state["active_modalities"]:
                if modality in input_data:
                    features[modality] = await self._extract_features(
                        modality,
                        input_data[modality]
                    )
                    
            # Apply attention mechanism
            attended_features = await self._apply_attention(features)
            
            # Update context buffer
            self._update_context(attended_features)
            
            return {
                "features": attended_features,
                "context": self.state["context_buffer"],
                "attention_weights": self.state["attention_weights"]
            }
        except Exception as e:
            print(f"Error processing input in {self.name}: {str(e)}")
            return {}
            
    async def _extract_features(self, modality: str, data: Any) -> np.ndarray:
        """Extract features from input data for a specific modality."""
        extractor = self.feature_extractors[modality]
        features = extractor(data)
        self.state["feature_cache"][modality] = features
        return features
        
    async def _apply_attention(self, features: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Apply attention mechanism to features."""
        if not features:
            return {}
            
        # Implement multi-head attention
        attention_weights = {}
        attended_features = {}
        
        for modality, feature in features.items():
            # Simple attention implementation - should be enhanced
            weights = np.ones(feature.shape[0]) / feature.shape[0]
            attention_weights[modality] = weights
            attended_features[modality] = feature * weights[:, None]
            
        self.state["attention_weights"] = attention_weights
        return attended_features
        
    def _update_context(self, features: Dict[str, np.ndarray]) -> None:
        """Update context buffer with new features."""
        self.state["context_buffer"].append({
            "timestamp": np.datetime64('now'),
            "features": features
        })
        
        # Maintain buffer size
        while len(self.state["context_buffer"]) > self.memory_size:
            self.state["context_buffer"].pop(0)
            
    async def communicate(self, message: Dict[str, Any], target_id: UUID) -> bool:
        """Share perception results with other agents."""
        try:
            # Prepare perception results for communication
            payload = {
                "type": "perception_update",
                "source_id": str(self.id),
                "features": self.state["feature_cache"],
                "attention_weights": self.state["attention_weights"],
                "timestamp": str(np.datetime64('now'))
            }
            
            # In a real implementation, this would use a communication protocol
            print(f"Sending perception update to agent {target_id}")
            return True
        except Exception as e:
            print(f"Error in communication from {self.name}: {str(e)}")
            return False
            
    async def learn(self, experience: Dict[str, Any]) -> bool:
        """Update perception models based on experience."""
        try:
            if "feedback" in experience:
                # Update feature extractors based on feedback
                for modality, feedback in experience["feedback"].items():
                    if modality in self.feature_extractors:
                        # Implement online learning for feature extractors
                        pass
                        
            return True
        except Exception as e:
            print(f"Error in learning for {self.name}: {str(e)}")
            return False
            
    async def reflect(self) -> Dict[str, Any]:
        """Perform self-assessment of perception capabilities."""
        return {
            "active_modalities": self.state["active_modalities"],
            "feature_extraction_status": {
                modality: "active" for modality in self.feature_extractors
            },
            "attention_config": self.attention_config,
            "context_buffer_size": len(self.state["context_buffer"]),
            "last_update": str(np.datetime64('now'))
        } 