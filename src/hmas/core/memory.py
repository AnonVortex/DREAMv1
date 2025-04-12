"""Memory module for H-MAS AGI system."""

from typing import Dict, List, Optional, Any, Tuple, Union, Set
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import logging
from enum import Enum
import json
from datetime import datetime
from uuid import UUID, uuid4

from .consciousness import ConsciousnessCore
from .perception import PerceptionCore

class MemoryType(Enum):
    """Types of memory supported by the system."""
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"
    WORKING = "working"
    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"

@dataclass
class MemoryConfig:
    """Configuration for memory system."""
    embedding_dim: int = 512
    max_episodes: int = 10000
    max_concepts: int = 100000
    max_procedures: int = 1000
    working_memory_size: int = 10
    consolidation_threshold: float = 0.7
    retrieval_threshold: float = 0.6
    forgetting_rate: float = 0.1
    learning_rate: float = 0.0001
    save_dir: str = "memory_data"

@dataclass
class MemoryTrace:
    """Base class for memory traces."""
    id: UUID
    created_at: datetime
    last_accessed: datetime
    memory_type: str
    encoding: torch.Tensor
    importance: float
    context: Dict[str, Any]
    metadata: Dict[str, Any]

@dataclass
class EpisodicMemory(MemoryTrace):
    """Representation of an episodic memory."""
    episode_data: Dict[str, Any]
    temporal_context: Dict[str, Any]
    emotional_valence: float
    participants: List[str]
    location: Optional[str] = None

@dataclass
class SemanticMemory(MemoryTrace):
    """Representation of semantic knowledge."""
    concept: str
    properties: Dict[str, Any]
    relationships: Dict[str, List[str]]
    confidence: float

@dataclass
class ProceduralMemory(MemoryTrace):
    """Representation of procedural knowledge."""
    procedure_name: str
    steps: List[Dict[str, Any]]
    prerequisites: List[str]
    success_rate: float
    execution_time: float

class MemoryEncoder(nn.Module):
    """Neural network for encoding memories."""
    
    def __init__(self, config: MemoryConfig):
        """Initialize memory encoder."""
        super().__init__()
        self.config = config
        
        # Create encoding networks
        self.episodic_encoder = self._create_episodic_encoder()
        self.semantic_encoder = self._create_semantic_encoder()
        self.procedural_encoder = self._create_procedural_encoder()
        
        # Create memory consolidation network
        self.consolidation_network = self._create_consolidation_network()
        
    def _create_episodic_encoder(self) -> nn.Module:
        """Create encoder for episodic memories."""
        return nn.Sequential(
            nn.Linear(self.config.embedding_dim * 2, self.config.embedding_dim),
            nn.ReLU(),
            nn.Linear(self.config.embedding_dim, self.config.embedding_dim)
        )
        
    def _create_semantic_encoder(self) -> nn.Module:
        """Create encoder for semantic memories."""
        return nn.Sequential(
            nn.Linear(self.config.embedding_dim, self.config.embedding_dim),
            nn.ReLU(),
            nn.Linear(self.config.embedding_dim, self.config.embedding_dim)
        )
        
    def _create_procedural_encoder(self) -> nn.Module:
        """Create encoder for procedural memories."""
        return nn.Sequential(
            nn.LSTM(
                input_size=self.config.embedding_dim,
                hidden_size=self.config.embedding_dim,
                num_layers=2,
                batch_first=True
            )
        )
        
    def _create_consolidation_network(self) -> nn.Module:
        """Create network for memory consolidation."""
        return nn.Sequential(
            nn.Linear(self.config.embedding_dim * 2, self.config.embedding_dim),
            nn.ReLU(),
            nn.Linear(self.config.embedding_dim, self.config.embedding_dim),
            nn.Sigmoid()
        )

class MemorySystem:
    """Core memory system implementation."""
    
    def __init__(
        self,
        config: MemoryConfig,
        consciousness: ConsciousnessCore,
        perception: Optional[PerceptionCore] = None
    ):
        """Initialize memory system."""
        self.config = config
        self.consciousness = consciousness
        self.perception = perception
        self.logger = logging.getLogger("memory_system")
        
        # Create save directory
        Path(config.save_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize memory encoder
        self.encoder = MemoryEncoder(config)
        
        # Initialize memory stores
        self.episodic_store: Dict[UUID, EpisodicMemory] = {}
        self.semantic_store: Dict[str, SemanticMemory] = {}
        self.procedural_store: Dict[str, ProceduralMemory] = {}
        self.working_memory: List[UUID] = []
        
        # Initialize indices
        self.temporal_index: Dict[datetime, List[UUID]] = {}
        self.spatial_index: Dict[str, List[UUID]] = {}
        self.concept_index: Dict[str, List[UUID]] = {}
        self.procedure_index: Dict[str, UUID] = {}
        
    async def store_episodic(
        self,
        episode_data: Dict[str, Any],
        context: Dict[str, Any]
    ) -> UUID:
        """Store an episodic memory."""
        # Generate memory encoding
        encoding = await self._encode_episode(episode_data, context)
        
        # Create memory trace
        memory_id = uuid4()
        timestamp = datetime.now()
        
        memory = EpisodicMemory(
            id=memory_id,
            created_at=timestamp,
            last_accessed=timestamp,
            memory_type=MemoryType.EPISODIC.value,
            encoding=encoding,
            importance=self._calculate_importance(episode_data),
            context=context,
            metadata={},
            episode_data=episode_data,
            temporal_context=self._extract_temporal_context(context),
            emotional_valence=self._calculate_emotional_valence(episode_data),
            participants=self._extract_participants(episode_data),
            location=context.get("location")
        )
        
        # Store memory
        self.episodic_store[memory_id] = memory
        
        # Update indices
        self._update_temporal_index(timestamp, memory_id)
        if memory.location:
            self._update_spatial_index(memory.location, memory_id)
            
        # Trigger memory consolidation
        await self._consolidate_memory(memory)
        
        return memory_id
        
    async def store_semantic(
        self,
        concept: str,
        properties: Dict[str, Any],
        relationships: Dict[str, List[str]],
        confidence: float
    ) -> None:
        """Store semantic knowledge."""
        # Generate concept encoding
        encoding = await self._encode_concept(concept, properties)
        
        # Create memory trace
        timestamp = datetime.now()
        memory = SemanticMemory(
            id=uuid4(),
            created_at=timestamp,
            last_accessed=timestamp,
            memory_type=MemoryType.SEMANTIC.value,
            encoding=encoding,
            importance=self._calculate_concept_importance(concept),
            context={},
            metadata={},
            concept=concept,
            properties=properties,
            relationships=relationships,
            confidence=confidence
        )
        
        # Store memory
        self.semantic_store[concept] = memory
        
        # Update concept index
        self._update_concept_index(concept, memory.id)
        
        # Link related concepts
        await self._link_related_concepts(concept, relationships)
        
    async def store_procedural(
        self,
        procedure_name: str,
        steps: List[Dict[str, Any]],
        prerequisites: List[str]
    ) -> None:
        """Store procedural knowledge."""
        # Generate procedure encoding
        encoding = await self._encode_procedure(steps)
        
        # Create memory trace
        timestamp = datetime.now()
        memory = ProceduralMemory(
            id=uuid4(),
            created_at=timestamp,
            last_accessed=timestamp,
            memory_type=MemoryType.PROCEDURAL.value,
            encoding=encoding,
            importance=self._calculate_procedure_importance(steps),
            context={},
            metadata={},
            procedure_name=procedure_name,
            steps=steps,
            prerequisites=prerequisites,
            success_rate=1.0,  # Initial success rate
            execution_time=0.0  # Initial execution time
        )
        
        # Store memory
        self.procedural_store[procedure_name] = memory
        
        # Update procedure index
        self.procedure_index[procedure_name] = memory.id
        
    async def retrieve_episodic(
        self,
        query: Dict[str, Any],
        limit: Optional[int] = None
    ) -> List[EpisodicMemory]:
        """Retrieve episodic memories matching query."""
        # Generate query encoding
        query_encoding = await self._encode_query(query)
        
        # Find matching memories
        matches = []
        for memory in self.episodic_store.values():
            similarity = self._calculate_similarity(
                query_encoding,
                memory.encoding
            )
            if similarity > self.config.retrieval_threshold:
                matches.append((similarity, memory))
                
        # Sort by similarity and apply limit
        matches.sort(reverse=True, key=lambda x: x[0])
        if limit:
            matches = matches[:limit]
            
        # Update access timestamps
        for _, memory in matches:
            memory.last_accessed = datetime.now()
            
        return [m for _, m in matches]
        
    async def retrieve_semantic(
        self,
        concept: str,
        include_related: bool = False
    ) -> Optional[Dict[str, Any]]:
        """Retrieve semantic knowledge about concept."""
        if concept not in self.semantic_store:
            return None
            
        # Get concept memory
        memory = self.semantic_store[concept]
        memory.last_accessed = datetime.now()
        
        result = {
            "concept": memory.concept,
            "properties": memory.properties,
            "relationships": memory.relationships,
            "confidence": memory.confidence
        }
        
        if include_related:
            # Get related concepts
            related = {}
            for rel_type, rel_concepts in memory.relationships.items():
                related[rel_type] = []
                for rel_concept in rel_concepts:
                    if rel_concept in self.semantic_store:
                        rel_memory = self.semantic_store[rel_concept]
                        related[rel_type].append({
                            "concept": rel_concept,
                            "properties": rel_memory.properties,
                            "confidence": rel_memory.confidence
                        })
            result["related_concepts"] = related
            
        return result
        
    async def retrieve_procedural(
        self,
        procedure_name: str
    ) -> Optional[ProceduralMemory]:
        """Retrieve procedural knowledge."""
        if procedure_name not in self.procedural_store:
            return None
            
        memory = self.procedural_store[procedure_name]
        memory.last_accessed = datetime.now()
        return memory
        
    async def update_working_memory(
        self,
        items: List[Union[UUID, str]]
    ) -> None:
        """Update working memory contents."""
        # Clear current working memory
        self.working_memory.clear()
        
        # Add new items
        for item in items[:self.config.working_memory_size]:
            if isinstance(item, UUID) and item in self.episodic_store:
                self.working_memory.append(item)
            elif isinstance(item, str):
                if item in self.semantic_store:
                    self.working_memory.append(self.semantic_store[item].id)
                elif item in self.procedural_store:
                    self.working_memory.append(self.procedural_store[item].id)
                    
    async def consolidate_memories(self) -> None:
        """Consolidate short-term memories into long-term storage."""
        # Get memories needing consolidation
        recent_memories = self._get_recent_memories()
        
        for memory in recent_memories:
            if self._should_consolidate(memory):
                await self._consolidate_memory(memory)
                
    async def forget_memories(self) -> None:
        """Apply forgetting to memories based on importance and access patterns."""
        current_time = datetime.now()
        
        # Check episodic memories
        for memory_id, memory in list(self.episodic_store.items()):
            if self._should_forget(memory, current_time):
                self._remove_memory(memory_id)
                
        # Check semantic memories
        for concept, memory in list(self.semantic_store.items()):
            if self._should_forget(memory, current_time):
                self._remove_concept(concept)
                
        # Check procedural memories
        for proc_name, memory in list(self.procedural_store.items()):
            if self._should_forget(memory, current_time):
                self._remove_procedure(proc_name)
                
    def _calculate_similarity(
        self,
        encoding1: torch.Tensor,
        encoding2: torch.Tensor
    ) -> float:
        """Calculate similarity between memory encodings."""
        return float(torch.cosine_similarity(encoding1, encoding2, dim=0))
        
    def _should_consolidate(self, memory: MemoryTrace) -> bool:
        """Determine if memory should be consolidated."""
        # Check importance and age
        importance_check = memory.importance > self.config.consolidation_threshold
        age_check = (datetime.now() - memory.created_at).total_seconds() > 3600
        
        return importance_check and age_check
        
    def _should_forget(self, memory: MemoryTrace, current_time: datetime) -> bool:
        """Determine if memory should be forgotten."""
        # Calculate memory age and access frequency
        age = (current_time - memory.created_at).total_seconds()
        last_access = (current_time - memory.last_accessed).total_seconds()
        
        # Apply forgetting curve
        forgetting_factor = np.exp(-self.config.forgetting_rate * age)
        importance_factor = memory.importance
        
        return (forgetting_factor * importance_factor) < self.config.forgetting_rate
        
    def _remove_memory(self, memory_id: UUID) -> None:
        """Remove memory and update indices."""
        if memory_id in self.episodic_store:
            memory = self.episodic_store[memory_id]
            
            # Remove from indices
            self._remove_from_temporal_index(memory.created_at, memory_id)
            if memory.location:
                self._remove_from_spatial_index(memory.location, memory_id)
                
            # Remove from store
            del self.episodic_store[memory_id]
            
    def _remove_concept(self, concept: str) -> None:
        """Remove concept and update indices."""
        if concept in self.semantic_store:
            memory = self.semantic_store[concept]
            
            # Remove from concept index
            self._remove_from_concept_index(concept, memory.id)
            
            # Remove from store
            del self.semantic_store[concept]
            
    def _remove_procedure(self, procedure_name: str) -> None:
        """Remove procedure and update indices."""
        if procedure_name in self.procedural_store:
            # Remove from procedure index
            del self.procedure_index[procedure_name]
            
            # Remove from store
            del self.procedural_store[procedure_name]
            
    async def _consolidate_memory(self, memory: MemoryTrace) -> None:
        """Consolidate memory into long-term storage."""
        if memory.memory_type == MemoryType.EPISODIC.value:
            # Extract semantic knowledge
            concepts = await self._extract_concepts(memory)
            for concept, properties in concepts.items():
                if concept not in self.semantic_store:
                    await self.store_semantic(
                        concept,
                        properties,
                        {},  # Initial empty relationships
                        0.7  # Initial confidence
                    )
                    
            # Extract procedures
            procedures = await self._extract_procedures(memory)
            for proc_name, proc_data in procedures.items():
                if proc_name not in self.procedural_store:
                    await self.store_procedural(
                        proc_name,
                        proc_data["steps"],
                        proc_data["prerequisites"]
                    )
                    
    def save_state(self, save_path: str) -> None:
        """Save memory system state."""
        state = {
            "episodic_store": {
                str(k): self._serialize_memory(v)
                for k, v in self.episodic_store.items()
            },
            "semantic_store": {
                k: self._serialize_memory(v)
                for k, v in self.semantic_store.items()
            },
            "procedural_store": {
                k: self._serialize_memory(v)
                for k, v in self.procedural_store.items()
            },
            "working_memory": [str(x) for x in self.working_memory],
            "indices": {
                "temporal": {
                    k.isoformat(): [str(x) for x in v]
                    for k, v in self.temporal_index.items()
                },
                "spatial": self.spatial_index,
                "concept": self.concept_index,
                "procedure": {
                    k: str(v) for k, v in self.procedure_index.items()
                }
            }
        }
        
        # Save neural network states
        torch.save(
            self.encoder.state_dict(),
            str(Path(save_path).with_suffix(".pth"))
        )
        
        # Save memory state
        with open(save_path, "w") as f:
            json.dump(state, f, indent=2, default=str)
            
    def load_state(self, load_path: str) -> None:
        """Load memory system state."""
        # Load neural network states
        self.encoder.load_state_dict(
            torch.load(str(Path(load_path).with_suffix(".pth")))
        )
        
        # Load memory state
        with open(load_path, "r") as f:
            state = json.load(f)
            
        self.episodic_store = {
            UUID(k): self._deserialize_memory(v, MemoryType.EPISODIC)
            for k, v in state["episodic_store"].items()
        }
        
        self.semantic_store = {
            k: self._deserialize_memory(v, MemoryType.SEMANTIC)
            for k, v in state["semantic_store"].items()
        }
        
        self.procedural_store = {
            k: self._deserialize_memory(v, MemoryType.PROCEDURAL)
            for k, v in state["procedural_store"].items()
        }
        
        self.working_memory = [UUID(x) for x in state["working_memory"]]
        
        # Restore indices
        self.temporal_index = {
            datetime.fromisoformat(k): [UUID(x) for x in v]
            for k, v in state["indices"]["temporal"].items()
        }
        self.spatial_index = state["indices"]["spatial"]
        self.concept_index = state["indices"]["concept"]
        self.procedure_index = {
            k: UUID(v) for k, v in state["indices"]["procedure"].items()
        } 