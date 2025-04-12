from typing import Dict, Any, List, Optional, Union
from uuid import UUID
import numpy as np
from datetime import datetime
from ..core import Agent

class MemoryAgent(Agent):
    """Agent specialized in managing different types of memory systems."""
    
    def __init__(
        self,
        name: str,
        memory_types: List[str] = ["episodic", "semantic", "procedural"],
        memory_size: int = 10000,
        consolidation_interval: int = 3600,  # in seconds
        team_id: Optional[UUID] = None,
        org_id: Optional[UUID] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            name=name,
            capabilities=["memory_management"] + memory_types,
            memory_size=memory_size,
            team_id=team_id,
            org_id=org_id,
            config=config
        )
        self.memory_types = memory_types
        self.consolidation_interval = consolidation_interval
        self.state.update({
            "memories": {mtype: [] for mtype in memory_types},
            "working_memory": [],
            "last_consolidation": datetime.now(),
            "retrieval_stats": {},
            "memory_usage": {}
        })
        
    async def initialize(self) -> bool:
        """Initialize memory systems."""
        try:
            # Initialize memory structures for each type
            for memory_type in self.memory_types:
                self.state["retrieval_stats"][memory_type] = {
                    "total_retrievals": 0,
                    "successful_retrievals": 0
                }
                self.state["memory_usage"][memory_type] = 0
                
            return True
        except Exception as e:
            print(f"Error initializing memory agent {self.name}: {str(e)}")
            return False
            
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process memory operations."""
        try:
            operation = input_data.get("operation")
            memory_type = input_data.get("memory_type")
            content = input_data.get("content")
            
            if operation == "store":
                return await self._store_memory(memory_type, content)
            elif operation == "retrieve":
                return await self._retrieve_memory(memory_type, content)
            elif operation == "consolidate":
                return await self._consolidate_memories()
            else:
                return {"error": "Invalid memory operation"}
                
        except Exception as e:
            print(f"Error processing memory operation in {self.name}: {str(e)}")
            return {"error": str(e)}
            
    async def _store_memory(
        self,
        memory_type: str,
        content: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Store new memory."""
        if memory_type not in self.memory_types:
            return {"error": f"Invalid memory type: {memory_type}"}
            
        memory_entry = {
            "id": str(UUID()),
            "type": memory_type,
            "content": content,
            "timestamp": datetime.now(),
            "access_count": 0,
            "last_access": None,
            "importance": content.get("importance", 0.5)
        }
        
        # Add to working memory first
        self.state["working_memory"].append(memory_entry)
        
        # Check if consolidation is needed
        if self._should_consolidate():
            await self._consolidate_memories()
            
        return {"status": "success", "memory_id": memory_entry["id"]}
        
    async def _retrieve_memory(
        self,
        memory_type: str,
        query: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Retrieve memories based on query."""
        if memory_type not in self.memory_types:
            return {"error": f"Invalid memory type: {memory_type}"}
            
        self.state["retrieval_stats"][memory_type]["total_retrievals"] += 1
        
        # Search in both working memory and consolidated memories
        results = []
        
        # Simple matching implementation - should be enhanced with semantic search
        for memory in self.state["memories"][memory_type]:
            if self._matches_query(memory, query):
                results.append(memory)
                memory["access_count"] += 1
                memory["last_access"] = datetime.now()
                
        for memory in self.state["working_memory"]:
            if memory["type"] == memory_type and self._matches_query(memory, query):
                results.append(memory)
                memory["access_count"] += 1
                memory["last_access"] = datetime.now()
                
        if results:
            self.state["retrieval_stats"][memory_type]["successful_retrievals"] += 1
            
        return {
            "status": "success",
            "results": results,
            "count": len(results)
        }
        
    def _matches_query(self, memory: Dict[str, Any], query: Dict[str, Any]) -> bool:
        """Check if memory matches query criteria."""
        # Implement more sophisticated matching logic
        for key, value in query.items():
            if key in memory["content"]:
                if isinstance(value, (str, int, float, bool)):
                    if memory["content"][key] != value:
                        return False
                elif isinstance(value, dict):
                    if not self._matches_query(memory["content"][key], value):
                        return False
        return True
        
    async def _consolidate_memories(self) -> Dict[str, Any]:
        """Consolidate memories from working memory to long-term storage."""
        consolidated_count = 0
        
        for memory in self.state["working_memory"]:
            memory_type = memory["type"]
            
            # Apply consolidation criteria
            if self._should_retain_memory(memory):
                self.state["memories"][memory_type].append(memory)
                consolidated_count += 1
                
            # Update memory usage statistics
            self.state["memory_usage"][memory_type] = len(
                self.state["memories"][memory_type]
            )
            
        # Clear working memory
        self.state["working_memory"] = []
        self.state["last_consolidation"] = datetime.now()
        
        # Prune old memories if needed
        await self._prune_memories()
        
        return {
            "status": "success",
            "consolidated_count": consolidated_count
        }
        
    def _should_consolidate(self) -> bool:
        """Check if memory consolidation is needed."""
        time_since_consolidation = (
            datetime.now() - self.state["last_consolidation"]
        ).total_seconds()
        
        return (
            time_since_consolidation >= self.consolidation_interval
            or len(self.state["working_memory"]) >= self.memory_size * 0.1
        )
        
    def _should_retain_memory(self, memory: Dict[str, Any]) -> bool:
        """Determine if memory should be retained during consolidation."""
        # Implement more sophisticated retention criteria
        return (
            memory["importance"] > 0.3
            or memory["access_count"] > 0
            or (datetime.now() - memory["timestamp"]).total_seconds() < 86400
        )
        
    async def _prune_memories(self) -> None:
        """Remove least important memories when exceeding capacity."""
        for memory_type in self.memory_types:
            memories = self.state["memories"][memory_type]
            if len(memories) > self.memory_size:
                # Sort by importance and recency
                memories.sort(
                    key=lambda x: (x["importance"], x["access_count"]),
                    reverse=True
                )
                self.state["memories"][memory_type] = memories[:self.memory_size]
                
    async def communicate(self, message: Dict[str, Any], target_id: UUID) -> bool:
        """Share memory updates with other agents."""
        try:
            # Prepare memory update for communication
            payload = {
                "type": "memory_update",
                "source_id": str(self.id),
                "memory_stats": self.state["retrieval_stats"],
                "memory_usage": self.state["memory_usage"],
                "timestamp": str(datetime.now())
            }
            
            # In a real implementation, this would use a communication protocol
            print(f"Sending memory update to agent {target_id}")
            return True
        except Exception as e:
            print(f"Error in communication from {self.name}: {str(e)}")
            return False
            
    async def learn(self, experience: Dict[str, Any]) -> bool:
        """Update memory management strategies based on experience."""
        try:
            if "feedback" in experience:
                # Adjust retention criteria based on feedback
                if "retention_threshold" in experience["feedback"]:
                    self.config["retention_threshold"] = experience["feedback"][
                        "retention_threshold"
                    ]
                    
            return True
        except Exception as e:
            print(f"Error in learning for {self.name}: {str(e)}")
            return False
            
    async def reflect(self) -> Dict[str, Any]:
        """Perform self-assessment of memory systems."""
        return {
            "memory_types": self.memory_types,
            "memory_usage": self.state["memory_usage"],
            "retrieval_stats": self.state["retrieval_stats"],
            "working_memory_size": len(self.state["working_memory"]),
            "last_consolidation": str(self.state["last_consolidation"]),
            "consolidation_interval": self.consolidation_interval
        } 