import aiohttp
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from ..memory.memory_main import Memory, MemoryType, ExperienceType

logger = logging.getLogger(__name__)

class MemoryClient:
    """Client for interacting with the memory system."""
    def __init__(self, base_url: str = "http://localhost:8401"):
        self.base_url = base_url.rstrip('/')
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
            
    async def ensure_session(self):
        """Ensure aiohttp session exists."""
        if not self.session:
            self.session = aiohttp.ClientSession()
            
    async def check_health(self) -> bool:
        """Check if memory system is healthy."""
        try:
            await self.ensure_session()
            async with self.session.get(f"{self.base_url}/health") as response:
                return response.status == 200
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
            
    async def store_memory(self, memory: Memory) -> Dict[str, Any]:
        """Store a memory in the memory system."""
        try:
            await self.ensure_session()
            async with self.session.post(
                f"{self.base_url}/memory",
                json=memory.dict()
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    raise Exception(f"Failed to store memory: {error_text}")
        except Exception as e:
            logger.error(f"Error storing memory: {e}")
            raise
            
    async def query_memories(
        self,
        memory_type: Optional[MemoryType] = None,
        experience_type: Optional[ExperienceType] = None,
        content_filter: Optional[Dict[str, Any]] = None,
        time_range: Optional[List[str]] = None,
        limit: int = 10
    ) -> Dict[str, Any]:
        """Query memories based on criteria."""
        try:
            await self.ensure_session()
            query = {
                "type": memory_type.value if memory_type else None,
                "experience_type": experience_type.value if experience_type else None,
                "content_filter": content_filter,
                "time_range": time_range,
                "limit": limit
            }
            
            async with self.session.post(
                f"{self.base_url}/memory/query",
                json=query
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    raise Exception(f"Failed to query memories: {error_text}")
        except Exception as e:
            logger.error(f"Error querying memories: {e}")
            raise
            
    async def get_replay_batch(self, batch_size: int = 32) -> Dict[str, Any]:
        """Get a batch of memories for experience replay."""
        try:
            await self.ensure_session()
            async with self.session.get(
                f"{self.base_url}/memory/replay",
                params={"batch_size": batch_size}
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    raise Exception(f"Failed to get replay batch: {error_text}")
        except Exception as e:
            logger.error(f"Error getting replay batch: {e}")
            raise
            
    async def find_similar_memories(
        self,
        embedding: List[float],
        k: int = 5
    ) -> Dict[str, Any]:
        """Find similar memories based on embedding."""
        try:
            await self.ensure_session()
            async with self.session.post(
                f"{self.base_url}/memory/similar",
                json={"embedding": embedding},
                params={"k": k}
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    raise Exception(f"Failed to find similar memories: {error_text}")
        except Exception as e:
            logger.error(f"Error finding similar memories: {e}")
            raise 