import os
import logging.config
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Tuple, Optional
from datetime import datetime
from enum import Enum

from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.util import get_remote_address
from prometheus_fastapi_instrumentator import Instrumentator
from starlette.middleware.trustedhost import TrustedHostMiddleware
import uvicorn
import redis.asyncio as redis
import numpy as np
from sklearn.neighbors import NearestNeighbors

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Import configuration from local config.py
from .config import settings

LOGGING_CONFIG_PATH = "logging.conf"
if os.path.exists(LOGGING_CONFIG_PATH):
    logging.config.fileConfig(LOGGING_CONFIG_PATH, disable_existing_loggers=False)
logger = logging.getLogger(__name__)

class MemoryType(str, Enum):
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"
    WORKING = "working"

class ExperienceType(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    EXPLORATION = "exploration"
    INTERACTION = "interaction"

class Memory(BaseModel):
    id: str = Field(default_factory=lambda: datetime.now().isoformat())
    type: MemoryType
    experience_type: ExperienceType
    content: Dict[str, Any]
    metadata: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.now)
    importance_score: float = Field(default=0.0, ge=0.0, le=1.0)
    embedding: Optional[List[float]] = None

class MemoryQuery(BaseModel):
    type: Optional[MemoryType] = None
    experience_type: Optional[ExperienceType] = None
    content_filter: Optional[Dict[str, Any]] = None
    time_range: Optional[Tuple[datetime, datetime]] = None
    similarity_threshold: Optional[float] = Field(default=0.8, ge=0.0, le=1.0)
    limit: Optional[int] = Field(default=10, gt=0)

class ExperienceReplayBuffer:
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.buffer: List[Memory] = []
        self.priorities: np.ndarray = np.array([])
        self._knn = NearestNeighbors(n_neighbors=5, algorithm='ball_tree')
        
    def add(self, memory: Memory):
        if len(self.buffer) >= self.max_size:
            # Remove lowest priority memory
            min_priority_idx = np.argmin(self.priorities)
            self.buffer.pop(min_priority_idx)
            self.priorities = np.delete(self.priorities, min_priority_idx)
        
        self.buffer.append(memory)
        self.priorities = np.append(self.priorities, memory.importance_score)
        
        # Update KNN if we have embeddings
        if all(m.embedding for m in self.buffer):
            embeddings = np.array([m.embedding for m in self.buffer])
            self._knn.fit(embeddings)
    
    def sample(self, batch_size: int = 32) -> List[Memory]:
        if not self.buffer:
            return []
        
        # Prioritized sampling
        probs = self.priorities / np.sum(self.priorities)
        indices = np.random.choice(
            len(self.buffer),
            min(batch_size, len(self.buffer)),
            p=probs,
            replace=False
        )
        return [self.buffer[i] for i in indices]
    
    def get_similar(self, embedding: List[float], k: int = 5) -> List[Tuple[Memory, float]]:
        if not self.buffer or not all(m.embedding for m in self.buffer):
            return []
        
        distances, indices = self._knn.kneighbors([embedding], n_neighbors=min(k, len(self.buffer)))
        return [(self.buffer[i], 1.0 - d) for i, d in zip(indices[0], distances[0])]

class MemorySystem:
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.replay_buffer = ExperienceReplayBuffer()
        self.consolidation_threshold = 100  # Number of memories before consolidation
        
    async def store(self, memory: Memory):
        # Store in Redis
        key = f"memory:{memory.type}:{memory.id}"
        await self.redis.hset(key, mapping={
            "content": str(memory.content),
            "metadata": str(memory.metadata),
            "importance": str(memory.importance_score),
            "timestamp": memory.timestamp.isoformat()
        })
        
        # Add to replay buffer if it's an experience
        if memory.type in [MemoryType.EPISODIC, MemoryType.PROCEDURAL]:
            self.replay_buffer.add(memory)
        
        # Check if consolidation is needed
        stored_count = await self.redis.dbsize()
        if stored_count >= self.consolidation_threshold:
            await self.consolidate_memories()
    
    async def retrieve(self, query: MemoryQuery) -> List[Memory]:
        memories = []
        pattern = f"memory:{query.type or '*'}:*"
        
        async for key in self.redis.scan_iter(pattern):
            memory_data = await self.redis.hgetall(key)
            if not memory_data:
                continue
            
            memory = self._construct_memory(key, memory_data)
            
            if self._matches_query(memory, query):
                memories.append(memory)
        
        return sorted(
            memories,
            key=lambda m: m.importance_score,
            reverse=True
        )[:query.limit]
    
    async def consolidate_memories(self):
        """Consolidate memories by updating importance scores and removing low-importance ones."""
        pattern = "memory:*"
        
        async for key in self.redis.scan_iter(pattern):
            memory_data = await self.redis.hgetall(key)
            if not memory_data:
                continue
            
            memory = self._construct_memory(key, memory_data)
            
            # Update importance score based on access patterns and age
            access_count = await self.redis.hget(f"{key}:stats", "access_count") or 0
            age_days = (datetime.now() - memory.timestamp).days
            
            new_score = min(1.0, float(access_count) / 100) * np.exp(-age_days / 365)
            
            if new_score < 0.1:  # Remove low-importance memories
                await self.redis.delete(key)
            else:
                await self.redis.hset(key, "importance", str(new_score))
    
    def _construct_memory(self, key: str, data: Dict[str, str]) -> Memory:
        """Construct a Memory object from Redis data."""
        _, type_str, id_str = key.split(":")
        return Memory(
            id=id_str,
            type=MemoryType(type_str),
            content=eval(data["content"]),
            metadata=eval(data["metadata"]),
            importance_score=float(data["importance"]),
            timestamp=datetime.fromisoformat(data["timestamp"])
        )
    
    def _matches_query(self, memory: Memory, query: MemoryQuery) -> bool:
        """Check if a memory matches the query criteria."""
        if query.type and memory.type != query.type:
            return False
            
        if query.experience_type and memory.experience_type != query.experience_type:
            return False
            
        if query.time_range:
            start, end = query.time_range
            if not (start <= memory.timestamp <= end):
                return False
                
        if query.content_filter:
            for key, value in query.content_filter.items():
                if key not in memory.content or memory.content[key] != value:
                    return False
        
        return True

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("[Memory] Starting up Memory Module...")
    app.state.memory_system = MemorySystem(redis_client)
    yield
    logger.info("[Memory] Shutting down Memory Module...")

app = FastAPI(
    title="HMAS Memory Module",
    version="1.0.0",
    lifespan=lifespan
)

# Redis client for potential caching (if needed)
redis_client = redis.Redis.from_url(settings.REDIS_URL, decode_responses=True)

# Rate limiting
limiter = Limiter(key_func=get_remote_address, default_limits=["10/minute"])
app.state.limiter = limiter

# Prometheus monitoring
Instrumentator().instrument(app).expose(app, endpoint="/metrics")

# Middleware
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.get("/ready")
async def ready_check():
    try:
        await redis_client.ping()
        return {"status": "ready"}
    except Exception as e:
        logger.warning(f"[Memory] Redis not ready: {e}")
        raise HTTPException(status_code=500, detail="Redis not ready")

@app.post("/memory")
@limiter.limit("10/minute")
async def store_memory(request: Request, memory: Memory):
    """Store a new memory in the system."""
    try:
        await app.state.memory_system.store(memory)
        return {"status": "success", "memory_id": memory.id}
    except Exception as e:
        logger.error(f"[Memory] Error storing memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/memory/query")
@limiter.limit("20/minute")
async def query_memories(request: Request, query: MemoryQuery):
    """Query memories based on specified criteria."""
    try:
        memories = await app.state.memory_system.retrieve(query)
        return {"memories": [m.dict() for m in memories]}
    except Exception as e:
        logger.error(f"[Memory] Error querying memories: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/memory/replay")
@limiter.limit("30/minute")
async def get_replay_batch(
    request: Request,
    batch_size: int = Query(default=32, gt=0, le=128)
):
    """Get a batch of memories for experience replay."""
    try:
        memories = app.state.memory_system.replay_buffer.sample(batch_size)
        return {"memories": [m.dict() for m in memories]}
    except Exception as e:
        logger.error(f"[Memory] Error sampling replay buffer: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/memory/similar")
@limiter.limit("20/minute")
async def find_similar_memories(
    request: Request,
    embedding: List[float],
    k: int = Query(default=5, gt=0, le=20)
):
    """Find similar memories based on embedding."""
    try:
        similar = app.state.memory_system.replay_buffer.get_similar(embedding, k)
        return {
            "memories": [
                {"memory": m.dict(), "similarity": s}
                for m, s in similar
            ]
        }
    except Exception as e:
        logger.error(f"[Memory] Error finding similar memories: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("memory_main:app", host="0.0.0.0", port=8401, reload=True)
