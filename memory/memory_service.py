import os
import logging.config
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Any, Union, Set
from enum import Enum
from datetime import datetime, timedelta
import json
import asyncio
import numpy as np
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.util import get_remote_address
from prometheus_fastapi_instrumentator import Instrumentator
import redis
from redis.client import Redis
from pymongo import MongoClient
from pymongo.collection import Collection
from sklearn.metrics.pairwise import cosine_similarity

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Setup logging
LOGGING_CONFIG_PATH = "logging.conf"
if os.path.exists(LOGGING_CONFIG_PATH):
    logging.config.fileConfig(LOGGING_CONFIG_PATH, disable_existing_loggers=False)
logger = logging.getLogger(__name__)

class StorageType(str, Enum):
    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"
    WORKING = "working"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"

class MemoryPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class MemoryStatus(str, Enum):
    ACTIVE = "active"
    ARCHIVED = "archived"
    DELETED = "deleted"

class MemoryEntry(BaseModel):
    entry_id: str
    storage_type: StorageType
    content: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None
    priority: MemoryPriority = MemoryPriority.MEDIUM
    status: MemoryStatus = MemoryStatus.ACTIVE
    created_at: datetime = Field(default_factory=datetime.now)
    last_accessed: datetime = Field(default_factory=datetime.now)
    access_count: int = 0
    tags: List[str] = []

class MemoryQuery(BaseModel):
    storage_type: Optional[StorageType] = None
    query: Dict[str, Any]
    filters: Optional[Dict[str, Any]] = None
    sort_by: Optional[str] = None
    sort_order: Optional[str] = "desc"
    limit: Optional[int] = 10
    offset: Optional[int] = 0

class MemoryResult(BaseModel):
    query_id: str
    entries: List[MemoryEntry]
    total_count: int
    metadata: Optional[Dict[str, Any]] = None

class MemoryConfig(BaseModel):
    memory_types: List[StorageType]
    storage_types: List[StorageType]
    cache_size: int = 1000
    ttl_default: int = 3600  # 1 hour
    similarity_threshold: float = 0.7

class CacheManager:
    def __init__(self, redis_url: str, max_size: int = 1000):
        self.redis: Redis = redis.from_url(redis_url)
        self.max_size = max_size
        
    async def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        try:
            value = self.redis.get(key)
            if value:
                return json.loads(value)
            return None
        except Exception as e:
            logger.error(f"Error getting from cache: {str(e)}")
            return None
            
    async def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set item in cache with optional TTL."""
        try:
            # Convert value to JSON string
            value_str = json.dumps(value)
            
            # Check cache size
            if self.redis.dbsize() >= self.max_size:
                # Remove oldest items
                keys = self.redis.keys()
                keys.sort(key=lambda k: float(self.redis.get(f"{k}:timestamp") or 0))
                for old_key in keys[:len(keys) - self.max_size + 1]:
                    self.redis.delete(old_key)
                    
            # Store item with timestamp
            self.redis.set(key, value_str)
            self.redis.set(f"{key}:timestamp", datetime.now().timestamp())
            
            if ttl:
                self.redis.expire(key, ttl)
                
        except Exception as e:
            logger.error(f"Error setting cache: {str(e)}")
            
    async def delete(self, key: str):
        """Delete item from cache."""
        try:
            self.redis.delete(key)
            self.redis.delete(f"{key}:timestamp")
        except Exception as e:
            logger.error(f"Error deleting from cache: {str(e)}")

class StorageManager:
    def __init__(self, mongo_url: str, database: str = "hmas_memory"):
        self.client = MongoClient(mongo_url)
        self.db = self.client[database]
        self.collections: Dict[StorageType, Collection] = {
            storage_type: self.db[storage_type.value]
            for storage_type in StorageType
        }
        
        # Create indexes
        for collection in self.collections.values():
            collection.create_index("timestamp")
            collection.create_index("metadata")
            collection.create_index([("embeddings", 1)])
            
    async def store(self, item: MemoryEntry) -> str:
        """Store memory item."""
        try:
            collection = self.collections[item.storage_type]
            result = collection.insert_one(item.dict())
            return str(result.inserted_id)
        except Exception as e:
            logger.error(f"Error storing item: {str(e)}")
            raise
            
    async def retrieve(self, storage_type: StorageType, item_id: str) -> Optional[MemoryEntry]:
        """Retrieve memory item by ID."""
        try:
            collection = self.collections[storage_type]
            item = collection.find_one({"entry_id": item_id})
            return MemoryEntry(**item) if item else None
        except Exception as e:
            logger.error(f"Error retrieving item: {str(e)}")
            return None
            
    async def search(
        self,
        storage_type: StorageType,
        query: Dict[str, Any],
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 10
    ) -> List[MemoryEntry]:
        """Search memory items."""
        try:
            collection = self.collections[storage_type]
            
            # Combine query and filters
            search_query = query.copy()
            if filters:
                search_query.update(filters)
                
            cursor = collection.find(search_query).limit(limit)
            return [MemoryEntry(**item) for item in cursor]
            
        except Exception as e:
            logger.error(f"Error searching items: {str(e)}")
            return []
            
    async def update(self, item: MemoryEntry) -> bool:
        """Update memory item."""
        try:
            collection = self.collections[item.storage_type]
            result = collection.update_one(
                {"entry_id": item.entry_id},
                {"$set": item.dict()}
            )
            return result.modified_count > 0
        except Exception as e:
            logger.error(f"Error updating item: {str(e)}")
            return False
            
    async def delete(self, storage_type: StorageType, item_id: str) -> bool:
        """Delete memory item."""
        try:
            collection = self.collections[storage_type]
            result = collection.delete_one({"entry_id": item_id})
            return result.deleted_count > 0
        except Exception as e:
            logger.error(f"Error deleting item: {str(e)}")
            return False

class QueryEngine:
    def __init__(self, storage_manager: StorageManager):
        self.storage_manager = storage_manager
        
    async def search(self, query: MemoryQuery) -> MemoryResult:
        """Search for memory entries."""
        entries = []
        
        # Get all entries of specified storage type
        storage = (
            self.storage_manager.collections[query.storage_type]
            if query.storage_type
            else {
                entry_id: entry
                for storage in self.storage_manager.collections.values()
                for entry_id, entry in storage.items()
            }
        )
        
        # Apply query filters
        filtered_entries = self._apply_filters(storage.values(), query)
        
        # Sort results
        if query.sort_by:
            filtered_entries = sorted(
                filtered_entries,
                key=lambda x: getattr(x, query.sort_by),
                reverse=query.sort_order == "desc"
            )
            
        # Apply pagination
        total_count = len(filtered_entries)
        start_idx = query.offset if query.offset else 0
        end_idx = start_idx + (query.limit if query.limit else total_count)
        entries = filtered_entries[start_idx:end_idx]
        
        return MemoryResult(
            query_id=f"query_{datetime.now().isoformat()}",
            entries=entries,
            total_count=total_count,
            metadata={
                "filters_applied": bool(query.filters),
                "sort_by": query.sort_by,
                "sort_order": query.sort_order
            }
        )
        
    def _apply_filters(
        self,
        entries: List[MemoryEntry],
        query: MemoryQuery
    ) -> List[MemoryEntry]:
        """Apply filters to entries."""
        if not query.filters:
            return list(entries)
            
        filtered = []
        for entry in entries:
            if self._matches_filters(entry, query.filters):
                filtered.append(entry)
                
        return filtered
        
    def _matches_filters(self, entry: MemoryEntry, filters: Dict[str, Any]) -> bool:
        """Check if entry matches filters."""
        for key, value in filters.items():
            if key == "tags":
                if not all(tag in entry.tags for tag in value):
                    return False
            elif key == "date_range":
                if not (
                    value["start"] <= entry.created_at <= value["end"]
                ):
                    return False
            elif key == "priority":
                if entry.priority != value:
                    return False
            elif key == "status":
                if entry.status != value:
                    return False
            elif key == "content":
                if not self._content_matches(entry.content, value):
                    return False
        return True
        
    def _content_matches(self, content: Dict[str, Any], query: Dict[str, Any]) -> bool:
        """Check if content matches query."""
        for key, value in query.items():
            if key not in content:
                return False
            if isinstance(value, dict):
                if not self._content_matches(content[key], value):
                    return False
            elif content[key] != value:
                return False
        return True

class MemoryManager:
    def __init__(self):
        # Initialize storage managers
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        mongo_url = os.getenv("MONGO_URL", "mongodb://localhost:27017/")
        
        self.cache = CacheManager(redis_url)
        self.storage = StorageManager(mongo_url)
        
        self.configs: Dict[str, MemoryConfig] = {}
        
        self.query_engine = QueryEngine(self.storage)
        
    def register_config(self, config_id: str, config: MemoryConfig):
        """Register a memory configuration."""
        self.configs[config_id] = config
        
    async def store_memory(
        self,
        entry: MemoryEntry,
        background_tasks: BackgroundTasks
    ) -> str:
        """Store a new memory entry."""
        try:
            entry_id = await self.storage.store(entry)
            return entry_id
        except Exception as e:
            logger.error(f"Error storing memory: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
            
    async def retrieve_memory(
        self,
        entry_id: str,
        storage_type: StorageType
    ) -> MemoryEntry:
        """Retrieve a memory entry."""
        entry = await self.storage.retrieve(storage_type, entry_id)
        if not entry:
            raise HTTPException(
                status_code=404,
                detail=f"Memory entry {entry_id} not found"
            )
        return entry
        
    async def update_memory(
        self,
        entry: MemoryEntry,
        background_tasks: BackgroundTasks
    ) -> bool:
        """Update a memory entry."""
        try:
            success = await self.storage.update(entry)
            if not success:
                raise HTTPException(
                    status_code=404,
                    detail=f"Memory entry {entry.entry_id} not found"
                )
            return success
        except HTTPException as e:
            raise e
        except Exception as e:
            logger.error(f"Error updating memory: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
            
    async def delete_memory(
        self,
        entry_id: str,
        storage_type: StorageType
    ) -> bool:
        """Delete a memory entry."""
        try:
            success = await self.storage.delete(storage_type, entry_id)
            if not success:
                raise HTTPException(
                    status_code=404,
                    detail=f"Memory entry {entry_id} not found"
                )
            return success
        except HTTPException as e:
            raise e
        except Exception as e:
            logger.error(f"Error deleting memory: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
            
    async def search_memories(
        self,
        query: MemoryQuery,
        background_tasks: BackgroundTasks
    ) -> MemoryResult:
        """Search memory entries."""
        try:
            return await self.query_engine.search(query)
        except Exception as e:
            logger.error(f"Error searching memories: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

# Initialize FastAPI app
@asynccontextmanager
async def lifespan(app: FastAPI):
    # STARTUP
    logger.info("Initializing memory service...")
    try:
        memory_manager = MemoryManager()
        app.state.memory_manager = memory_manager
        logger.info("Memory service initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize memory service: {str(e)}")
        raise
        
    yield
    
    # SHUTDOWN
    logger.info("Shutting down memory service...")

app = FastAPI(title="HMAS Memory Service", lifespan=lifespan)

# Rate limiting
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

# Prometheus metrics
Instrumentator().instrument(app).expose(app)

@app.post("/config/{config_id}")
@limiter.limit("20/minute")
async def register_config(
    request: Request,
    config_id: str,
    config: MemoryConfig
):
    """Register a memory configuration."""
    try:
        request.app.state.memory_manager.register_config(config_id, config)
        return {"status": "success", "config_id": config_id}
    except Exception as e:
        logger.error(f"Error registering configuration: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/memory")
@limiter.limit("100/minute")
async def store_memory(
    request: Request,
    entry: MemoryEntry,
    background_tasks: BackgroundTasks
):
    """Store a new memory entry."""
    try:
        entry_id = await request.app.state.memory_manager.store_memory(
            entry,
            background_tasks
        )
        return {"status": "success", "entry_id": entry_id}
    except Exception as e:
        logger.error(f"Error storing memory: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/memory/{storage_type}/{entry_id}")
@limiter.limit("200/minute")
async def retrieve_memory(
    request: Request,
    storage_type: StorageType,
    entry_id: str
):
    """Retrieve a memory entry."""
    try:
        entry = await request.app.state.memory_manager.retrieve_memory(
            entry_id,
            storage_type
        )
        return entry.dict()
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error retrieving memory: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/memory/search")
@limiter.limit("50/minute")
async def search_memories(
    request: Request,
    query: MemoryQuery,
    background_tasks: BackgroundTasks
):
    """Search memory entries."""
    try:
        result = await request.app.state.memory_manager.search_memories(
            query,
            background_tasks
        )
        return result.dict()
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error searching memories: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/memory")
@limiter.limit("50/minute")
async def update_memory(
    request: Request,
    entry: MemoryEntry,
    background_tasks: BackgroundTasks
):
    """Update a memory entry."""
    try:
        success = await request.app.state.memory_manager.update_memory(
            entry,
            background_tasks
        )
        return {"status": "success", "updated": success}
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error updating memory: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/memory/{storage_type}/{entry_id}")
@limiter.limit("30/minute")
async def delete_memory(
    request: Request,
    storage_type: StorageType,
    entry_id: str
):
    """Delete a memory entry."""
    try:
        success = await request.app.state.memory_manager.delete_memory(
            entry_id,
            storage_type
        )
        return {"status": "success", "deleted": success}
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error deleting memory: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8800) 