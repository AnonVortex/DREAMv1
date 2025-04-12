"""
Memory service configuration module.
Extends the base HMAS configuration with memory-specific settings.
"""

from typing import List, Optional, Dict, Any
from enum import Enum
from pydantic import Field

from shared.config import ServiceConfig

class StorageType(str, Enum):
    """Types of memory storage"""
    REDIS = "redis"
    MONGODB = "mongodb"
    POSTGRESQL = "postgresql"
    HYBRID = "hybrid"

class MemoryConfig(ServiceConfig):
    """
    Memory service specific configuration.
    Extends the base service configuration with memory-specific settings.
    """
    # Storage Configuration
    storage_type: StorageType = Field(
        default=StorageType.REDIS,
        description="Type of storage backend to use"
    )
    redis_url: str = Field(
        default="redis://localhost:6379",
        description="Redis connection URL"
    )
    mongodb_url: Optional[str] = Field(
        default=None,
        description="MongoDB connection URL for hybrid storage"
    )
    postgresql_url: Optional[str] = Field(
        default=None,
        description="PostgreSQL connection URL for hybrid storage"
    )
    
    # Memory Management
    max_memory_size: int = Field(
        default=1024 * 1024 * 1024,  # 1GB
        description="Maximum memory size in bytes"
    )
    eviction_policy: str = Field(
        default="lru",
        description="Memory eviction policy (lru, lfu, random)"
    )
    
    # Caching Configuration
    enable_caching: bool = Field(
        default=True,
        description="Enable memory caching"
    )
    cache_ttl: int = Field(
        default=3600,
        description="Default cache TTL in seconds"
    )
    max_cache_size: int = Field(
        default=100000,
        description="Maximum number of cached items"
    )
    
    # Persistence Configuration
    enable_persistence: bool = Field(
        default=True,
        description="Enable data persistence"
    )
    snapshot_interval: int = Field(
        default=3600,
        description="Snapshot interval in seconds"
    )
    backup_count: int = Field(
        default=3,
        description="Number of backup snapshots to maintain"
    )
    
    # Query Configuration
    max_query_time: int = Field(
        default=30,
        description="Maximum query execution time in seconds"
    )
    max_results: int = Field(
        default=1000,
        description="Maximum number of results per query"
    )
    
    def get_storage_config(self) -> Dict[str, Any]:
        """Get storage configuration dictionary"""
        return {
            "type": self.storage_type,
            "redis_url": self.redis_url,
            "mongodb_url": self.mongodb_url,
            "postgresql_url": self.postgresql_url,
            "max_size": self.max_memory_size,
            "eviction_policy": self.eviction_policy
        }
    
    def get_cache_config(self) -> Dict[str, Any]:
        """Get cache configuration dictionary"""
        return {
            "enabled": self.enable_caching,
            "ttl": self.cache_ttl,
            "max_size": self.max_cache_size
        }
    
    def get_persistence_config(self) -> Dict[str, Any]:
        """Get persistence configuration dictionary"""
        return {
            "enabled": self.enable_persistence,
            "snapshot_interval": self.snapshot_interval,
            "backup_count": self.backup_count
        }
    
    def get_query_config(self) -> Dict[str, Any]:
        """Get query configuration dictionary"""
        return {
            "max_time": self.max_query_time,
            "max_results": self.max_results
        }
    
    class Config:
        """Pydantic configuration"""
        env_prefix = "MEMORY_"

def load_memory_config(config_file: Optional[str] = None) -> MemoryConfig:
    """
    Load memory service configuration.
    
    Args:
        config_file: Optional path to config file
        
    Returns:
        MemoryConfig instance
    """
    return MemoryConfig(
        service_name="memory",
        config_file=config_file if config_file else None
    )
