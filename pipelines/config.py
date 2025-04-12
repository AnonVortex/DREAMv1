import os
from dotenv import load_dotenv
from typing import Dict, Any
from pydantic import BaseSettings

load_dotenv()

class PipelineConfig(BaseSettings):
    # Service Configuration
    MODULE_NAME: str = "pipeline_service"
    HOST: str = "0.0.0.0"
    PORT: int = 8500
    LOG_LEVEL: str = "INFO"
    
    # Pipeline Execution Settings
    MAX_CONCURRENT_PIPELINES: int = 10
    PIPELINE_TIMEOUT_SECONDS: int = 3600
    MAX_RETRIES: int = 3
    RETRY_DELAY_SECONDS: int = 60
    
    # Resource Management
    MAX_MEMORY_MB: int = 4096
    MAX_CPU_CORES: int = 4
    DISK_SPACE_LIMIT_MB: int = 10240
    
    # Monitoring Settings
    ENABLE_METRICS: bool = True
    METRICS_PORT: int = 8501
    HEALTH_CHECK_INTERVAL: int = 30
    
    # Storage Settings
    PIPELINE_STORAGE_PATH: str = "pipeline_data"
    TEMP_STORAGE_PATH: str = "temp_data"
    
    # Redis Configuration
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", 6379))
    REDIS_DB: int = int(os.getenv("REDIS_DB", 0))
    
    # MongoDB Configuration
    MONGO_URI: str = os.getenv("MONGO_URI", "mongodb://localhost:27017")
    MONGO_DB: str = os.getenv("MONGO_DB", "pipeline_db")
    
    # Rate Limiting
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_WINDOW_SECONDS: int = 60
    
    # Pipeline Execution Settings
    max_concurrent_pipelines: int = int(os.getenv("MAX_CONCURRENT_PIPELINES", "10"))
    max_pipeline_retries: int = int(os.getenv("MAX_PIPELINE_RETRIES", "3"))
    retry_delay_seconds: int = int(os.getenv("RETRY_DELAY_SECONDS", "5"))
    pipeline_timeout_seconds: int = int(os.getenv("PIPELINE_TIMEOUT_SECONDS", "3600"))
    
    # Pipeline Dependencies
    enforce_dependencies: bool = bool(os.getenv("ENFORCE_DEPENDENCIES", "True"))
    dependency_timeout_seconds: int = int(os.getenv("DEPENDENCY_TIMEOUT_SECONDS", "600"))
    
    # Error Handling
    error_notification_url: str = os.getenv("ERROR_NOTIFICATION_URL", "")
    error_notification_threshold: int = int(os.getenv("ERROR_NOTIFICATION_THRESHOLD", "5"))
    critical_error_types: list = os.getenv("CRITICAL_ERROR_TYPES", "DataCorruption,SystemFailure").split(",")
    
    # Metrics Collection
    enable_detailed_metrics: bool = bool(os.getenv("ENABLE_DETAILED_METRICS", "True"))
    metrics_collection_interval: int = int(os.getenv("METRICS_COLLECTION_INTERVAL", "60"))
    performance_threshold_ms: int = int(os.getenv("PERFORMANCE_THRESHOLD_MS", "1000"))
    metrics_retention_days: int = int(os.getenv("METRICS_RETENTION_DAYS", "30"))
    
    class Config:
        env_file = ".env"
        
    def get_redis_config(self) -> Dict[str, Any]:
        return {
            "host": self.REDIS_HOST,
            "port": self.REDIS_PORT,
            "db": self.REDIS_DB
        }
        
    def get_mongo_config(self) -> Dict[str, Any]:
        return {
            "uri": self.MONGO_URI,
            "db": self.MONGO_DB
        }

    def get_execution_settings(self) -> Dict[str, Any]:
        """Get pipeline execution settings."""
        return {
            "max_concurrent_pipelines": self.max_concurrent_pipelines,
            "max_retries": self.max_pipeline_retries,
            "retry_delay": self.retry_delay_seconds,
            "timeout": self.pipeline_timeout_seconds,
            "enforce_dependencies": self.enforce_dependencies,
            "dependency_timeout": self.dependency_timeout_seconds
        }

    def get_error_handling_config(self) -> Dict[str, Any]:
        """Get error handling configuration."""
        return {
            "notification_url": self.error_notification_url,
            "notification_threshold": self.error_notification_threshold,
            "critical_errors": self.critical_error_types
        }

    def get_metrics_config(self) -> Dict[str, Any]:
        """Get metrics collection configuration."""
        return {
            "enable_detailed": self.enable_detailed_metrics,
            "collection_interval": self.metrics_collection_interval,
            "performance_threshold": self.performance_threshold_ms,
            "retention_days": self.metrics_retention_days
        }

settings = PipelineConfig()

# Service configuration
SERVICE_NAME = "pipeline_service"
HOST = os.getenv("PIPELINE_SERVICE_HOST", "0.0.0.0")
PORT = int(os.getenv("PIPELINE_SERVICE_PORT", "8600"))

# Redis configuration
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))

# MongoDB configuration
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB = os.getenv("MONGO_DB", "pipeline_service")

# Rate limiting
RATE_LIMIT = os.getenv("RATE_LIMIT", "100/minute")

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Pipeline settings
MAX_PIPELINE_LENGTH = int(os.getenv("MAX_PIPELINE_LENGTH", "50"))
MAX_CONCURRENT_PIPELINES = int(os.getenv("MAX_CONCURRENT_PIPELINES", "10"))
PIPELINE_TIMEOUT = int(os.getenv("PIPELINE_TIMEOUT", "3600"))  # 1 hour in seconds
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
RETRY_DELAY = int(os.getenv("RETRY_DELAY", "5"))  # seconds

# Service endpoints
MEMORY_SERVICE_URL = os.getenv("MEMORY_SERVICE_URL", "http://localhost:8300")
LEARNING_SERVICE_URL = os.getenv("LEARNING_SERVICE_URL", "http://localhost:8200")
REASONING_SERVICE_URL = os.getenv("REASONING_SERVICE_URL", "http://localhost:8400")
PERCEPTION_SERVICE_URL = os.getenv("PERCEPTION_SERVICE_URL", "http://localhost:8100")

# Service URLs
SERVICE_URLS: Dict[str, str] = {
    "perception": os.getenv("PERCEPTION_URL", "http://localhost:8100"),
    "memory": os.getenv("MEMORY_URL", "http://localhost:8200"),
    "learning": os.getenv("LEARNING_URL", "http://localhost:8300"),
    "reasoning": os.getenv("REASONING_URL", "http://localhost:8400"),
    "communication": os.getenv("COMMUNICATION_URL", "http://localhost:8500"),
    "feedback": os.getenv("FEEDBACK_URL", "http://localhost:8600"),
    "specialized": os.getenv("SPECIALIZED_URL", "http://localhost:8700")
}

# Pipeline configuration
DEFAULT_STEP_TIMEOUT = float(os.getenv("DEFAULT_STEP_TIMEOUT", 60.0))
DEFAULT_RETRY_COUNT = int(os.getenv("DEFAULT_RETRY_COUNT", 3))
MAX_PIPELINE_STEPS = int(os.getenv("MAX_PIPELINE_STEPS", 100))

# Logging configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = os.getenv("LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")

class PipelineSettings:
    INGESTION_URL = os.getenv("INGESTION_URL", "http://localhost:8000/ingest")
    PERCEPTION_URL = os.getenv("PERCEPTION_URL", "http://localhost:8100/perceive")
    INTEGRATION_URL = os.getenv("INTEGRATION_URL", "http://localhost:8200/integrate")
    ROUTING_URL = os.getenv("ROUTING_URL", "http://localhost:8300/route")
    SPECIALIZED_URL = os.getenv("SPECIALIZED_URL", "http://localhost:8400/specialize")
    META_URL = os.getenv("META_URL", "http://localhost:8301/meta")
    MEMORY_URL = os.getenv("MEMORY_URL", "http://localhost:8401/memory")
    AGGREGATION_URL = os.getenv("AGGREGATION_URL", "http://localhost:8500/aggregate")
    FEEDBACK_URL = os.getenv("FEEDBACK_URL", "http://localhost:8600/feedback")
    MONITORING_URL = os.getenv("MONITORING_URL", "http://localhost:8700/monitor")
    GRAPH_RL_URL = os.getenv("GRAPH_RL_URL", "http://localhost:8800/graph_rl")
    COMM_URL = os.getenv("COMM_URL", "http://localhost:8900/optimize")
    PIPELINE_PORT = int(os.getenv("PIPELINE_PORT", "9000"))

settings = PipelineSettings()
