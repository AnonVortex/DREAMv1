import os
import logging.config
from typing import Dict, List, Optional, Any, Union, Set
from enum import Enum
from datetime import datetime
import json
import asyncio
import aiohttp
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from prometheus_client import Counter, Histogram, start_http_server
import redis
from redis.client import Redis

# Import configuration from local config.py
from .config import settings

# Load logging configuration if available
LOGGING_CONFIG_PATH = "logging.conf"
if os.path.exists(LOGGING_CONFIG_PATH):
    logging.config.fileConfig(LOGGING_CONFIG_PATH, disable_existing_loggers=False)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("[Routing] Starting up Routing Module...")
    await startup_event()
    yield
    logger.info("[Routing] Shutting down Routing Module...")
    await shutdown_event()

app = FastAPI(
    title="HMAS Routing Module",
    version="1.0.0",
    lifespan=lifespan
)

# Rate limiting
limiter = Limiter(key_func=get_remote_address, default_limits=["10/minute"])
app.state.limiter = limiter
app.add_exception_handler(429, _rate_limit_exceeded_handler)

# Prometheus monitoring
REQUESTS = Counter('routing_requests_total', 'Total requests to routing service')
LATENCY = Histogram('routing_request_latency_seconds', 'Request latency in seconds')

# Middleware to trust any host (adjust as needed)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])

# Initialize Redis client
redis_client = redis.Redis(
    host=os.getenv('REDIS_HOST', 'localhost'),
    port=int(os.getenv('REDIS_PORT', 6379)),
    decode_responses=True
)

class TaskType(str, Enum):
    PERCEPTION = "perception"
    MEMORY = "memory"
    LEARNING = "learning"
    REASONING = "reasoning"
    COMMUNICATION = "communication"
    FEEDBACK = "feedback"
    SPECIALIZED = "specialized"

class TaskPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class RoutingStrategy(str, Enum):
    ROUND_ROBIN = "round_robin"
    LEAST_LOAD = "least_load"
    PRIORITY_BASED = "priority_based"
    CAPABILITY_BASED = "capability_based"

class ServiceStatus(str, Enum):
    ACTIVE = "active"
    BUSY = "busy"
    INACTIVE = "inactive"
    ERROR = "error"

class TaskRequest(BaseModel):
    task_id: str = Field(default_factory=lambda: datetime.now().isoformat())
    task_type: TaskType
    priority: TaskPriority = TaskPriority.MEDIUM
    data: Dict[str, Any]
    requirements: Optional[Dict[str, Any]] = None
    timeout: Optional[float] = 30.0

class ServiceInfo(BaseModel):
    service_id: str
    service_type: TaskType
    endpoint: str
    capabilities: Set[str]
    status: ServiceStatus = ServiceStatus.ACTIVE
    current_load: float = 0.0
    last_heartbeat: datetime = Field(default_factory=datetime.utcnow)

class RoutingConfig(BaseModel):
    strategy: RoutingStrategy
    load_threshold: float = 0.8
    timeout_threshold: float = 5.0
    retry_count: int = 3
    health_check_interval: float = 30.0

class RoutingManager:
    def __init__(self):
        self.services: Dict[str, ServiceInfo] = {}
        self.task_history: Dict[str, Dict[str, Any]] = {}
        self.session: Optional[aiohttp.ClientSession] = None
        self.config = RoutingConfig(strategy=RoutingStrategy.LEAST_LOAD)
    
    async def start(self):
        """Initialize the aiohttp session."""
        self.session = aiohttp.ClientSession()
        # Start health check loop
        asyncio.create_task(self.health_check_loop())
    
    async def stop(self):
        """Clean up resources."""
        if self.session:
            await self.session.close()
    
    async def register_service(self, service: ServiceInfo):
        """Register a new service."""
        self.services[service.service_id] = service
        await self._update_service_registry()
    
    async def unregister_service(self, service_id: str):
        """Unregister a service."""
        if service_id in self.services:
            del self.services[service_id]
            await self._update_service_registry()
    
    async def _update_service_registry(self):
        """Update service registry in Redis."""
        registry = {
            service_id: service.dict()
            for service_id, service in self.services.items()
        }
        await redis_client.set('service_registry', json.dumps(registry))
    
    async def health_check_loop(self):
        """Continuously check the health of registered services."""
        while True:
            for service_id, service in list(self.services.items()):
                try:
                    async with self.session.get(
                        f"{service.endpoint}/health",
                        timeout=self.config.timeout_threshold
                    ) as response:
                        if response.status == 200:
                            service.status = ServiceStatus.ACTIVE
                            service.last_heartbeat = datetime.utcnow()
                        else:
                            service.status = ServiceStatus.INACTIVE
                except Exception as e:
                    logger.error(f"Health check failed for {service_id}: {str(e)}")
                    service.status = ServiceStatus.INACTIVE
            
            # Remove inactive services
            current_time = datetime.utcnow()
            for service_id, service in list(self.services.items()):
                if (current_time - service.last_heartbeat).total_seconds() > 60:
                    await self.unregister_service(service_id)
            
            await asyncio.sleep(self.config.health_check_interval)
    
    def _select_service_round_robin(self, task_type: TaskType) -> Optional[ServiceInfo]:
        """Select service using round-robin strategy."""
        available_services = [
            service for service in self.services.values()
            if service.service_type == task_type and service.status == ServiceStatus.ACTIVE
        ]
        if not available_services:
            return None
        
        # Get last used index from Redis
        last_index = int(redis_client.get(f'round_robin_{task_type}') or '0')
        next_index = (last_index + 1) % len(available_services)
        redis_client.set(f'round_robin_{task_type}', str(next_index))
        
        return available_services[next_index]
    
    def _select_service_least_load(self, task_type: TaskType) -> Optional[ServiceInfo]:
        """Select service with least load."""
        available_services = [
            service for service in self.services.values()
            if service.service_type == task_type and service.status == ServiceStatus.ACTIVE
        ]
        if not available_services:
            return None
        
        return min(available_services, key=lambda s: s.current_load)
    
    def _select_service_priority(self, task: TaskRequest) -> Optional[ServiceInfo]:
        """Select service based on task priority."""
        available_services = [
            service for service in self.services.values()
            if service.service_type == task.task_type and service.status == ServiceStatus.ACTIVE
        ]
        if not available_services:
            return None
        
        # For high priority tasks, select service with least load
        if task.priority in [TaskPriority.HIGH, TaskPriority.CRITICAL]:
            return min(available_services, key=lambda s: s.current_load)
        
        # For lower priority tasks, use round-robin
        return self._select_service_round_robin(task.task_type)
    
    def _select_service_capability(self, task: TaskRequest) -> Optional[ServiceInfo]:
        """Select service based on required capabilities."""
        if not task.requirements:
            return self._select_service_least_load(task.task_type)
        
        required_capabilities = set(task.requirements.get('capabilities', []))
        available_services = [
            service for service in self.services.values()
            if service.service_type == task.task_type
            and service.status == ServiceStatus.ACTIVE
            and required_capabilities.issubset(service.capabilities)
        ]
        
        if not available_services:
            return None
        
        return min(available_services, key=lambda s: s.current_load)
    
    async def route_task(self, task: TaskRequest) -> Dict[str, Any]:
        """Route a task to an appropriate service."""
        # Select service based on strategy
        service = None
        if self.config.strategy == RoutingStrategy.ROUND_ROBIN:
            service = self._select_service_round_robin(task.task_type)
        elif self.config.strategy == RoutingStrategy.LEAST_LOAD:
            service = self._select_service_least_load(task.task_type)
        elif self.config.strategy == RoutingStrategy.PRIORITY_BASED:
            service = self._select_service_priority(task)
        elif self.config.strategy == RoutingStrategy.CAPABILITY_BASED:
            service = self._select_service_capability(task)
        
        if not service:
            raise HTTPException(
                status_code=503,
                detail=f"No available services for task type {task.task_type}"
            )
        
        # Forward task to selected service
        try:
            async with self.session.post(
                f"{service.endpoint}/process",
                json=task.dict(),
                timeout=task.timeout
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    # Update service load
                    service.current_load = float(response.headers.get('X-Service-Load', '0'))
                    return result
                else:
                    raise HTTPException(
                        status_code=response.status,
                        detail="Service processing failed"
                    )
        except asyncio.TimeoutError:
            service.status = ServiceStatus.BUSY
            raise HTTPException(status_code=504, detail="Service timeout")
        except Exception as e:
            logger.error(f"Error routing task: {str(e)}")
            raise HTTPException(status_code=500, detail="Routing failed")

# Initialize RoutingManager
routing_manager = RoutingManager()

@app.on_event("startup")
async def startup_event():
    await routing_manager.start()

@app.on_event("shutdown")
async def shutdown_event():
    await routing_manager.stop()

@app.post("/register")
@limiter.limit("50/minute")
async def register_service(service: ServiceInfo, request: Request):
    """Register a new service."""
    REQUESTS.inc()
    with LATENCY.time():
        await routing_manager.register_service(service)
        return {"status": "success", "service_id": service.service_id}

@app.delete("/unregister/{service_id}")
@limiter.limit("50/minute")
async def unregister_service(service_id: str, request: Request):
    """Unregister a service."""
    REQUESTS.inc()
    with LATENCY.time():
        await routing_manager.unregister_service(service_id)
        return {"status": "success"}

@app.post("/route")
@limiter.limit("1000/minute")
async def route_task(task: TaskRequest, request: Request):
    """Route a task to an appropriate service."""
    REQUESTS.inc()
    with LATENCY.time():
        result = await routing_manager.route_task(task)
        return result

@app.get("/services")
@limiter.limit("100/minute")
async def list_services(request: Request):
    """List all registered services."""
    REQUESTS.inc()
    with LATENCY.time():
        return routing_manager.services

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.utcnow()}

# Start Prometheus metrics server
start_http_server(8901)

# Run the service
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8900)
