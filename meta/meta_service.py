import os
import logging.config
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from datetime import datetime
import json
import asyncio
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from prometheus_fastapi_instrumentator import Instrumentator
import numpy as np
from prometheus_client import Counter, Histogram, start_http_server
import redis
from pymongo import MongoClient
import psutil
import docker

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Setup logging
LOGGING_CONFIG_PATH = "logging.conf"
if os.path.exists(LOGGING_CONFIG_PATH):
    logging.config.fileConfig(LOGGING_CONFIG_PATH, disable_existing_loggers=False)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="HMAS Meta Service")

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(429, _rate_limit_exceeded_handler)

# Initialize metrics
REQUESTS = Counter('meta_requests_total', 'Total requests to meta service')
LATENCY = Histogram('meta_request_latency_seconds', 'Request latency in seconds')

# Initialize Redis and MongoDB clients
redis_client = redis.Redis(host=os.getenv('REDIS_HOST', 'localhost'), 
                         port=int(os.getenv('REDIS_PORT', 6379)), 
                         decode_responses=True)

mongo_client = MongoClient(os.getenv('MONGO_URI', 'mongodb://localhost:27017'))
db = mongo_client.hmas_meta

# Initialize Docker client
docker_client = docker.from_env()

class OptimizationType(str, Enum):
    RESOURCE = "resource"
    PERFORMANCE = "performance"
    RELIABILITY = "reliability"
    COST = "cost"
    ENERGY = "energy"

class SystemComponent(str, Enum):
    PERCEPTION = "perception"
    REASONING = "reasoning"
    MEMORY = "memory"
    LEARNING = "learning"
    COMMUNICATION = "communication"
    INTEGRATION = "integration"
    MONITORING = "monitoring"

class OptimizationObjective(BaseModel):
    type: OptimizationType
    target: float
    weight: float = 1.0
    constraints: Optional[Dict[str, Any]] = None

class SystemMetrics(BaseModel):
    component: SystemComponent
    metrics: Dict[str, float]
    timestamp: datetime = Field(default_factory=datetime.now)

class OptimizationConfig(BaseModel):
    objectives: List[OptimizationObjective]
    components: List[SystemComponent]
    update_interval: float = 60.0  # seconds
    exploration_rate: float = 0.1

class ResourceType(str, Enum):
    CPU = "cpu"
    MEMORY = "memory"
    GPU = "gpu"
    NETWORK = "network"
    STORAGE = "storage"

class ServiceHealth(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"

class ResourceMetrics(BaseModel):
    cpu_usage: float
    memory_usage: float
    gpu_usage: Optional[float] = None
    network_io: Dict[str, float]
    disk_io: Dict[str, float]
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class ServiceMetrics(BaseModel):
    service_name: str
    health_status: ServiceHealth
    response_time: float
    error_rate: float
    request_rate: float
    resource_metrics: ResourceMetrics

class ResourceAllocation(BaseModel):
    service_name: str
    cpu_limit: int
    memory_limit: int
    gpu_limit: Optional[int] = None
    priority: int = 1

class MetaOptimizer:
    def __init__(self):
        self.metrics_history: Dict[SystemComponent, List[SystemMetrics]] = {
            component: [] for component in SystemComponent
        }
        self.current_configs: Dict[SystemComponent, Dict[str, Any]] = {}
        
    def update_metrics(self, metrics: SystemMetrics):
        """Update metrics history for a component."""
        self.metrics_history[metrics.component].append(metrics)
        
        # Keep only recent history
        max_history = 1000
        if len(self.metrics_history[metrics.component]) > max_history:
            self.metrics_history[metrics.component] = self.metrics_history[metrics.component][-max_history:]
            
    def calculate_objective_score(
        self,
        objective: OptimizationObjective,
        metrics: Dict[str, float]
    ) -> float:
        """Calculate score for a single objective."""
        if objective.type == OptimizationType.RESOURCE:
            usage = metrics.get("resource_usage", 1.0)
            return 1.0 - abs(usage - objective.target)
            
        elif objective.type == OptimizationType.PERFORMANCE:
            latency = metrics.get("latency", float("inf"))
            throughput = metrics.get("throughput", 0.0)
            score = (1.0 / (1.0 + latency)) * throughput
            return score / objective.target
            
        elif objective.type == OptimizationType.RELIABILITY:
            uptime = metrics.get("uptime", 0.0)
            error_rate = metrics.get("error_rate", 1.0)
            return (uptime * (1.0 - error_rate)) / objective.target
            
        elif objective.type == OptimizationType.COST:
            cost = metrics.get("cost", float("inf"))
            return 1.0 - (cost / objective.target)
            
        elif objective.type == OptimizationType.ENERGY:
            energy = metrics.get("energy_consumption", float("inf"))
            return 1.0 - (energy / objective.target)
            
        return 0.0
        
    def optimize_component(
        self,
        component: SystemComponent,
        objectives: List[OptimizationObjective],
        exploration_rate: float
    ) -> Dict[str, Any]:
        """Generate optimized configuration for a component."""
        if not self.metrics_history[component]:
            return self.current_configs.get(component, {})
            
        # Get recent metrics
        recent_metrics = self.metrics_history[component][-1].metrics
        
        # Calculate current performance
        current_score = sum(
            objective.weight * self.calculate_objective_score(objective, recent_metrics)
            for objective in objectives
        )
        
        # Decide whether to explore or exploit
        if np.random.random() < exploration_rate:
            # Exploration: Try new configuration
            config = self._generate_exploration_config(component, objectives)
        else:
            # Exploitation: Optimize current configuration
            config = self._optimize_current_config(
                component,
                objectives,
                recent_metrics
            )
            
        self.current_configs[component] = config
        return config
        
    def _generate_exploration_config(
        self,
        component: SystemComponent,
        objectives: List[OptimizationObjective]
    ) -> Dict[str, Any]:
        """Generate exploration configuration based on objectives."""
        config = {}
        
        if component == SystemComponent.PERCEPTION:
            config.update({
                "batch_size": np.random.randint(16, 128),
                "attention_heads": np.random.randint(4, 12),
                "processing_threads": np.random.randint(2, 8)
            })
            
        elif component == SystemComponent.REASONING:
            config.update({
                "inference_depth": np.random.randint(3, 10),
                "confidence_threshold": np.random.uniform(0.6, 0.9),
                "cache_size": np.random.randint(1000, 10000)
            })
            
        elif component == SystemComponent.MEMORY:
            config.update({
                "cache_policy": np.random.choice(["lru", "lfu", "fifo"]),
                "retention_period": np.random.randint(3600, 86400),
                "compression_level": np.random.randint(1, 9)
            })
            
        elif component == SystemComponent.LEARNING:
            config.update({
                "learning_rate": np.random.uniform(0.0001, 0.01),
                "batch_size": np.random.randint(32, 256),
                "optimizer": np.random.choice(["adam", "sgd", "rmsprop"])
            })
            
        elif component == SystemComponent.COMMUNICATION:
            config.update({
                "buffer_size": np.random.randint(1024, 8192),
                "compression_enabled": np.random.choice([True, False]),
                "retry_limit": np.random.randint(3, 10)
            })
            
        elif component == SystemComponent.INTEGRATION:
            config.update({
                "fusion_method": np.random.choice(["weighted", "attention", "concat"]),
                "timeout": np.random.uniform(1.0, 5.0),
                "batch_enabled": np.random.choice([True, False])
            })
            
        elif component == SystemComponent.MONITORING:
            config.update({
                "sampling_rate": np.random.uniform(0.1, 1.0),
                "aggregation_interval": np.random.randint(10, 60),
                "alert_threshold": np.random.uniform(0.8, 0.95)
            })
            
        return config
        
    def _optimize_current_config(
        self,
        component: SystemComponent,
        objectives: List[OptimizationType],
        metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """Optimize current configuration based on metrics."""
        current_config = self.current_configs.get(component, {})
        if not current_config:
            return self._generate_exploration_config(component, objectives)
            
        # Apply small adjustments to current configuration
        optimized = current_config.copy()
        
        # Adjust based on metrics and objectives
        for objective in objectives:
            if objective.type == OptimizationType.RESOURCE:
                if metrics.get("resource_usage", 0) > objective.target:
                    # Reduce resource usage
                    if "batch_size" in optimized:
                        optimized["batch_size"] = max(16, int(optimized["batch_size"] * 0.9))
                    if "processing_threads" in optimized:
                        optimized["processing_threads"] = max(1, optimized["processing_threads"] - 1)
                        
            elif objective.type == OptimizationType.PERFORMANCE:
                if metrics.get("latency", 0) > 1.0 / objective.target:
                    # Improve performance
                    if "batch_size" in optimized:
                        optimized["batch_size"] = min(256, int(optimized["batch_size"] * 1.1))
                    if "cache_size" in optimized:
                        optimized["cache_size"] = int(optimized["cache_size"] * 1.2)
                        
            elif objective.type == OptimizationType.RELIABILITY:
                if metrics.get("error_rate", 1.0) > 1.0 - objective.target:
                    # Improve reliability
                    if "confidence_threshold" in optimized:
                        optimized["confidence_threshold"] = min(0.95, optimized["confidence_threshold"] + 0.05)
                    if "retry_limit" in optimized:
                        optimized["retry_limit"] = optimized["retry_limit"] + 1
                        
        return optimized

class MetaManager:
    def __init__(self):
        self.optimizer = MetaOptimizer()
        self.configs: Dict[str, OptimizationConfig] = {}
        self.optimization_tasks: Dict[str, asyncio.Task] = {}
        self.optimization_configs: Dict[str, OptimizationConfig] = {}
        self.service_metrics: Dict[str, ServiceMetrics] = {}
        self.resource_allocations: Dict[str, ResourceAllocation] = {}
        
    def register_config(self, config_id: str, config: OptimizationConfig):
        """Register an optimization configuration."""
        self.configs[config_id] = config
        
    async def update_metrics(self, metrics: SystemMetrics) -> Dict[str, Any]:
        """Update system metrics and trigger optimization if needed."""
        self.optimizer.update_metrics(metrics)
        
        # Check if optimization is needed
        optimizations = {}
        for config_id, config in self.configs.items():
            if metrics.component in config.components:
                optimized_config = self.optimizer.optimize_component(
                    metrics.component,
                    config.objectives,
                    config.exploration_rate
                )
                optimizations[config_id] = optimized_config
                
        return optimizations
        
    async def start_optimization_loop(
        self,
        config_id: str,
        background_tasks: BackgroundTasks
    ):
        """Start continuous optimization loop for a configuration."""
        if config_id not in self.configs:
            raise HTTPException(
                status_code=404,
                detail=f"Configuration {config_id} not found"
            )
            
        if config_id in self.optimization_tasks:
            # Stop existing task
            self.optimization_tasks[config_id].cancel()
            
        # Create new optimization task
        task = asyncio.create_task(
            self._optimization_loop(config_id)
        )
        self.optimization_tasks[config_id] = task
        
    async def _optimization_loop(self, config_id: str):
        """Continuous optimization loop for a configuration."""
        config = self.configs[config_id]
        
        while True:
            try:
                # Optimize each component
                for component in config.components:
                    optimized_config = self.optimizer.optimize_component(
                        component,
                        config.objectives,
                        config.exploration_rate
                    )
                    
                    # Here you would typically send the optimized config
                    # to the component through your communication system
                    logger.info(f"Generated optimization for {component}: {optimized_config}")
                    
                # Wait for next update interval
                await asyncio.sleep(config.update_interval)
                
            except asyncio.CancelledError:
                logger.info(f"Optimization loop cancelled for config {config_id}")
                break
            except Exception as e:
                logger.error(f"Error in optimization loop: {str(e)}")
                await asyncio.sleep(5)  # Wait before retrying

    async def collect_system_metrics(self) -> Dict[str, Any]:
        """Collect system-wide metrics."""
        metrics = {
            "cpu": psutil.cpu_percent(interval=1),
            "memory": psutil.virtual_memory().percent,
            "disk": psutil.disk_usage('/').percent,
            "network": {
                "sent": psutil.net_io_counters().bytes_sent,
                "received": psutil.net_io_counters().bytes_recv
            }
        }
        return metrics
    
    async def collect_container_metrics(self) -> Dict[str, Any]:
        """Collect metrics from all containers."""
        container_metrics = {}
        for container in docker_client.containers.list():
            stats = container.stats(stream=False)
            container_metrics[container.name] = {
                "cpu": stats["cpu_stats"]["cpu_usage"]["total_usage"],
                "memory": stats["memory_stats"]["usage"],
                "network": stats["networks"] if "networks" in stats else {}
            }
        return container_metrics
    
    async def optimize_resources(self) -> Dict[str, ResourceAllocation]:
        """Optimize resource allocation based on current metrics and configs."""
        system_metrics = await self.collect_system_metrics()
        container_metrics = await self.collect_container_metrics()
        
        # Implement resource optimization logic
        optimized_allocations = {}
        for service, metrics in self.service_metrics.items():
            if metrics.health_status == ServiceHealth.DEGRADED:
                # Increase resources for degraded services
                current_allocation = self.resource_allocations.get(service)
                if current_allocation:
                    optimized_allocations[service] = ResourceAllocation(
                        service_name=service,
                        cpu_limit=int(current_allocation.cpu_limit * 1.2),
                        memory_limit=int(current_allocation.memory_limit * 1.2),
                        gpu_limit=current_allocation.gpu_limit,
                        priority=current_allocation.priority
                    )
        
        return optimized_allocations
    
    async def analyze_performance(self) -> Dict[str, Any]:
        """Analyze system-wide performance metrics."""
        metrics = await self.collect_system_metrics()
        container_metrics = await self.collect_container_metrics()
        
        analysis = {
            "system_health": ServiceHealth.HEALTHY if metrics["cpu"] < 80 else ServiceHealth.DEGRADED,
            "bottlenecks": [],
            "recommendations": []
        }
        
        # Identify bottlenecks
        if metrics["cpu"] > 80:
            analysis["bottlenecks"].append("High CPU usage")
            analysis["recommendations"].append("Consider scaling CPU resources")
        
        if metrics["memory"] > 80:
            analysis["bottlenecks"].append("High memory usage")
            analysis["recommendations"].append("Consider increasing memory allocation")
            
        return analysis
    
    async def schedule_resources(self) -> None:
        """Schedule and apply resource allocations."""
        optimized_allocations = await self.optimize_resources()
        for service, allocation in optimized_allocations.items():
            try:
                container = docker_client.containers.get(service)
                container.update(
                    cpu_quota=int(allocation.cpu_limit * 100000),
                    mem_limit=f"{allocation.memory_limit}m"
                )
                logger.info(f"Updated resources for {service}")
            except Exception as e:
                logger.error(f"Failed to update resources for {service}: {str(e)}")

# Initialize MetaManager
meta_manager = MetaManager()

# Initialize FastAPI app
@asynccontextmanager
async def lifespan(app: FastAPI):
    # STARTUP
    logger.info("Initializing meta service...")
    try:
        app.state.meta_manager = meta_manager
        logger.info("Meta service initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize meta service: {str(e)}")
        raise
        
    yield
    
    # SHUTDOWN
    logger.info("Shutting down meta service...")
    # Cancel all optimization tasks
    for task in app.state.meta_manager.optimization_tasks.values():
        task.cancel()
    await asyncio.gather(*app.state.meta_manager.optimization_tasks.values(), return_exceptions=True)

app = FastAPI(title="HMAS Meta Service", lifespan=lifespan)

# Prometheus metrics
Instrumentator().instrument(app).expose(app)

@app.post("/config/{config_id}")
@limiter.limit("20/minute")
async def register_config(
    request: Request,
    config_id: str,
    config: OptimizationConfig
):
    """Register an optimization configuration."""
    try:
        request.app.state.meta_manager.register_config(config_id, config)
        return {"status": "success", "config_id": config_id}
    except Exception as e:
        logger.error(f"Error registering configuration: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/metrics")
@limiter.limit("100/minute")
async def update_metrics(
    request: Request,
    metrics: SystemMetrics
):
    """Update system metrics and get optimization suggestions."""
    try:
        optimizations = await request.app.state.meta_manager.update_metrics(metrics)
        return {
            "status": "success",
            "optimizations": optimizations
        }
    except Exception as e:
        logger.error(f"Error updating metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/optimize/{config_id}/start")
@limiter.limit("20/minute")
async def start_optimization(
    request: Request,
    config_id: str,
    background_tasks: BackgroundTasks
):
    """Start continuous optimization for a configuration."""
    try:
        await request.app.state.meta_manager.start_optimization_loop(
            config_id,
            background_tasks
        )
        return {"status": "success", "message": "Optimization loop started"}
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error starting optimization: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/optimization/config")
@limiter.limit("10/minute")
async def set_optimization_config(config: OptimizationConfig, request: Request):
    """Set optimization configuration for the system."""
    REQUESTS.inc()
    with LATENCY.time():
        request.app.state.meta_manager.optimization_configs[config.optimization_type] = config
        return {"status": "success", "message": "Optimization config updated"}

@app.post("/metrics/service")
@limiter.limit("100/minute")
async def update_service_metrics(metrics: ServiceMetrics, request: Request):
    """Update metrics for a specific service."""
    REQUESTS.inc()
    with LATENCY.time():
        request.app.state.meta_manager.service_metrics[metrics.service_name] = metrics
        return {"status": "success", "message": "Service metrics updated"}

@app.get("/analysis/performance")
@limiter.limit("10/minute")
async def get_performance_analysis(request: Request):
    """Get system-wide performance analysis."""
    REQUESTS.inc()
    with LATENCY.time():
        analysis = await request.app.state.meta_manager.analyze_performance()
        return analysis

@app.post("/resources/allocate")
@limiter.limit("10/minute")
async def allocate_resources(allocation: ResourceAllocation, request: Request):
    """Allocate resources to a specific service."""
    REQUESTS.inc()
    with LATENCY.time():
        request.app.state.meta_manager.resource_allocations[allocation.service_name] = allocation
        await request.app.state.meta_manager.schedule_resources()
        return {"status": "success", "message": "Resources allocated"}

@app.get("/metrics/system")
@limiter.limit("60/minute")
async def get_system_metrics(request: Request):
    """Get current system metrics."""
    REQUESTS.inc()
    with LATENCY.time():
        metrics = await request.app.state.meta_manager.collect_system_metrics()
        return metrics

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.utcnow()}

# Background task for continuous optimization
@app.on_event("startup")
async def start_optimization_loop():
    async def optimize_loop():
        while True:
            try:
                await request.app.state.meta_manager.optimize_resources()
                await request.app.state.meta_manager.schedule_resources()
            except Exception as e:
                logger.error(f"Optimization loop error: {str(e)}")
            await asyncio.sleep(60)  # Run every minute
    
    asyncio.create_task(optimize_loop())

# Start Prometheus metrics server
start_http_server(8801)

# Run the service
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8800) 