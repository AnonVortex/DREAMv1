"""
Base service module providing standardized service functionality for HMAS services.
"""

import asyncio
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type
from datetime import datetime
import logging
from pydantic import BaseModel

from shared.config import ServiceConfig

class ServiceStatus(BaseModel):
    """Service status information"""
    name: str
    status: str  # "starting", "running", "stopping", "stopped", "error"
    uptime: float
    last_error: Optional[str] = None
    metrics: Dict[str, Any] = {}
    dependencies: Dict[str, bool] = {}
    version: str
    start_time: datetime

class ServiceDependency(BaseModel):
    """Service dependency configuration"""
    name: str
    required: bool = True
    timeout: float = 30.0
    retry_interval: float = 1.0
    max_retries: int = 3

class BaseService(ABC):
    """
    Base service class providing standardized functionality for all HMAS services.
    
    Features:
    - Lifecycle management (start, stop, restart)
    - Health checks and monitoring
    - Dependency management
    - Configuration management
    - Error handling and recovery
    - Metrics collection
    - Event handling
    """
    
    def __init__(
        self,
        config: ServiceConfig,
        dependencies: Optional[List[ServiceDependency]] = None
    ):
        self.config = config
        self.name = config.service_name
        self.logger = logging.getLogger(self.name)
        self.status = ServiceStatus(
            name=self.name,
            status="stopped",
            uptime=0.0,
            version=config.version,
            start_time=datetime.now()
        )
        self.dependencies = dependencies or []
        self._stop_event = asyncio.Event()
        self._started = False
        self._tasks: List[asyncio.Task] = []
        
    async def start(self) -> None:
        """Start the service and its components"""
        if self._started:
            self.logger.warning(f"Service {self.name} is already running")
            return
            
        try:
            self.logger.info(f"Starting service {self.name}")
            self.status.status = "starting"
            
            # Check dependencies
            await self._check_dependencies()
            
            # Initialize components
            await self._initialize()
            
            # Start background tasks
            await self._start_background_tasks()
            
            # Start service-specific components
            await self._start()
            
            self._started = True
            self.status.status = "running"
            self.status.start_time = datetime.now()
            self.logger.info(f"Service {self.name} started successfully")
            
        except Exception as e:
            self.status.status = "error"
            self.status.last_error = str(e)
            self.logger.error(f"Failed to start service {self.name}: {e}")
            raise
    
    async def stop(self) -> None:
        """Stop the service and cleanup resources"""
        if not self._started:
            self.logger.warning(f"Service {self.name} is not running")
            return
            
        try:
            self.logger.info(f"Stopping service {self.name}")
            self.status.status = "stopping"
            
            # Signal stop to background tasks
            self._stop_event.set()
            
            # Stop service-specific components
            await self._stop()
            
            # Cancel background tasks
            for task in self._tasks:
                task.cancel()
            await asyncio.gather(*self._tasks, return_exceptions=True)
            
            # Cleanup
            await self._cleanup()
            
            self._started = False
            self.status.status = "stopped"
            self.logger.info(f"Service {self.name} stopped successfully")
            
        except Exception as e:
            self.status.status = "error"
            self.status.last_error = str(e)
            self.logger.error(f"Failed to stop service {self.name}: {e}")
            raise
    
    async def restart(self) -> None:
        """Restart the service"""
        await self.stop()
        await self.start()
    
    async def get_status(self) -> ServiceStatus:
        """Get current service status"""
        if self._started:
            self.status.uptime = (datetime.now() - self.status.start_time).total_seconds()
        return self.status
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get service metrics"""
        return await self._collect_metrics()
    
    async def health_check(self) -> bool:
        """Perform health check"""
        try:
            # Check if service is running
            if not self._started:
                return False
                
            # Check dependencies
            for dep in self.dependencies:
                if dep.required and not self.status.dependencies.get(dep.name, False):
                    return False
            
            # Service-specific health check
            return await self._health_check()
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return False
    
    @abstractmethod
    async def _start(self) -> None:
        """Service-specific start logic"""
        pass
    
    @abstractmethod
    async def _stop(self) -> None:
        """Service-specific stop logic"""
        pass
    
    @abstractmethod
    async def _health_check(self) -> bool:
        """Service-specific health check logic"""
        pass
    
    async def _initialize(self) -> None:
        """Initialize service components"""
        # Override in subclass if needed
        pass
    
    async def _cleanup(self) -> None:
        """Cleanup service resources"""
        # Override in subclass if needed
        pass
    
    async def _collect_metrics(self) -> Dict[str, Any]:
        """Collect service-specific metrics"""
        # Override in subclass if needed
        return {}
    
    async def _check_dependencies(self) -> None:
        """Check service dependencies"""
        for dep in self.dependencies:
            success = False
            retries = 0
            
            while not success and retries < dep.max_retries:
                try:
                    # Attempt to connect to dependency
                    success = await self._check_dependency(dep)
                    self.status.dependencies[dep.name] = success
                    
                    if not success and dep.required:
                        await asyncio.sleep(dep.retry_interval)
                        retries += 1
                    else:
                        break
                        
                except Exception as e:
                    self.logger.error(f"Failed to check dependency {dep.name}: {e}")
                    if dep.required:
                        raise
    
    async def _check_dependency(self, dependency: ServiceDependency) -> bool:
        """Check specific dependency - override in subclass"""
        return True
    
    async def _start_background_tasks(self) -> None:
        """Start service background tasks"""
        # Start monitoring task
        self._tasks.append(
            asyncio.create_task(self._monitoring_task())
        )
        
        # Start metrics collection task
        self._tasks.append(
            asyncio.create_task(self._metrics_task())
        )
    
    async def _monitoring_task(self) -> None:
        """Background task for service monitoring"""
        while not self._stop_event.is_set():
            try:
                # Update service status
                is_healthy = await self.health_check()
                if not is_healthy and self.status.status == "running":
                    self.status.status = "error"
                    self.logger.error(f"Service {self.name} health check failed")
                
                # Check dependencies
                for dep in self.dependencies:
                    self.status.dependencies[dep.name] = await self._check_dependency(dep)
                
            except Exception as e:
                self.logger.error(f"Monitoring task error: {e}")
            
            await asyncio.sleep(self.config.monitoring_interval)
    
    async def _metrics_task(self) -> None:
        """Background task for metrics collection"""
        while not self._stop_event.is_set():
            try:
                self.status.metrics = await self._collect_metrics()
            except Exception as e:
                self.logger.error(f"Metrics collection error: {e}")
            
            await asyncio.sleep(self.config.metrics_interval)
    
    def _handle_error(self, error: Exception, context: str = "") -> None:
        """Handle service errors"""
        self.status.last_error = f"{context}: {str(error)}"
        self.logger.error(f"Service error in {context}: {error}")
        
        # Update error metrics
        if "errors" not in self.status.metrics:
            self.status.metrics["errors"] = {}
        error_type = type(error).__name__
        self.status.metrics["errors"][error_type] = self.status.metrics["errors"].get(error_type, 0) + 1 