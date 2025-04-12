from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Tuple, Set
import docker
from docker.errors import DockerException
import psutil
import time
from datetime import datetime, timedelta
import json
import os
import random
import logging
import logging.config
import numpy as np
from sklearn.linear_model import LinearRegression
from dataclasses import dataclass
from enum import Enum
import requests
import copy
from collections import defaultdict

# Configure logging
logging.config.dictConfig({
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
        'detailed': {
            'format': '%(asctime)s [%(levelname)s] %(name)s:%(lineno)d: %(message)s'
        }
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'standard',
            'level': 'INFO'
        },
        'file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': 'adaptation_service.log',
            'maxBytes': 10485760,  # 10MB
            'backupCount': 5,
            'formatter': 'detailed',
            'level': 'DEBUG'
        }
    },
    'loggers': {
        '': {  # Root logger
            'handlers': ['console', 'file'],
            'level': 'DEBUG'
        }
    }
})

logger = logging.getLogger(__name__)

app = FastAPI(title="HMAS Adaptation Service")

class ResourcePrediction(BaseModel):
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_usage: float
    energy_consumption: float
    confidence: float
    timestamp: datetime

class ResourceMetrics(BaseModel):
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_usage: float
    energy_consumption: float
    timestamp: datetime
    gpu_metrics: Optional[Dict[str, float]] = None

class ResourceThresholds(BaseModel):
    cpu_warning: float = 80.0
    cpu_critical: float = 90.0
    memory_warning: float = 75.0
    memory_critical: float = 85.0
    disk_warning: float = 80.0
    disk_critical: float = 90.0
    energy_warning: float = 80.0
    energy_critical: float = 90.0

class CachePolicy(Enum):
    LRU = "lru"
    LFU = "lfu"
    ADAPTIVE = "adaptive"

@dataclass
class ResourceCache:
    capacity: int
    policy: CachePolicy
    data: Dict[str, Any]
    access_count: Dict[str, int]
    last_access: Dict[str, datetime]

# Resource Management
class ResourceManager:
    def __init__(self):
        self.metrics_history = []
        self.prediction_model = None
        self.last_training_time = None
        self.thresholds = ResourceThresholds()
        self.cache = ResourceCache(
            capacity=1000,
            policy=CachePolicy.ADAPTIVE,
            data={},
            access_count={},
            last_access={}
        )
        self.logger = logging.getLogger(f"{__name__}.ResourceManager")
        self.logger.info("Initializing ResourceManager")
        
        # Initialize GPU monitoring if available
        self.gpu_available = self._check_gpu_availability()
    
    def _check_gpu_availability(self) -> bool:
        """Check if GPU monitoring is available."""
        try:
            import py3nvml.py3nvml as nvml
            nvml.nvmlInit()
            self.nvml = nvml
            return True
        except (ImportError, Exception):
            self.logger.info("GPU monitoring not available")
            return False
    
    def collect_metrics(self) -> ResourceMetrics:
        self.logger.debug("Collecting system metrics")
        try:
            # Basic metrics collection
            cpu_usage = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            network = psutil.net_io_counters()
            
            # Energy consumption monitoring
            energy_consumption = self._collect_energy_metrics()
            
            # GPU metrics if available
            gpu_metrics = self._collect_gpu_metrics() if self.gpu_available else None
            
            metrics = ResourceMetrics(
                cpu_usage=cpu_usage,
                memory_usage=memory.percent,
                disk_usage=disk.percent,
                network_usage=(network.bytes_sent + network.bytes_recv) / (1024 * 1024),
                energy_consumption=energy_consumption,
                timestamp=datetime.now(),
                gpu_metrics=gpu_metrics
            )
            
            self._update_cache(metrics)
            self.metrics_history.append(metrics)
            
            # Trim history if too long
            if len(self.metrics_history) > 1000:
                self.metrics_history = self.metrics_history[-1000:]
            
            self._check_thresholds(metrics)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error collecting metrics: {str(e)}", exc_info=True)
            raise
    
    def _collect_energy_metrics(self) -> float:
        """Collect energy consumption metrics."""
        try:
            # Try to get energy info from RAPL (Running Average Power Limit) interface
            energy_consumption = 0.0
            if os.path.exists("/sys/class/powercap/intel-rapl"):
                with open("/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj") as f:
                    energy_consumption = float(f.read()) / 1_000_000  # Convert to Watts
            else:
                # Estimate based on CPU usage and memory
                cpu_power = psutil.cpu_percent() * 2  # Rough estimate: 2W per % CPU
                memory = psutil.virtual_memory()
                memory_power = memory.percent * 0.5  # Rough estimate: 0.5W per % memory
                energy_consumption = cpu_power + memory_power
            
            return energy_consumption
        except Exception as e:
            self.logger.warning(f"Error collecting energy metrics: {str(e)}")
            return 0.0
    
    def _collect_gpu_metrics(self) -> Dict[str, float]:
        """Collect GPU metrics if available."""
        try:
            metrics = {}
            device_count = self.nvml.nvmlDeviceGetCount()
            
            for i in range(device_count):
                handle = self.nvml.nvmlDeviceGetHandleByIndex(i)
                info = self.nvml.nvmlDeviceGetMemoryInfo(handle)
                utilization = self.nvml.nvmlDeviceGetUtilizationRates(handle)
                
                metrics[f"gpu_{i}"] = {
                    "memory_used": info.used / info.total * 100,
                    "utilization": utilization.gpu
                }
            
            return metrics
        except Exception as e:
            self.logger.warning(f"Error collecting GPU metrics: {str(e)}")
            return {}
    
    def _check_thresholds(self, metrics: ResourceMetrics):
        """Check if metrics exceed warning or critical thresholds."""
        warnings = []
        
        if metrics.cpu_usage > self.thresholds.cpu_critical:
            self.logger.critical(f"CPU usage critical: {metrics.cpu_usage}%")
            warnings.append(("cpu", "critical"))
        elif metrics.cpu_usage > self.thresholds.cpu_warning:
            self.logger.warning(f"CPU usage high: {metrics.cpu_usage}%")
            warnings.append(("cpu", "warning"))
        
        if metrics.memory_usage > self.thresholds.memory_critical:
            self.logger.critical(f"Memory usage critical: {metrics.memory_usage}%")
            warnings.append(("memory", "critical"))
        elif metrics.memory_usage > self.thresholds.memory_warning:
            self.logger.warning(f"Memory usage high: {metrics.memory_usage}%")
            warnings.append(("memory", "warning"))
        
        return warnings
    
    def _update_cache(self, metrics: ResourceMetrics):
        """Update the resource metrics cache."""
        cache_key = metrics.timestamp.isoformat()
        
        if len(self.cache.data) >= self.cache.capacity:
            if self.cache.policy == CachePolicy.LRU:
                # Remove least recently used
                oldest_key = min(self.cache.last_access.items(), key=lambda x: x[1])[0]
                self._remove_from_cache(oldest_key)
            elif self.cache.policy == CachePolicy.LFU:
                # Remove least frequently used
                least_used_key = min(self.cache.access_count.items(), key=lambda x: x[1])[0]
                self._remove_from_cache(least_used_key)
            elif self.cache.policy == CachePolicy.ADAPTIVE:
                # Use a combination of frequency and recency
                scores = {}
                now = datetime.now()
                for key in self.cache.data:
                    frequency = self.cache.access_count[key]
                    recency = (now - self.cache.last_access[key]).total_seconds()
                    scores[key] = frequency / (1 + recency)
                worst_key = min(scores.items(), key=lambda x: x[1])[0]
                self._remove_from_cache(worst_key)
        
        self.cache.data[cache_key] = metrics
        self.cache.access_count[cache_key] = 1
        self.cache.last_access[cache_key] = datetime.now()
    
    def _remove_from_cache(self, key: str):
        """Remove an item from all cache dictionaries."""
        self.cache.data.pop(key, None)
        self.cache.access_count.pop(key, None)
        self.cache.last_access.pop(key, None)
    
    def predict_resource_usage(self, horizon_minutes: int = 30) -> List[ResourcePrediction]:
        """Predict resource usage for the next specified minutes."""
        self.logger.debug(f"Predicting resource usage for next {horizon_minutes} minutes")
        
        try:
            if len(self.metrics_history) < 10:
                self.logger.warning("Insufficient data for prediction")
                return []
            
            # Prepare training data
            X = np.array([(m.timestamp - self.metrics_history[0].timestamp).total_seconds() 
                         for m in self.metrics_history]).reshape(-1, 1)
            y_cpu = np.array([m.cpu_usage for m in self.metrics_history])
            y_memory = np.array([m.memory_usage for m in self.metrics_history])
            y_disk = np.array([m.disk_usage for m in self.metrics_history])
            y_network = np.array([m.network_usage for m in self.metrics_history])
            y_energy = np.array([m.energy_consumption for m in self.metrics_history])
            
            # Train models if needed
            if (not self.prediction_model or 
                not self.last_training_time or 
                datetime.now() - self.last_training_time > timedelta(minutes=5)):
                
                self.prediction_model = {
                    'cpu': LinearRegression().fit(X, y_cpu),
                    'memory': LinearRegression().fit(X, y_memory),
                    'disk': LinearRegression().fit(X, y_disk),
                    'network': LinearRegression().fit(X, y_network),
                    'energy': LinearRegression().fit(X, y_energy)
                }
                self.last_training_time = datetime.now()
            
            # Generate predictions
            predictions = []
            last_timestamp = self.metrics_history[-1].timestamp
            for i in range(horizon_minutes):
                future_time = (i + 1) * 60  # seconds
                X_pred = np.array([[X[-1][0] + future_time]])
                
                prediction = ResourcePrediction(
                    cpu_usage=max(0, min(100, float(self.prediction_model['cpu'].predict(X_pred)[0]))),
                    memory_usage=max(0, min(100, float(self.prediction_model['memory'].predict(X_pred)[0]))),
                    disk_usage=max(0, min(100, float(self.prediction_model['disk'].predict(X_pred)[0]))),
                    network_usage=max(0, float(self.prediction_model['network'].predict(X_pred)[0])),
                    energy_consumption=max(0, float(self.prediction_model['energy'].predict(X_pred)[0])),
                    confidence=self._calculate_prediction_confidence(),
                    timestamp=last_timestamp + timedelta(minutes=i+1)
                )
                predictions.append(prediction)
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"Error predicting resource usage: {str(e)}", exc_info=True)
            return []
    
    def _calculate_prediction_confidence(self) -> float:
        """Calculate confidence score for predictions."""
        if not self.prediction_model:
            return 0.0
        
        try:
            # Use R² scores as basis for confidence
            confidence_scores = [
                max(0, model.score(
                    np.array([(m.timestamp - self.metrics_history[0].timestamp).total_seconds() 
                             for m in self.metrics_history]).reshape(-1, 1),
                    np.array([getattr(m, f"{metric}_usage") for m in self.metrics_history])
                ))
                for metric, model in self.prediction_model.items()
            ]
            
            # Average confidence score
            return sum(confidence_scores) / len(confidence_scores)
        except Exception as e:
            self.logger.warning(f"Error calculating prediction confidence: {str(e)}")
            return 0.0

# Dynamic Scaling
class ScalingPolicy(BaseModel):
    min_instances: int = 1
    max_instances: int = 10
    cpu_threshold_up: float = 80.0
    cpu_threshold_down: float = 20.0
    memory_threshold_up: float = 75.0
    memory_threshold_down: float = 25.0
    cooldown_period: int = 300  # seconds
    scale_up_factor: float = 2.0
    scale_down_factor: float = 0.5
    enable_gpu_scaling: bool = False

class ScalingHistory(BaseModel):
    timestamp: datetime
    action: str
    agent_id: str
    old_config: Dict[str, Any]
    new_config: Dict[str, Any]
    reason: str
    success: bool

class AgentConfig(BaseModel):
    agent_id: str
    agent_type: str
    resources: Dict[str, float]
    dependencies: List[str]
    priority: int
    gpu_config: Optional[Dict[str, Any]] = None
    scaling_policy: Optional[ScalingPolicy] = None

class DynamicScaler:
    def __init__(self):
        self.docker_client = docker.from_env()
        self.active_agents = {}
        self.scaling_history = []
        self.logger = logging.getLogger(f"{__name__}.DynamicScaler")
        self.logger.info("Initializing DynamicScaler")
        
        # Initialize GPU support if available
        self.gpu_available = self._check_gpu_availability()
    
    def _check_gpu_availability(self) -> bool:
        """Check if GPU support is available."""
        try:
            import py3nvml.py3nvml as nvml
            nvml.nvmlInit()
            self.nvml = nvml
            return True
        except (ImportError, Exception):
            self.logger.info("GPU support not available")
            return False
    
    def scale_agent(self, agent_id: str, scale_factor: float, reason: str = "manual") -> bool:
        """Scale an agent's resources by the given factor."""
        self.logger.info(f"Scaling agent {agent_id} with factor {scale_factor}")
        
        try:
            # Check cooldown period
            if not self._check_cooldown(agent_id):
                self.logger.warning(f"Agent {agent_id} in cooldown period, skipping scaling")
                return False
            
            container = self.docker_client.containers.get(agent_id)
            current_config = container.attrs['HostConfig']
            
            # Store old configuration for history
            old_config = {
                'cpu_quota': current_config.get('NanoCpus', 1000000000),
                'memory': current_config.get('Memory', 1024*1024*1024),
                'gpu_config': current_config.get('DeviceRequests', [])
            }
            
            # Calculate new resource limits
            new_cpu = int(float(old_config['cpu_quota']) * scale_factor)
            new_memory = int(float(old_config['memory']) * scale_factor)
            
            # Update container resources
            update_kwargs = {
                'cpu_quota': new_cpu,
                'memory': new_memory
            }
            
            # Handle GPU scaling if available and enabled
            if self.gpu_available and self._should_scale_gpu(agent_id):
                gpu_config = self._calculate_gpu_config(agent_id, scale_factor)
                if gpu_config:
                    update_kwargs['device_requests'] = gpu_config
            
            # Apply the update
            container.update(**update_kwargs)
            
            # Record the scaling action
            new_config = {
                'cpu_quota': new_cpu,
                'memory': new_memory,
                'gpu_config': update_kwargs.get('device_requests', [])
            }
            
            self._record_scaling_history(
                agent_id=agent_id,
                action="scale",
                old_config=old_config,
                new_config=new_config,
                reason=reason,
                success=True
            )
            
            # Update active agents tracking
            self.active_agents[agent_id] = {
                'scale_factor': scale_factor,
                'last_scaled': datetime.now()
            }
            
            self.logger.info(f"Successfully scaled agent {agent_id}")
            return True
            
        except DockerException as e:
            self.logger.error(f"Docker error scaling agent {agent_id}: {str(e)}", exc_info=True)
            self._record_scaling_history(
                agent_id=agent_id,
                action="scale",
                old_config={},
                new_config={},
                reason=reason,
                success=False
            )
            raise HTTPException(status_code=500, detail=str(e))
        except Exception as e:
            self.logger.error(f"Unexpected error scaling agent {agent_id}: {str(e)}", exc_info=True)
            self._record_scaling_history(
                agent_id=agent_id,
                action="scale",
                old_config={},
                new_config={},
                reason=reason,
                success=False
            )
            raise
    
    def create_agent(self, config: AgentConfig) -> str:
        """Create a new agent with the specified configuration."""
        self.logger.info(f"Creating new agent with ID {config.agent_id}")
        
        try:
            self.logger.debug(f"Agent configuration: {config.dict()}")
            
            # Prepare container configuration
            container_config = {
                'image': f"hmas/{config.agent_type}:latest",
                'name': config.agent_id,
                'environment': {
                    "AGENT_ID": config.agent_id,
                    "AGENT_TYPE": config.agent_type
                },
                'resources': {
                    'cpu_shares': int(config.resources.get('cpu', 1024)),
                    'mem_limit': f"{config.resources.get('memory', 1024)}m"
                },
                'detach': True
            }
            
            # Add GPU configuration if available and requested
            if self.gpu_available and config.gpu_config:
                container_config['device_requests'] = [
                    {
                        'Driver': 'nvidia',
                        'Count': config.gpu_config.get('count', -1),  # -1 for all available
                        'Capabilities': [['gpu'], ['compute'], ['utility']]
                    }
                ]
            
            # Create the container
            container = self.docker_client.containers.run(**container_config)
            
            # Record creation in history
            self._record_scaling_history(
                agent_id=config.agent_id,
                action="create",
                old_config={},
                new_config=container_config,
                reason="initial_creation",
                success=True
            )
            
            # Initialize scaling history
            self.active_agents[config.agent_id] = {
                'scale_factor': 1.0,
                'last_scaled': datetime.now(),
                'policy': config.scaling_policy or ScalingPolicy()
            }
            
            return container.id
            
        except Exception as e:
            self.logger.error(f"Error creating agent: {str(e)}", exc_info=True)
            self._record_scaling_history(
                agent_id=config.agent_id,
                action="create",
                old_config={},
                new_config={},
                reason="initial_creation",
                success=False
            )
            raise
    
    def _check_cooldown(self, agent_id: str) -> bool:
        """Check if the agent is in cooldown period."""
        if agent_id not in self.active_agents:
            return True
        
        last_scaled = self.active_agents[agent_id]['last_scaled']
        cooldown_period = self.active_agents[agent_id].get('policy', ScalingPolicy()).cooldown_period
        
        return (datetime.now() - last_scaled).total_seconds() > cooldown_period
    
    def _should_scale_gpu(self, agent_id: str) -> bool:
        """Determine if GPU scaling should be performed."""
        if not self.gpu_available:
            return False
        
        return self.active_agents.get(agent_id, {}).get('policy', ScalingPolicy()).enable_gpu_scaling
    
    def _calculate_gpu_config(self, agent_id: str, scale_factor: float) -> Optional[List[Dict[str, Any]]]:
        """Calculate new GPU configuration based on scaling factor."""
        try:
            if not self.gpu_available:
                return None
            
            container = self.docker_client.containers.get(agent_id)
            current_config = container.attrs['HostConfig'].get('DeviceRequests', [])
            
            # Find NVIDIA configuration
            nvidia_config = next((cfg for cfg in current_config if cfg.get('Driver') == 'nvidia'), None)
            
            if nvidia_config:
                # Adjust GPU memory limit if supported
                if 'Options' in nvidia_config:
                    current_memory = int(nvidia_config['Options'].get('gpu-memory-limit', 0))
                    new_memory = int(current_memory * scale_factor)
                    nvidia_config['Options']['gpu-memory-limit'] = str(new_memory)
                
                return [nvidia_config]
            
            return None
            
        except Exception as e:
            self.logger.warning(f"Error calculating GPU config: {str(e)}")
            return None
    
    def _record_scaling_history(self, agent_id: str, action: str, old_config: Dict[str, Any],
                              new_config: Dict[str, Any], reason: str, success: bool):
        """Record a scaling action in the history."""
        history_entry = ScalingHistory(
            timestamp=datetime.now(),
            action=action,
            agent_id=agent_id,
            old_config=old_config,
            new_config=new_config,
            reason=reason,
            success=success
        )
        
        self.scaling_history.append(history_entry)
        
        # Trim history if too long
        if len(self.scaling_history) > 1000:
            self.scaling_history = self.scaling_history[-1000:]
    
    def get_scaling_history(self, agent_id: Optional[str] = None, 
                          start_time: Optional[datetime] = None,
                          end_time: Optional[datetime] = None) -> List[ScalingHistory]:
        """Get scaling history with optional filters."""
        filtered_history = self.scaling_history
        
        if agent_id:
            filtered_history = [h for h in filtered_history if h.agent_id == agent_id]
        
        if start_time:
            filtered_history = [h for h in filtered_history if h.timestamp >= start_time]
        
        if end_time:
            filtered_history = [h for h in filtered_history if h.timestamp <= end_time]
        
        return filtered_history
    
    def get_agent_metrics(self, agent_id: str) -> Dict[str, Any]:
        """Get current metrics for an agent."""
        try:
            container = self.docker_client.containers.get(agent_id)
            stats = container.stats(stream=False)
            
            metrics = {
                'cpu_usage': self._calculate_cpu_percent(stats),
                'memory_usage': self._calculate_memory_percent(stats),
                'network_io': self._calculate_network_io(stats)
            }
            
            if self.gpu_available and self._should_scale_gpu(agent_id):
                metrics['gpu_metrics'] = self._get_gpu_metrics(container)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error getting agent metrics: {str(e)}")
            return {}
    
    def _calculate_cpu_percent(self, stats: Dict[str, Any]) -> float:
        """Calculate CPU usage percentage from stats."""
        cpu_delta = stats['cpu_stats']['cpu_usage']['total_usage'] - \
                   stats['precpu_stats']['cpu_usage']['total_usage']
        system_delta = stats['cpu_stats']['system_cpu_usage'] - \
                      stats['precpu_stats']['system_cpu_usage']
        
        if system_delta > 0.0:
            return (cpu_delta / system_delta) * 100.0
        return 0.0
    
    def _calculate_memory_percent(self, stats: Dict[str, Any]) -> float:
        """Calculate memory usage percentage from stats."""
        usage = stats['memory_stats']['usage']
        limit = stats['memory_stats']['limit']
        
        if limit > 0:
            return (usage / limit) * 100.0
        return 0.0
    
    def _calculate_network_io(self, stats: Dict[str, Any]) -> Dict[str, int]:
        """Calculate network I/O from stats."""
        networks = stats.get('networks', {})
        rx_bytes = sum(net.get('rx_bytes', 0) for net in networks.values())
        tx_bytes = sum(net.get('tx_bytes', 0) for net in networks.values())
        
        return {
            'rx_bytes': rx_bytes,
            'tx_bytes': tx_bytes
        }
    
    def _get_gpu_metrics(self, container: Any) -> Dict[str, Any]:
        """Get GPU metrics for a container."""
        try:
            if not self.gpu_available:
                return {}
            
            # Get container's GPU devices
            device_requests = container.attrs['HostConfig'].get('DeviceRequests', [])
            nvidia_request = next((req for req in device_requests if req.get('Driver') == 'nvidia'), None)
            
            if not nvidia_request:
                return {}
            
            metrics = {}
            device_count = self.nvml.nvmlDeviceGetCount()
            
            for i in range(device_count):
                handle = self.nvml.nvmlDeviceGetHandleByIndex(i)
                info = self.nvml.nvmlDeviceGetMemoryInfo(handle)
                utilization = self.nvml.nvmlDeviceGetUtilizationRates(handle)
                
                metrics[f'gpu_{i}'] = {
                    'memory_used': info.used,
                    'memory_total': info.total,
                    'utilization': utilization.gpu
                }
            
            return metrics
            
        except Exception as e:
            self.logger.warning(f"Error getting GPU metrics: {str(e)}")
            return {}

# Load Balancing
class LoadBalancingStrategy(str, Enum):
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    RESOURCE_BASED = "resource_based"
    WEIGHTED = "weighted"
    LATENCY_BASED = "latency_based"

class ServiceMeshConfig(BaseModel):
    enabled: bool = False
    mesh_provider: str = "istio"  # or "linkerd", "consul"
    sidecar_resources: Dict[str, str] = {}
    timeout_ms: int = 1000
    retry_count: int = 3
    circuit_breaker: Dict[str, Any] = {
        "max_connections": 100,
        "max_pending_requests": 100,
        "max_requests": 100,
        "max_retries": 3
    }

class LoadBalancerMetrics(BaseModel):
    total_requests: int = 0
    active_connections: int = 0
    request_latencies: List[float] = []
    error_count: int = 0
    last_reset: datetime = datetime.now()

class LoadBalancerConfig(BaseModel):
    strategy: LoadBalancingStrategy = LoadBalancingStrategy.RESOURCE_BASED
    service_mesh: ServiceMeshConfig = ServiceMeshConfig()
    sticky_sessions: bool = False
    session_timeout: int = 3600  # seconds
    health_check_interval: int = 10  # seconds
    health_check_timeout: int = 5  # seconds
    max_retries: int = 3

class LoadBalancer:
    def __init__(self, config: Optional[LoadBalancerConfig] = None):
        self.config = config or LoadBalancerConfig()
        self.agents = {}  # Dict[str, AgentState]
        self.session_store = {}  # Dict[str, str] for sticky sessions
        self.metrics = LoadBalancerMetrics()
        self.logger = logging.getLogger(f"{__name__}.LoadBalancer")
        self.logger.info("Initializing LoadBalancer")
        
        # Initialize service mesh if enabled
        if self.config.service_mesh.enabled:
            self._setup_service_mesh()
    
    def _setup_service_mesh(self):
        """Set up service mesh integration."""
        try:
            self.logger.info(f"Setting up service mesh integration with {self.config.service_mesh.mesh_provider}")
            
            if self.config.service_mesh.mesh_provider == "istio":
                # Initialize Istio client
                from istio_api import networking_v1alpha3_pb2 as istio_networking
                from istio_api import security_v1beta1_pb2 as istio_security
                self.mesh_client = self._create_istio_client()
            elif self.config.service_mesh.mesh_provider == "linkerd":
                # Initialize Linkerd client
                import linkerd2_api
                self.mesh_client = self._create_linkerd_client()
            elif self.config.service_mesh.mesh_provider == "consul":
                # Initialize Consul client
                import consul
                self.mesh_client = consul.Consul()
            
            self.logger.info("Service mesh integration completed")
            
        except Exception as e:
            self.logger.error(f"Failed to set up service mesh: {str(e)}")
            self.config.service_mesh.enabled = False
    
    def distribute_load(self, request_id: str, task: Dict[str, Any]) -> str:
        """Distribute load among available agents."""
        self.logger.debug(f"Distributing load for request {request_id}")
        self.metrics.total_requests += 1
        
        try:
            # Check sticky session
            if self.config.sticky_sessions and request_id in self.session_store:
                agent_id = self.session_store[request_id]
                if self._is_agent_healthy(agent_id):
                    self.logger.debug(f"Using sticky session for request {request_id} -> agent {agent_id}")
                    return agent_id
            
            # Get available agents
            available_agents = self._get_available_agents()
            if not available_agents:
                raise HTTPException(status_code=503, detail="No agents available")
            
            # Select agent based on strategy
            selected_agent = self._select_agent(available_agents, task)
            
            # Update sticky session if enabled
            if self.config.sticky_sessions:
                self.session_store[request_id] = selected_agent
                self._cleanup_old_sessions()
            
            # Update metrics
            self.agents[selected_agent]['active_connections'] += 1
            
            return selected_agent
            
        except Exception as e:
            self.logger.error(f"Error distributing load: {str(e)}")
            self.metrics.error_count += 1
            raise
    
    def _select_agent(self, available_agents: List[str], task: Dict[str, Any]) -> str:
        """Select an agent based on the configured strategy."""
        if self.config.strategy == LoadBalancingStrategy.ROUND_ROBIN:
            return self._round_robin_select(available_agents)
        elif self.config.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            return self._least_connections_select(available_agents)
        elif self.config.strategy == LoadBalancingStrategy.RESOURCE_BASED:
            return self._resource_based_select(available_agents, task)
        elif self.config.strategy == LoadBalancingStrategy.WEIGHTED:
            return self._weighted_select(available_agents)
        elif self.config.strategy == LoadBalancingStrategy.LATENCY_BASED:
            return self._latency_based_select(available_agents)
        else:
            return self._resource_based_select(available_agents, task)  # default
    
    def _round_robin_select(self, available_agents: List[str]) -> str:
        """Simple round-robin selection."""
        if not hasattr(self, '_round_robin_index'):
            self._round_robin_index = 0
        
        selected = available_agents[self._round_robin_index % len(available_agents)]
        self._round_robin_index += 1
        
        return selected
    
    def _least_connections_select(self, available_agents: List[str]) -> str:
        """Select agent with least active connections."""
        return min(
            available_agents,
            key=lambda x: self.agents[x]['active_connections']
        )
    
    def _resource_based_select(self, available_agents: List[str], task: Dict[str, Any]) -> str:
        """Select agent based on resource availability and task requirements."""
        best_agent = None
        best_score = float('-inf')
        
        for agent_id in available_agents:
            agent_state = self.agents[agent_id]
            
            # Calculate resource availability score
            cpu_avail = 100 - agent_state['metrics'].get('cpu_usage', 0)
            mem_avail = 100 - agent_state['metrics'].get('memory_usage', 0)
            
            # Calculate task compatibility score
            compatibility = self.calculate_task_compatibility(task, agent_state)
            
            # Combined score with weights
            score = (
                0.4 * cpu_avail +
                0.3 * mem_avail +
                0.3 * compatibility
            )
            
            if score > best_score:
                best_score = score
                best_agent = agent_id
        
        return best_agent
    
    def _weighted_select(self, available_agents: List[str]) -> str:
        """Select agent based on weights."""
        weights = [self.agents[agent_id]['weight'] for agent_id in available_agents]
        return random.choices(available_agents, weights=weights, k=1)[0]
    
    def _latency_based_select(self, available_agents: List[str]) -> str:
        """Select agent based on recent latency performance."""
        return min(
            available_agents,
            key=lambda x: self.agents[x].get('avg_latency', float('inf'))
        )
    
    def calculate_task_compatibility(self, task: Dict[str, Any], agent_state: Dict[str, Any]) -> float:
        """Calculate how well an agent can handle a specific task."""
        try:
            # Check required capabilities
            required_capabilities = set(task.get('requirements', []))
            agent_capabilities = set(agent_state.get('capabilities', []))
            
            if required_capabilities and not required_capabilities.issubset(agent_capabilities):
                return 0.0
            
            # Check resource requirements
            task_resources = task.get('resources', {})
            agent_resources = agent_state.get('metrics', {})
            
            resource_scores = []
            
            # CPU compatibility
            if 'cpu_required' in task_resources:
                cpu_avail = 100 - agent_resources.get('cpu_usage', 0)
                cpu_score = cpu_avail / task_resources['cpu_required'] if task_resources['cpu_required'] > 0 else 1.0
                resource_scores.append(min(cpu_score, 1.0))
            
            # Memory compatibility
            if 'memory_required' in task_resources:
                mem_avail = 100 - agent_resources.get('memory_usage', 0)
                mem_score = mem_avail / task_resources['memory_required'] if task_resources['memory_required'] > 0 else 1.0
                resource_scores.append(min(mem_score, 1.0))
            
            # GPU compatibility if needed
            if 'gpu_required' in task_resources and task_resources['gpu_required']:
                gpu_metrics = agent_resources.get('gpu_metrics', {})
                if not gpu_metrics:
                    return 0.0
                
                gpu_scores = []
                for gpu_id, gpu_stats in gpu_metrics.items():
                    gpu_avail = 100 - gpu_stats.get('utilization', 100)
                    gpu_scores.append(gpu_avail / 100)
                
                if gpu_scores:
                    resource_scores.append(max(gpu_scores))
            
            # Calculate final compatibility score
            if not resource_scores:
                return 1.0 if required_capabilities.issubset(agent_capabilities) else 0.0
            
            return sum(resource_scores) / len(resource_scores)
            
        except Exception as e:
            self.logger.error(f"Error calculating task compatibility: {str(e)}")
            return 0.0
    
    def _is_agent_healthy(self, agent_id: str) -> bool:
        """Check if an agent is healthy."""
        if agent_id not in self.agents:
            return False
        
        agent_state = self.agents[agent_id]
        last_health_check = agent_state.get('last_health_check', datetime.min)
        
        # Check if health check is due
        if (datetime.now() - last_health_check).total_seconds() > self.config.health_check_interval:
            healthy = self._perform_health_check(agent_id)
            self.agents[agent_id]['healthy'] = healthy
            self.agents[agent_id]['last_health_check'] = datetime.now()
            return healthy
        
        return agent_state.get('healthy', False)
    
    def _perform_health_check(self, agent_id: str) -> bool:
        """Perform health check on an agent."""
        try:
            # Basic connectivity check
            container = self.docker_client.containers.get(agent_id)
            if container.status != 'running':
                return False
            
            # Check if service mesh is enabled
            if self.config.service_mesh.enabled:
                return self._check_service_mesh_health(agent_id)
            
            # Custom health check endpoint
            health_check_url = f"http://{container.attrs['NetworkSettings']['IPAddress']}:8000/health"
            response = requests.get(health_check_url, timeout=self.config.health_check_timeout)
            
            return response.status_code == 200
            
        except Exception as e:
            self.logger.warning(f"Health check failed for agent {agent_id}: {str(e)}")
            return False
    
    def _check_service_mesh_health(self, agent_id: str) -> bool:
        """Check agent health through service mesh."""
        try:
            if self.config.service_mesh.mesh_provider == "istio":
                return self._check_istio_health(agent_id)
            elif self.config.service_mesh.mesh_provider == "linkerd":
                return self._check_linkerd_health(agent_id)
            elif self.config.service_mesh.mesh_provider == "consul":
                return self._check_consul_health(agent_id)
            return False
        except Exception as e:
            self.logger.error(f"Service mesh health check failed: {str(e)}")
            return False
    
    def _cleanup_old_sessions(self):
        """Clean up expired sticky sessions."""
        current_time = datetime.now()
        expired_sessions = [
            session_id for session_id, timestamp in self.session_store.items()
            if (current_time - timestamp).total_seconds() > self.config.session_timeout
        ]
        
        for session_id in expired_sessions:
            self.session_store.pop(session_id, None)
    
    def _get_available_agents(self) -> List[str]:
        """Get list of available and healthy agents."""
        return [
            agent_id for agent_id in self.agents
            if self._is_agent_healthy(agent_id)
        ]
    
    def update_agent_metrics(self, agent_id: str, metrics: Dict[str, Any]):
        """Update metrics for an agent."""
        if agent_id not in self.agents:
            self.agents[agent_id] = {
                'metrics': {},
                'active_connections': 0,
                'weight': 1.0,
                'capabilities': [],
                'healthy': True,
                'last_health_check': datetime.min
            }
        
        self.agents[agent_id]['metrics'].update(metrics)
    
    def get_metrics(self) -> LoadBalancerMetrics:
        """Get current load balancer metrics."""
        return self.metrics
    
    def reset_metrics(self):
        """Reset load balancer metrics."""
        self.metrics = LoadBalancerMetrics()
    
    def register_agent(self, agent_id: str, capabilities: List[str], weight: float = 1.0):
        """Register a new agent with the load balancer."""
        self.agents[agent_id] = {
            'metrics': {},
            'active_connections': 0,
            'weight': weight,
            'capabilities': capabilities,
            'healthy': True,
            'last_health_check': datetime.min
        }
    
    def deregister_agent(self, agent_id: str):
        """Remove an agent from the load balancer."""
        self.agents.pop(agent_id, None)
        # Clean up any sticky sessions pointing to this agent
        self.session_store = {
            k: v for k, v in self.session_store.items()
            if v != agent_id
        }

# Self-Modification
class ModificationConstraints(BaseModel):
    max_components: int = 100
    min_components: int = 1
    max_connections_per_component: int = 10
    max_total_connections: int = 1000
    max_resource_increase: float = 2.0
    min_resource_decrease: float = 0.5
    max_modification_rate: int = 10  # per hour
    required_redundancy: int = 2
    max_critical_path_length: int = 5

class ModificationMetrics(BaseModel):
    total_modifications: int = 0
    successful_modifications: int = 0
    failed_modifications: int = 0
    last_modification: Optional[datetime] = None
    modification_history: List[Dict[str, Any]] = []
    current_architecture_version: int = 1

class ModificationRule(BaseModel):
    condition: str
    action: str
    target_component: str
    parameters: Dict[str, Any]
    priority: int = 1
    safety_checks: List[str] = []
    rollback_strategy: Optional[str] = None

class ArchitectureState(BaseModel):
    components: List[str]
    connections: List[Tuple[str, str]]
    resources: Dict[str, Dict[str, float]]
    version: int
    last_modified: datetime
    health_status: Dict[str, str]
    performance_metrics: Dict[str, float]

class SelfModifier:
    def __init__(self, constraints: Optional[ModificationConstraints] = None):
        self.constraints = constraints or ModificationConstraints()
        self.metrics = ModificationMetrics()
        self.current_state = None
        self.modification_lock = False
        self.logger = logging.getLogger(f"{__name__}.SelfModifier")
        self.logger.info("Initializing SelfModifier")
    
    def modify_architecture(self, current_state: ArchitectureState, rule: ModificationRule) -> ArchitectureState:
        """Modify system architecture based on adaptation rule."""
        self.logger.info(f"Attempting architecture modification with rule: {rule.action}")
        
        try:
            # Check modification rate limit
            if not self._check_modification_rate():
                raise ValueError("Modification rate limit exceeded")
            
            # Acquire modification lock
            if not self._acquire_lock():
                raise ValueError("Another modification is in progress")
            
            try:
                # Validate current state
                self._validate_state(current_state)
                
                # Create working copy
                new_state = self._create_state_copy(current_state)
                
                # Perform safety checks
                self._perform_safety_checks(new_state, rule)
                
                # Apply modification
                new_state = self._apply_modification(new_state, rule)
                
                # Validate modified state
                self._validate_modified_state(new_state)
                
                # Update metrics
                self._update_metrics(True, rule)
                
                # Update current state
                self.current_state = new_state
                
                return new_state
                
            finally:
                self._release_lock()
                
        except Exception as e:
            self.logger.error(f"Error modifying architecture: {str(e)}", exc_info=True)
            self._update_metrics(False, rule)
            raise
    
    def _check_modification_rate(self) -> bool:
        """Check if modification rate is within limits."""
        if not self.metrics.last_modification:
            return True
        
        recent_modifications = len([
            m for m in self.metrics.modification_history
            if (datetime.now() - m['timestamp']).total_seconds() < 3600
        ])
        
        return recent_modifications < self.constraints.max_modification_rate
    
    def _acquire_lock(self) -> bool:
        """Acquire modification lock."""
        if self.modification_lock:
            return False
        self.modification_lock = True
        return True
    
    def _release_lock(self):
        """Release modification lock."""
        self.modification_lock = False
    
    def _validate_state(self, state: ArchitectureState):
        """Validate architecture state."""
        if not state.components:
            raise ValueError("Architecture must have at least one component")
        
        if len(state.components) > self.constraints.max_components:
            raise ValueError(f"Too many components: {len(state.components)}")
        
        # Check connection limits
        connection_counts = {}
        for src, dst in state.connections:
            connection_counts[src] = connection_counts.get(src, 0) + 1
            if connection_counts[src] > self.constraints.max_connections_per_component:
                raise ValueError(f"Too many connections for component {src}")
        
        if len(state.connections) > self.constraints.max_total_connections:
            raise ValueError("Too many total connections")
        
        # Check critical path length
        if self._get_critical_path_length(state) > self.constraints.max_critical_path_length:
            raise ValueError("Critical path too long")
    
    def _create_state_copy(self, state: ArchitectureState) -> ArchitectureState:
        """Create a deep copy of architecture state."""
        return ArchitectureState(
            components=state.components.copy(),
            connections=state.connections.copy(),
            resources=copy.deepcopy(state.resources),
            version=state.version + 1,
            last_modified=datetime.now(),
            health_status=state.health_status.copy(),
            performance_metrics=state.performance_metrics.copy()
        )
    
    def _perform_safety_checks(self, state: ArchitectureState, rule: ModificationRule):
        """Perform safety checks before modification."""
        # Check redundancy requirements
        critical_components = self._identify_critical_components(state)
        for component in critical_components:
            if self._count_redundant_components(state, component) < self.constraints.required_redundancy:
                raise ValueError(f"Insufficient redundancy for critical component {component}")
        
        # Check resource constraints
        if rule.action == "scale_up":
            current_resources = state.resources[rule.target_component]
            for resource, value in current_resources.items():
                if value * rule.parameters.get('scale_factor', 1.0) > value * self.constraints.max_resource_increase:
                    raise ValueError(f"Resource increase exceeds maximum for {resource}")
        
        # Check custom safety rules
        for check in rule.safety_checks:
            if not self._evaluate_safety_check(state, check):
                raise ValueError(f"Safety check failed: {check}")
    
    def _apply_modification(self, state: ArchitectureState, rule: ModificationRule) -> ArchitectureState:
        """Apply modification rule to architecture state."""
        if rule.action == "add_component":
            return self._add_component(state, rule)
        elif rule.action == "remove_component":
            return self._remove_component(state, rule)
        elif rule.action == "add_connection":
            return self._add_connection(state, rule)
        elif rule.action == "remove_connection":
            return self._remove_connection(state, rule)
        elif rule.action == "scale_resources":
            return self._scale_resources(state, rule)
        elif rule.action == "modify_component":
            return self._modify_component(state, rule)
        else:
            raise ValueError(f"Unknown modification action: {rule.action}")
    
    def _validate_modified_state(self, state: ArchitectureState):
        """Validate state after modification."""
        self._validate_state(state)
        
        # Additional checks for modified state
        if not self._verify_connectivity(state):
            raise ValueError("Modified architecture has disconnected components")
        
        if not self._verify_performance(state):
            raise ValueError("Modified architecture fails performance requirements")
    
    def _update_metrics(self, success: bool, rule: ModificationRule):
        """Update modification metrics."""
        self.metrics.total_modifications += 1
        if success:
            self.metrics.successful_modifications += 1
        else:
            self.metrics.failed_modifications += 1
        
        self.metrics.last_modification = datetime.now()
        self.metrics.modification_history.append({
            'timestamp': datetime.now(),
            'rule': rule.dict(),
            'success': success,
            'version': self.metrics.current_architecture_version
        })
        
        # Trim history if too long
        if len(self.metrics.modification_history) > 1000:
            self.metrics.modification_history = self.metrics.modification_history[-1000:]
        
        if success:
            self.metrics.current_architecture_version += 1
    
    def _get_critical_path_length(self, state: ArchitectureState) -> int:
        """Calculate the longest path in the architecture."""
        graph = defaultdict(list)
        for src, dst in state.connections:
            graph[src].append(dst)
        
        def dfs(node: str, visited: Set[str]) -> int:
            if node in visited:
                return 0
            visited.add(node)
            if node not in graph:
                return 1
            return 1 + max((dfs(next_node, visited.copy()) for next_node in graph[node]), default=0)
        
        return max((dfs(component, set()) for component in state.components), default=0)
    
    def _identify_critical_components(self, state: ArchitectureState) -> List[str]:
        """Identify critical components in the architecture."""
        critical = []
        for component in state.components:
            # Check if component is in critical path
            if self._is_in_critical_path(state, component):
                critical.append(component)
            # Check if component handles critical functionality
            elif self._has_critical_functionality(state, component):
                critical.append(component)
        return critical
    
    def _count_redundant_components(self, state: ArchitectureState, component: str) -> int:
        """Count number of redundant components for a given component."""
        component_type = self._get_component_type(state, component)
        return sum(1 for c in state.components 
                  if c != component and self._get_component_type(state, c) == component_type)
    
    def _verify_connectivity(self, state: ArchitectureState) -> bool:
        """Verify that all components are connected."""
        graph = defaultdict(list)
        for src, dst in state.connections:
            graph[src].append(dst)
            graph[dst].append(src)
        
        visited = set()
        def dfs(node: str):
            visited.add(node)
            for next_node in graph[node]:
                if next_node not in visited:
                    dfs(next_node)
        
        if not state.components:
            return True
        
        dfs(state.components[0])
        return len(visited) == len(state.components)
    
    def _verify_performance(self, state: ArchitectureState) -> bool:
        """Verify performance metrics of modified architecture."""
        # Check if critical metrics are within acceptable ranges
        for metric, value in state.performance_metrics.items():
            threshold = self._get_performance_threshold(metric)
            if value < threshold:
                return False
        return True
    
    def get_metrics(self) -> ModificationMetrics:
        """Get current modification metrics."""
        return self.metrics
    
    def get_current_state(self) -> Optional[ArchitectureState]:
        """Get current architecture state."""
        return self.current_state
    
    def reset_metrics(self):
        """Reset modification metrics."""
        self.metrics = ModificationMetrics()

# API endpoints
@app.post("/resources/metrics")
async def get_resource_metrics():
    logger.info("API: Resource metrics request received")
    try:
        resource_manager = ResourceManager()
        metrics = resource_manager.collect_metrics()
        logger.info("API: Resource metrics collected successfully")
        return metrics
    except Exception as e:
        logger.error(f"API: Error collecting resource metrics: {str(e)}", exc_info=True)
        raise

@app.post("/scale/agent")
async def scale_agent(agent_id: str, scale_factor: float):
    logger.info(f"API: Scale agent request received for {agent_id}")
    try:
        dynamic_scaler = DynamicScaler()
        dynamic_scaler.scale_agent(agent_id, scale_factor)
        logger.info(f"API: Agent {agent_id} scaled successfully")
        return {"status": "success"}
    except Exception as e:
        logger.error(f"API: Error scaling agent: {str(e)}", exc_info=True)
        raise

@app.post("/load/balance")
async def balance_load(task: Dict[str, Any]):
    logger.info("API: Load balance request received")
    try:
        load_balancer = LoadBalancer()
        agent_id = load_balancer.distribute_load(task)
        logger.info(f"API: Task assigned to agent {agent_id}")
        return {"agent_id": agent_id}
    except Exception as e:
        logger.error(f"API: Error balancing load: {str(e)}", exc_info=True)
        raise

@app.post("/modify/architecture")
async def modify_architecture(modification: Dict[str, Any]):
    logger.info("API: Architecture modification request received")
    try:
        self_modifier = SelfModifier()
        modification_id = self_modifier.modify_architecture(modification)
        logger.info(f"API: Architecture modification {modification_id} completed")
        return {"modification_id": modification_id}
    except Exception as e:
        logger.error(f"API: Error modifying architecture: {str(e)}", exc_info=True)
        raise

@app.post("/evolve/architecture")
async def evolve_architecture(constraints: Dict[str, Any]):
    logger.info("API: Architecture evolution request received")
    try:
        evolutionary_architect = EvolutionaryArchitect()
        new_architecture = evolutionary_architect.evolve_architecture(constraints)
        logger.info("API: Architecture evolution completed")
        return {"architecture": new_architecture}
    except Exception as e:
        logger.error(f"API: Error evolving architecture: {str(e)}", exc_info=True)
        raise 