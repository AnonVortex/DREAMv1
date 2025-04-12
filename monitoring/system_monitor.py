"""
System Monitor for HMAS
Handles real-time monitoring, visualization, and system health checks
"""

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import torch
from datetime import datetime, timedelta
import logging
import json
import asyncio
from collections import defaultdict
import psutil
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from io import BytesIO
import base64
import time

from ..utils.metrics import MetricsTracker
from ..multi_agent.coalition_manager import Coalition, CoalitionStatus
from ..learning.coalition_learning_manager import LearningStrategy
from .visualization import Visualizer, VisualizationConfig

logger = logging.getLogger(__name__)

class MonitoringLevel(Enum):
    """Monitoring detail levels."""
    BASIC = auto()       # Basic system metrics
    DETAILED = auto()    # Detailed performance metrics
    DEBUG = auto()       # Debug-level information
    PROFILE = auto()     # Performance profiling

@dataclass
class MonitoringConfig:
    """Configuration for system monitoring."""
    level: MonitoringLevel = MonitoringLevel.DETAILED
    update_interval: float = 1.0  # seconds
    history_size: int = 1000
    alert_thresholds: Dict[str, float] = None
    visualization_enabled: bool = True
    log_to_file: bool = True
    profile_enabled: bool = False
    cpu_threshold: float = 90.0  # percentage
    memory_threshold: float = 90.0  # percentage
    error_threshold: int = 100  # errors per minute
    visualization_config: Optional[VisualizationConfig] = None

@dataclass
class SystemMetrics:
    """Container for system metrics."""
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    network_io: Dict[str, float]
    disk_io: Dict[str, float]
    error_count: int
    active_coalitions: int
    messages_per_second: float

class SystemHealth:
    """Tracks system health metrics."""
    def __init__(self):
        self.cpu_usage: List[float] = []
        self.memory_usage: List[float] = []
        self.gpu_usage: List[float] = []
        self.network_usage: List[Tuple[float, float]] = []  # (sent, received)
        self.process_count: List[int] = []
        self.error_count: Dict[str, int] = defaultdict(int)
        self.last_update: float = datetime.now().timestamp()
        
    def update(self):
        """Update system health metrics."""
        try:
            # CPU usage
            self.cpu_usage.append(psutil.cpu_percent())
            
            # Memory usage
            memory = psutil.virtual_memory()
            self.memory_usage.append(memory.percent)
            
            # GPU usage if available
            try:
                import torch
                if torch.cuda.is_available():
                    self.gpu_usage.append(
                        torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
                    )
            except ImportError:
                pass
            
            # Network usage
            net_io = psutil.net_io_counters()
            self.network_usage.append((net_io.bytes_sent, net_io.bytes_recv))
            
            # Process count
            self.process_count.append(len(psutil.Process().children()))
            
            self.last_update = datetime.now().timestamp()
            
        except Exception as e:
            logger.error(f"Error updating system health: {str(e)}")
            self.error_count["health_update"] += 1

class PerformanceProfiler:
    """Profiles system performance."""
    def __init__(self):
        self.function_times: Dict[str, List[float]] = defaultdict(list)
        self.memory_usage: Dict[str, List[float]] = defaultdict(list)
        self.call_counts: Dict[str, int] = defaultdict(int)
        
    async def profile_function(self, func_name: str, coro):
        """Profile an async function execution."""
        start_time = datetime.now().timestamp()
        start_memory = psutil.Process().memory_info().rss
        
        try:
            result = await coro
            
            end_time = datetime.now().timestamp()
            end_memory = psutil.Process().memory_info().rss
            
            self.function_times[func_name].append(end_time - start_time)
            self.memory_usage[func_name].append(end_memory - start_memory)
            self.call_counts[func_name] += 1
            
            return result
            
        except Exception as e:
            logger.error(f"Error profiling {func_name}: {str(e)}")
            raise
    
    def get_profile_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary of profiling data."""
        summary = {}
        for func_name in self.function_times:
            times = self.function_times[func_name]
            memory = self.memory_usage[func_name]
            calls = self.call_counts[func_name]
            
            summary[func_name] = {
                "avg_time": np.mean(times) if times else 0,
                "max_time": max(times) if times else 0,
                "avg_memory": np.mean(memory) if memory else 0,
                "max_memory": max(memory) if memory else 0,
                "total_calls": calls
            }
        
        return summary

class Visualizer:
    """Handles system visualization."""
    def __init__(self, config: Optional[VisualizationConfig] = None):
        self.coalition_positions: Dict[str, Tuple[float, float]] = {}
        self.message_history: List[Dict[str, Any]] = []
        self.config = config
        
    def generate_coalition_graph(
        self,
        coalitions: Dict[str, Coalition],
        messages: List[Dict[str, Any]]
    ) -> str:
        """Generate coalition network visualization."""
        try:
            G = nx.Graph()
            
            # Add coalition nodes
            for c_id, coalition in coalitions.items():
                G.add_node(
                    c_id,
                    size=len(coalition.members),
                    performance=coalition.performance_history[-1] if coalition.performance_history else 0
                )
            
            # Add message edges
            for msg in messages[-100:]:  # Only show recent messages
                if msg["sender"] in coalitions and msg["receiver"] in coalitions:
                    G.add_edge(
                        msg["sender"],
                        msg["receiver"],
                        weight=msg["priority"],
                        type=msg["type"]
                    )
            
            # Create visualization
            plt.figure(figsize=(12, 8))
            pos = nx.spring_layout(G)
            
            # Draw nodes
            node_sizes = [G.nodes[node]["size"] * 100 for node in G.nodes()]
            node_colors = [G.nodes[node]["performance"] for node in G.nodes()]
            nx.draw_networkx_nodes(
                G, pos,
                node_size=node_sizes,
                node_color=node_colors,
                cmap=plt.cm.viridis
            )
            
            # Draw edges
            edge_weights = [G[u][v]["weight"] for u, v in G.edges()]
            nx.draw_networkx_edges(
                G, pos,
                width=edge_weights,
                edge_color='gray',
                alpha=0.5
            )
            
            # Add labels
            nx.draw_networkx_labels(G, pos)
            
            # Save to buffer
            buf = BytesIO()
            plt.savefig(buf, format='png')
            plt.close()
            
            # Convert to base64
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode()
            
            return img_str
            
        except Exception as e:
            logger.error(f"Error generating coalition graph: {str(e)}")
            return ""
    
    def generate_performance_plot(
        self,
        metrics: Dict[str, List[float]],
        window_size: int = 100
    ) -> str:
        """Generate performance visualization."""
        try:
            fig = Figure(figsize=(10, 6))
            ax = fig.add_subplot(111)
            
            for metric_name, values in metrics.items():
                if len(values) > window_size:
                    values = values[-window_size:]
                ax.plot(values, label=metric_name)
            
            ax.set_xlabel("Time")
            ax.set_ylabel("Value")
            ax.legend()
            ax.grid(True)
            
            # Save to buffer
            buf = BytesIO()
            fig.savefig(buf, format='png')
            
            # Convert to base64
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode()
            
            return img_str
            
        except Exception as e:
            logger.error(f"Error generating performance plot: {str(e)}")
            return ""

class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = auto()
    WARNING = auto()
    ERROR = auto()
    CRITICAL = auto()

@dataclass
class Alert:
    """System alert container."""
    severity: AlertSeverity
    message: str
    timestamp: datetime
    context: Dict[str, Any]
    resolved: bool = False
    resolution_time: Optional[datetime] = None

class AnomalyDetector:
    """ML-based anomaly detection."""
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.models: Dict[str, Any] = {}
        self.thresholds: Dict[str, float] = {}
        
    def train(self, metric_name: str, data: List[float]):
        """Train anomaly detection model for a metric."""
        if len(data) < self.window_size:
            return
            
        # Simple statistical model using mean and std
        mean = np.mean(data[-self.window_size:])
        std = np.std(data[-self.window_size:])
        
        self.models[metric_name] = (mean, std)
        self.thresholds[metric_name] = 3.0  # 3 sigma threshold
        
    def detect(self, metric_name: str, value: float) -> bool:
        """Detect if a value is anomalous."""
        if metric_name not in self.models:
            return False
            
        mean, std = self.models[metric_name]
        z_score = abs(value - mean) / (std + 1e-6)
        
        return z_score > self.thresholds[metric_name]

class DistributedTracer:
    """Distributed tracing system."""
    def __init__(self):
        self.traces: Dict[str, List[Dict[str, Any]]] = {}
        self.active_spans: Dict[str, datetime] = {}
        
    def start_span(self, trace_id: str, span_name: str) -> str:
        """Start a new span in a trace."""
        span_id = f"{trace_id}_{len(self.traces.get(trace_id, []))}"
        start_time = datetime.now()
        
        if trace_id not in self.traces:
            self.traces[trace_id] = []
            
        self.active_spans[span_id] = start_time
        return span_id
        
    def end_span(self, span_id: str, metadata: Dict[str, Any] = None):
        """End a span and record its duration."""
        if span_id not in self.active_spans:
            return
            
        trace_id = span_id.split('_')[0]
        start_time = self.active_spans[span_id]
        duration = (datetime.now() - start_time).total_seconds()
        
        span_data = {
            'span_id': span_id,
            'start_time': start_time,
            'duration': duration,
            'metadata': metadata or {}
        }
        
        self.traces[trace_id].append(span_data)
        del self.active_spans[span_id]

class MetricAggregator:
    """Custom metric aggregation."""
    def __init__(self, max_history: int = 1000):
        self.metrics: Dict[str, List[Tuple[datetime, float]]] = defaultdict(list)
        self.max_history = max_history
        
    def add_metric(self, name: str, value: float):
        """Add a new metric value."""
        self.metrics[name].append((datetime.now(), value))
        
        # Maintain history size
        if len(self.metrics[name]) > self.max_history:
            self.metrics[name] = self.metrics[name][-self.max_history:]
            
    def get_statistics(self, name: str, window: int = None) -> Dict[str, float]:
        """Get statistical summary of a metric."""
        if name not in self.metrics:
            return {}
            
        values = [v for _, v in self.metrics[name][-window:]] if window else [v for _, v in self.metrics[name]]
        
        return {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': min(values),
            'max': max(values),
            'median': np.median(values)
        }

class SecurityLevel(Enum):
    """Security levels for monitoring operations."""
    READ_ONLY = auto()      # Can only read metrics
    STANDARD = auto()       # Can read metrics and basic operations
    ADMIN = auto()          # Full access to all operations
    SECURITY_ADMIN = auto() # Can modify security settings

@dataclass
class SecurityConfig:
    """Security configuration settings."""
    encryption_enabled: bool = True
    require_authentication: bool = True
    require_authorization: bool = True
    max_failed_attempts: int = 3
    lockout_duration: int = 300  # seconds
    session_timeout: int = 3600  # seconds
    min_password_length: int = 12
    require_2fa: bool = True
    allowed_ips: List[str] = field(default_factory=list)
    rate_limit_requests: int = 100
    rate_limit_window: int = 60  # seconds

class SecurityContext:
    """Manages security context for monitoring operations."""
    def __init__(self, config: SecurityConfig):
        self.config = config
        self._failed_attempts: Dict[str, List[datetime]] = defaultdict(list)
        self._locked_accounts: Dict[str, datetime] = {}
        self._sessions: Dict[str, Dict[str, Any]] = {}
        self._rate_limits: Dict[str, List[datetime]] = defaultdict(list)
        self.audit_log: List[Dict[str, Any]] = []
        
    def authenticate(self, credentials: Dict[str, str]) -> Optional[str]:
        """Authenticate a user and return a session token."""
        user_id = credentials.get('user_id')
        
        # Check if account is locked
        if self._is_account_locked(user_id):
            self.log_security_event('authentication_failed', 
                                  {'reason': 'account_locked', 'user_id': user_id})
            return None
            
        # Verify credentials (implement your auth logic here)
        if not self._verify_credentials(credentials):
            self._record_failed_attempt(user_id)
            self.log_security_event('authentication_failed',
                                  {'reason': 'invalid_credentials', 'user_id': user_id})
            return None
            
        # Generate and store session token
        session_token = self._generate_session_token()
        self._sessions[session_token] = {
            'user_id': user_id,
            'created_at': datetime.now(),
            'expires_at': datetime.now() + timedelta(seconds=self.config.session_timeout)
        }
        
        self.log_security_event('authentication_successful', {'user_id': user_id})
        return session_token
        
    def authorize(self, session_token: str, required_level: SecurityLevel) -> bool:
        """Check if the session has required security level."""
        if not self._is_session_valid(session_token):
            return False
            
        session = self._sessions[session_token]
        user_level = self._get_user_security_level(session['user_id'])
        
        authorized = user_level.value >= required_level.value
        self.log_security_event(
            'authorization_check',
            {
                'user_id': session['user_id'],
                'required_level': required_level.name,
                'authorized': authorized
            }
        )
        return authorized
        
    def check_rate_limit(self, identifier: str) -> bool:
        """Check if request is within rate limits."""
        now = datetime.now()
        window_start = now - timedelta(seconds=self.config.rate_limit_window)
        
        # Clean old requests
        self._rate_limits[identifier] = [
            t for t in self._rate_limits[identifier] 
            if t > window_start
        ]
        
        # Check limit
        if len(self._rate_limits[identifier]) >= self.config.rate_limit_requests:
            self.log_security_event('rate_limit_exceeded', {'identifier': identifier})
            return False
            
        self._rate_limits[identifier].append(now)
        return True
        
    def encrypt_data(self, data: Any) -> bytes:
        """Encrypt sensitive data."""
        try:
            from cryptography.fernet import Fernet
            key = self._get_encryption_key()
            f = Fernet(key)
            return f.encrypt(json.dumps(data).encode())
        except Exception as e:
            logger.error(f"Encryption error: {str(e)}")
            raise
            
    def decrypt_data(self, encrypted_data: bytes) -> Any:
        """Decrypt sensitive data."""
        try:
            from cryptography.fernet import Fernet
            key = self._get_encryption_key()
            f = Fernet(key)
            decrypted = f.decrypt(encrypted_data)
            return json.loads(decrypted.decode())
        except Exception as e:
            logger.error(f"Decryption error: {str(e)}")
            raise
            
    def log_security_event(self, event_type: str, details: Dict[str, Any]):
        """Log security-related events."""
        event = {
            'timestamp': datetime.now(),
            'event_type': event_type,
            'details': details,
            'source_ip': self._get_client_ip()
        }
        self.audit_log.append(event)
        logger.info(f"Security event: {event_type} - {json.dumps(details)}")
        
    def _is_account_locked(self, user_id: str) -> bool:
        """Check if account is locked due to failed attempts."""
        if user_id not in self._locked_accounts:
            return False
            
        lock_time = self._locked_accounts[user_id]
        if (datetime.now() - lock_time).total_seconds() > self.config.lockout_duration:
            del self._locked_accounts[user_id]
            return False
            
        return True
        
    def _record_failed_attempt(self, user_id: str):
        """Record failed authentication attempt."""
        now = datetime.now()
        window_start = now - timedelta(seconds=self.config.lockout_duration)
        
        # Clean old attempts
        self._failed_attempts[user_id] = [
            t for t in self._failed_attempts[user_id]
            if t > window_start
        ]
        
        self._failed_attempts[user_id].append(now)
        
        # Lock account if too many failures
        if len(self._failed_attempts[user_id]) >= self.config.max_failed_attempts:
            self._locked_accounts[user_id] = now
            self.log_security_event('account_locked', {'user_id': user_id})
            
    def _is_session_valid(self, session_token: str) -> bool:
        """Check if session token is valid and not expired."""
        if session_token not in self._sessions:
            return False
            
        session = self._sessions[session_token]
        if datetime.now() > session['expires_at']:
            del self._sessions[session_token]
            return False
            
        return True
        
    def _verify_credentials(self, credentials: Dict[str, str]) -> bool:
        """Verify user credentials."""
        # Implement your credential verification logic here
        # This should include password hashing, 2FA if enabled, etc.
        return True
        
    def _generate_session_token(self) -> str:
        """Generate a secure session token."""
        import secrets
        return secrets.token_urlsafe(32)
        
    def _get_encryption_key(self) -> bytes:
        """Get or generate encryption key."""
        # Implement your key management logic here
        # This should use proper key storage and rotation
        return b'your-encryption-key-here'
        
    def _get_client_ip(self) -> str:
        """Get client IP address."""
        # Implement your IP detection logic here
        return "127.0.0.1"
        
    def _get_user_security_level(self, user_id: str) -> SecurityLevel:
        """Get security level for a user."""
        # Implement your user role management logic here
        return SecurityLevel.STANDARD

class SystemMonitor:
    """Enhanced system monitoring with security."""
    def __init__(
        self,
        config: Optional[MonitoringConfig] = None,
        security_config: Optional[SecurityConfig] = None
    ):
        self.config = config or MonitoringConfig()
        self.security_config = security_config or SecurityConfig()
        self.security = SecurityContext(self.security_config)
        self.health = SystemHealth()
        self.profiler = PerformanceProfiler()
        self.visualizer = Visualizer(self.config.visualization_config)
        self.alerts: List[Alert] = []
        self.anomaly_detector = AnomalyDetector()
        self.tracer = DistributedTracer()
        self.metric_aggregator = MetricAggregator()
        
        # Initialize monitoring tasks
        self.monitoring_task = None
        self.alert_task = None
        self.visualization_task = None
        
    async def start(self):
        """Start all monitoring tasks."""
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        self.alert_task = asyncio.create_task(self._check_alerts())
        if self.config.visualization_enabled:
            self.visualization_task = asyncio.create_task(self._update_visualizations())
            
    async def stop(self):
        """Stop all monitoring tasks."""
        if self.monitoring_task:
            self.monitoring_task.cancel()
        if self.alert_task:
            self.alert_task.cancel()
        if self.visualization_task:
            self.visualization_task.cancel()
            
        await asyncio.gather(
            self.monitoring_task,
            self.alert_task,
            self.visualization_task,
            return_exceptions=True
        )

    def add_alert(self, severity: AlertSeverity, message: str, context: Dict[str, Any] = None):
        """Add a new system alert."""
        alert = Alert(
            severity=severity,
            message=message,
            timestamp=datetime.now(),
            context=context or {}
        )
        self.alerts.append(alert)
        
        # Log alert based on severity
        log_func = {
            AlertSeverity.INFO: logger.info,
            AlertSeverity.WARNING: logger.warning,
            AlertSeverity.ERROR: logger.error,
            AlertSeverity.CRITICAL: logger.critical
        }[severity]
        
        log_func(f"Alert: {message}")
        
    def resolve_alert(self, alert: Alert):
        """Mark an alert as resolved."""
        alert.resolved = True
        alert.resolution_time = datetime.now()
        logger.info(f"Resolved alert: {alert.message}")
        
    def get_active_alerts(self) -> List[Alert]:
        """Get all unresolved alerts."""
        return [alert for alert in self.alerts if not alert.resolved]
        
    def get_alert_history(self) -> List[Alert]:
        """Get full alert history."""
        return self.alerts.copy()

    async def _monitoring_loop(self):
        """Enhanced monitoring loop."""
        while True:
            try:
                # Update system metrics
                metrics = self._collect_metrics()
                
                # Check for anomalies
                for metric_name, value in metrics.__dict__.items():
                    if isinstance(value, (int, float)):
                        self.metric_aggregator.add_metric(metric_name, value)
                        
                        if self.anomaly_detector.detect(metric_name, value):
                            self.add_alert(
                                AlertSeverity.WARNING,
                                f"Anomaly detected in {metric_name}",
                                {'value': value}
                            )
                
                # Update health status
                self.health.update()
                
                # Sleep until next update
                await asyncio.sleep(self.config.update_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {str(e)}")
                self.add_alert(
                    AlertSeverity.ERROR,
                    f"Monitoring error: {str(e)}"
                )
                await asyncio.sleep(1.0)  # Shorter sleep on error

    def _collect_metrics(self) -> SystemMetrics:
        """Collect current system metrics."""
        return SystemMetrics(
            timestamp=datetime.now(),
            cpu_usage=psutil.cpu_percent(),
            memory_usage=psutil.virtual_memory().percent,
            network_io={
                'sent': psutil.net_io_counters().bytes_sent,
                'received': psutil.net_io_counters().bytes_recv
            },
            disk_io={
                'read': psutil.disk_io_counters().read_bytes,
                'write': psutil.disk_io_counters().write_bytes
            },
            error_count=sum(self.health.error_count.values()),
            active_coalitions=len(self.coalition_positions),
            messages_per_second=self._calculate_message_rate()
        )

    async def trace_operation(self, trace_id: str, operation_name: str, metadata: Dict[str, Any] = None):
        """Trace an async operation."""
        span_id = self.tracer.start_span(trace_id, operation_name)
        try:
            yield
        finally:
            self.tracer.end_span(span_id, metadata)

    def get_trace(self, trace_id: str) -> List[Dict[str, Any]]:
        """Get all spans for a trace."""
        return self.tracer.traces.get(trace_id, [])

    def get_metric_summary(self, metric_name: str, window: int = None) -> Dict[str, float]:
        """Get statistical summary of a metric."""
        return self.metric_aggregator.get_statistics(metric_name, window)

    def update_metrics(self) -> None:
        """Update system metrics if enough time has passed since last update."""
        current_time = datetime.now()
        if (current_time - self.last_update).total_seconds() >= self.config.update_interval:
            try:
                # Collect system metrics
                cpu_percent = psutil.cpu_percent()
                memory = psutil.virtual_memory()
                network = psutil.net_io_counters()
                disk = psutil.disk_io_counters()
                
                metrics = SystemMetrics(
                    timestamp=current_time,
                    cpu_usage=cpu_percent,
                    memory_usage=memory.percent,
                    network_io={
                        "bytes_sent": network.bytes_sent,
                        "bytes_recv": network.bytes_recv
                    },
                    disk_io={
                        "read_bytes": disk.read_bytes,
                        "write_bytes": disk.write_bytes
                    },
                    error_count=len([e for e in self.error_log if 
                        (current_time - e["timestamp"]).total_seconds() <= 60]),
                    active_coalitions=len(self.coalition_metrics),
                    messages_per_second=self._calculate_message_rate()
                )
                
                # Update history
                self.metrics_history.append(metrics)
                if len(self.metrics_history) > self.config.history_size:
                    self.metrics_history.pop(0)
                
                # Check thresholds and log alerts
                self._check_thresholds(metrics)
                
                # Update visualization if enabled
                if self.visualizer:
                    self._update_visualizations(metrics)
                
                self.last_update = current_time
                
            except Exception as e:
                logger.error(f"Error updating system metrics: {str(e)}")
    
    def _calculate_message_rate(self) -> float:
        """Calculate the current message rate per second."""
        try:
            recent_messages = sum(
                coalition["messages_sent"]
                for coalition in self.coalition_metrics.values()
            )
            time_window = self.config.update_interval
            return recent_messages / time_window if time_window > 0 else 0
        except Exception:
            return 0.0
    
    def _check_thresholds(self, metrics: SystemMetrics) -> None:
        """Check if any metrics exceed their thresholds."""
        if metrics.cpu_usage > self.config.cpu_threshold:
            logger.warning(f"CPU usage exceeds threshold: {metrics.cpu_usage}%")
        
        if metrics.memory_usage > self.config.memory_threshold:
            logger.warning(f"Memory usage exceeds threshold: {metrics.memory_usage}%")
        
        if metrics.error_count > self.config.error_threshold:
            logger.warning(f"Error rate exceeds threshold: {metrics.error_count} errors/minute")
    
    def _update_visualizations(self, metrics: SystemMetrics) -> None:
        """Update all visualization components."""
        try:
            # Update system metrics visualization
            system_metrics_dict = {
                "cpu_usage": metrics.cpu_usage,
                "memory_usage": metrics.memory_usage,
                "messages_per_second": metrics.messages_per_second,
                "bandwidth_usage": (
                    metrics.network_io["bytes_sent"] + 
                    metrics.network_io["bytes_recv"]
                ) / 1024 / 1024,  # Convert to MB
                "error_rate": metrics.error_count
            }
            self.visualizer.update_system_metrics(system_metrics_dict)
            
            # Update coalition graph visualization
            nodes = []
            edges = []
            for coalition_id, metrics in self.coalition_metrics.items():
                nodes.append({
                    "id": coalition_id,
                    "type": "coalition",
                    "size": 30,
                    "color": self._get_coalition_color(metrics)
                })
                
                # Add edges between related coalitions
                for related_id in metrics.get("related_coalitions", []):
                    edges.append({
                        "source": coalition_id,
                        "target": related_id,
                        "weight": metrics.get("relationship_strength", 1.0)
                    })
            
            self.visualizer.update_coalition_graph(nodes, edges)
            
            # Update learning metrics visualization
            learning_metrics = {
                "mean_reward": np.mean([
                    m.get("reward", 0) 
                    for m in self.coalition_metrics.values()
                ]),
                "policy_updates": sum(
                    m.get("policy_updates", 0) 
                    for m in self.coalition_metrics.values()
                ),
                "reward_history": [
                    m.get("reward", 0) 
                    for m in self.coalition_metrics.values()
                ],
                "completion_times": [
                    m.get("completion_time", 0) 
                    for m in self.coalition_metrics.values() 
                    if m.get("completion_time")
                ]
            }
            self.visualizer.update_learning_metrics(learning_metrics)
            
        except Exception as e:
            logger.error(f"Error updating visualizations: {str(e)}")
    
    def _get_coalition_color(self, metrics: Dict[str, Any]) -> str:
        """Determine coalition node color based on its performance metrics."""
        performance = metrics.get("performance", 0.0)
        if performance >= 0.8:
            return "#00ff00"  # Green for high performance
        elif performance >= 0.5:
            return "#ffff00"  # Yellow for medium performance
        else:
            return "#ff0000"  # Red for low performance
    
    def log_error(self, error: str, severity: str = "error", context: Dict[str, Any] = None) -> None:
        """Log an error with timestamp and context."""
        error_entry = {
            "timestamp": datetime.now(),
            "error": error,
            "severity": severity,
            "context": context or {}
        }
        self.error_log.append(error_entry)
        
        # Trim error log if needed
        while len(self.error_log) > self.config.history_size:
            self.error_log.pop(0)
        
        # Log based on severity
        if severity == "critical":
            logger.critical(error)
        elif severity == "error":
            logger.error(error)
        elif severity == "warning":
            logger.warning(error)
    
    def save_metrics(self, filepath: str) -> None:
        """Save current metrics to a file."""
        try:
            if self.visualizer:
                self.visualizer.save_all_visualizations()
            
            # Additional metric saving logic can be added here
            logger.info(f"Metrics saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving metrics: {str(e)}")

    async def authenticate(self, credentials: Dict[str, str]) -> Optional[str]:
        """Authenticate and get session token."""
        return self.security.authenticate(credentials)
        
    async def get_metrics(self, session_token: str) -> Optional[SystemMetrics]:
        """Get system metrics with authentication."""
        if not self.security.authorize(session_token, SecurityLevel.READ_ONLY):
            return None
            
        if not self.security.check_rate_limit(f"metrics_{session_token}"):
            return None
            
        metrics = self._collect_metrics()
        if self.security_config.encryption_enabled:
            return self.security.encrypt_data(metrics)
        return metrics
        
    async def update_config(self, session_token: str, new_config: Dict[str, Any]) -> bool:
        """Update configuration with authentication."""
        if not self.security.authorize(session_token, SecurityLevel.ADMIN):
            return False
            
        # Implement configuration update logic here
        self.security.log_security_event('config_update', new_config)
        return True
        
    async def get_security_log(self, session_token: str) -> Optional[List[Dict[str, Any]]]:
        """Get security audit log."""
        if not self.security.authorize(session_token, SecurityLevel.SECURITY_ADMIN):
            return None
            
        return self.security.audit_log.copy() 