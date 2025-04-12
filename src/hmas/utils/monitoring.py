"""Monitoring utilities for H-MAS."""

import asyncio
from typing import Dict, Set, Optional, Any
import psutil
import logging
from datetime import datetime
from prometheus_client import Counter, Gauge, Histogram
import structlog
from uuid import UUID

from hmas.config import settings
from hmas.core.agent import Agent

logger = structlog.get_logger(__name__)

# Prometheus metrics
AGENT_COUNT = Gauge("hmas_agent_count", "Number of active agents")
AGENT_MEMORY = Gauge("hmas_agent_memory_bytes", "Agent memory usage", ["agent_id"])
SYSTEM_CPU = Gauge("hmas_system_cpu_percent", "System CPU usage")
SYSTEM_MEMORY = Gauge("hmas_system_memory_percent", "System memory usage")
MESSAGE_COUNT = Counter("hmas_messages_total", "Total messages processed", ["agent_id"])
MESSAGE_DURATION = Histogram(
    "hmas_message_duration_seconds",
    "Message processing duration",
    ["agent_id"]
)

class Monitor:
    """System and agent monitoring."""
    
    def __init__(self, interval: Optional[int] = None) -> None:
        """Initialize monitor.
        
        Args:
            interval: Monitoring interval in seconds (default: from settings)
        """
        self.interval = interval or settings.MONITORING_INTERVAL
        self.tracked_agents: Set[UUID] = set()
        self.metrics: Dict[str, Any] = {}
        self._running = False
        self._task: Optional[asyncio.Task] = None
        
    async def start(self) -> None:
        """Start monitoring."""
        if self._running:
            return
            
        logger.info("Starting monitoring")
        self._running = True
        self._task = asyncio.create_task(self._monitor_loop())
        
    async def stop(self) -> None:
        """Stop monitoring."""
        if not self._running:
            return
            
        logger.info("Stopping monitoring")
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            
    def track_agent(self, agent: Agent) -> None:
        """Start tracking an agent.
        
        Args:
            agent: Agent to track
        """
        self.tracked_agents.add(agent.id)
        AGENT_COUNT.inc()
        
    def untrack_agent(self, agent: Agent) -> None:
        """Stop tracking an agent.
        
        Args:
            agent: Agent to untrack
        """
        self.tracked_agents.discard(agent.id)
        AGENT_COUNT.dec()
        
    async def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while self._running:
            try:
                await self._collect_metrics()
                await self._check_thresholds()
                await asyncio.sleep(self.interval)
            except Exception as e:
                logger.error("Error in monitoring loop", error=str(e))
                
    async def _collect_metrics(self) -> None:
        """Collect system and agent metrics."""
        # System metrics
        cpu_percent = psutil.cpu_percent()
        memory_percent = psutil.virtual_memory().percent
        
        SYSTEM_CPU.set(cpu_percent)
        SYSTEM_MEMORY.set(memory_percent)
        
        self.metrics.update({
            "cpu_percent": cpu_percent,
            "memory_percent": memory_percent,
            "timestamp": datetime.utcnow()
        })
        
    async def _check_thresholds(self) -> None:
        """Check metrics against thresholds."""
        cpu_percent = self.metrics.get("cpu_percent", 0)
        memory_percent = self.metrics.get("memory_percent", 0)
        
        if cpu_percent > settings.ALERT_THRESHOLD_CPU:
            logger.warning("CPU usage above threshold",
                         cpu_percent=cpu_percent,
                         threshold=settings.ALERT_THRESHOLD_CPU)
            
        if memory_percent > settings.ALERT_THRESHOLD_MEMORY:
            logger.warning("Memory usage above threshold",
                         memory_percent=memory_percent,
                         threshold=settings.ALERT_THRESHOLD_MEMORY)
            
    def record_message(self, agent_id: UUID, duration: float) -> None:
        """Record message processing metrics.
        
        Args:
            agent_id: Agent UUID
            duration: Processing duration in seconds
        """
        MESSAGE_COUNT.labels(agent_id=str(agent_id)).inc()
        MESSAGE_DURATION.labels(agent_id=str(agent_id)).observe(duration)
        
    @property
    def is_running(self) -> bool:
        """Check if monitoring is running.
        
        Returns:
            bool: True if monitoring is running
        """
        return self._running 