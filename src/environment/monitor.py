"""
System Monitor

This module provides system monitoring capabilities for the multi-agent environment,
including health checks, performance metrics, and alerting.
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass
import psutil
import logging
import asyncio
from datetime import datetime

@dataclass
class SystemMetrics:
    """Represents system-wide metrics."""
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_io: Dict[str, float]
    process_count: int
    timestamp: datetime

class SystemMonitor:
    """
    Monitors system health and performance.
    Provides methods for collecting metrics and generating alerts.
    """

    def __init__(self):
        """Initialize the system monitor."""
        self.metrics = SystemMetrics(
            cpu_usage=0.0,
            memory_usage=0.0,
            disk_usage=0.0,
            network_io={},
            process_count=0,
            timestamp=datetime.now()
        )
        self.logger = logging.getLogger(__name__)
        self._monitor_task = None
        self._alert_thresholds = {
            'cpu_usage': 90.0,
            'memory_usage': 85.0,
            'disk_usage': 80.0
        }

    async def start(self) -> None:
        """Start system monitoring."""
        self._monitor_task = asyncio.create_task(self._monitor_system())
        self.logger.info("System monitoring started")

    async def stop(self) -> None:
        """Stop system monitoring."""
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        self.logger.info("System monitoring stopped")

    async def _monitor_system(self) -> None:
        """Periodically monitor system metrics."""
        while True:
            try:
                # Update CPU usage
                self.metrics.cpu_usage = psutil.cpu_percent()

                # Update memory usage
                memory = psutil.virtual_memory()
                self.metrics.memory_usage = memory.percent

                # Update disk usage
                disk = psutil.disk_usage('/')
                self.metrics.disk_usage = disk.percent

                # Update network I/O
                net_io = psutil.net_io_counters()
                self.metrics.network_io = {
                    'bytes_sent': net_io.bytes_sent,
                    'bytes_recv': net_io.bytes_recv
                }

                # Update process count
                self.metrics.process_count = len(psutil.pids())
                self.metrics.timestamp = datetime.now()

                # Check for alerts
                await self._check_alerts()

                await asyncio.sleep(1)  # Update every second
            except Exception as e:
                self.logger.error(f"Error monitoring system: {e}")
                await asyncio.sleep(5)  # Wait longer on error

    async def _check_alerts(self) -> None:
        """Check if any metrics exceed alert thresholds."""
        if self.metrics.cpu_usage > self._alert_thresholds['cpu_usage']:
            self.logger.warning(f"High CPU usage: {self.metrics.cpu_usage}%")
        
        if self.metrics.memory_usage > self._alert_thresholds['memory_usage']:
            self.logger.warning(f"High memory usage: {self.metrics.memory_usage}%")
        
        if self.metrics.disk_usage > self._alert_thresholds['disk_usage']:
            self.logger.warning(f"High disk usage: {self.metrics.disk_usage}%")

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get current system metrics.

        Returns:
            Dictionary of current system metrics
        """
        return {
            'cpu_usage': self.metrics.cpu_usage,
            'memory_usage': self.metrics.memory_usage,
            'disk_usage': self.metrics.disk_usage,
            'network_io': self.metrics.network_io,
            'process_count': self.metrics.process_count,
            'timestamp': self.metrics.timestamp.isoformat()
        }

    def set_alert_threshold(self, metric: str, threshold: float) -> None:
        """
        Set alert threshold for a specific metric.

        Args:
            metric: Name of the metric
            threshold: Threshold value
        """
        if metric in self._alert_thresholds:
            self._alert_thresholds[metric] = threshold
            self.logger.info(f"Alert threshold for {metric} set to {threshold}")
        else:
            self.logger.warning(f"Unknown metric: {metric}")

    def get_alert_thresholds(self) -> Dict[str, float]:
        """
        Get current alert thresholds.

        Returns:
            Dictionary of current alert thresholds
        """
        return self._alert_thresholds.copy() 