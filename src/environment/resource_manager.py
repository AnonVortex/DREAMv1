"""
Resource Manager

This module manages system resources for the multi-agent environment,
including memory allocation, CPU usage, and network bandwidth.
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass
import psutil
import logging
import asyncio

@dataclass
class ResourceState:
    """Represents the current state of system resources."""
    memory_usage: float
    cpu_usage: float
    network_usage: Dict[str, float]
    available_resources: Dict[str, float]

class ResourceManager:
    """
    Manages system resources for the multi-agent environment.
    Provides methods for monitoring and allocating resources.
    """

    def __init__(self):
        """Initialize the resource manager."""
        self.state = ResourceState(
            memory_usage=0.0,
            cpu_usage=0.0,
            network_usage={},
            available_resources={}
        )
        self.logger = logging.getLogger(__name__)
        self._update_task = None

    async def start(self) -> None:
        """Start resource monitoring."""
        self._update_task = asyncio.create_task(self._update_resources())
        self.logger.info("Resource monitoring started")

    async def stop(self) -> None:
        """Stop resource monitoring."""
        if self._update_task:
            self._update_task.cancel()
            try:
                await self._update_task
            except asyncio.CancelledError:
                pass
        self.logger.info("Resource monitoring stopped")

    async def _update_resources(self) -> None:
        """Periodically update resource metrics."""
        while True:
            try:
                # Update memory usage
                memory = psutil.virtual_memory()
                self.state.memory_usage = memory.percent

                # Update CPU usage
                self.state.cpu_usage = psutil.cpu_percent()

                # Update network usage
                net_io = psutil.net_io_counters()
                self.state.network_usage = {
                    'bytes_sent': net_io.bytes_sent,
                    'bytes_recv': net_io.bytes_recv
                }

                # Update available resources
                self.state.available_resources = {
                    'memory': memory.available,
                    'cpu': psutil.cpu_count(),
                    'disk': psutil.disk_usage('/').free
                }

                await asyncio.sleep(1)  # Update every second
            except Exception as e:
                self.logger.error(f"Error updating resources: {e}")
                await asyncio.sleep(5)  # Wait longer on error

    def get_resource_state(self) -> ResourceState:
        """
        Get the current state of system resources.

        Returns:
            Current resource state
        """
        return self.state

    def allocate_resources(self, agent_name: str, resources: Dict[str, float]) -> bool:
        """
        Allocate resources to an agent.

        Args:
            agent_name: Name of the agent requesting resources
            resources: Dictionary of requested resources and amounts

        Returns:
            True if resources were allocated successfully, False otherwise
        """
        try:
            # Check if requested resources are available
            for resource, amount in resources.items():
                if resource not in self.state.available_resources:
                    self.logger.warning(f"Unknown resource type: {resource}")
                    return False
                if amount > self.state.available_resources[resource]:
                    self.logger.warning(f"Insufficient {resource} for {agent_name}")
                    return False

            # Allocate resources
            for resource, amount in resources.items():
                self.state.available_resources[resource] -= amount

            self.logger.info(f"Resources allocated to {agent_name}: {resources}")
            return True
        except Exception as e:
            self.logger.error(f"Error allocating resources: {e}")
            return False

    def release_resources(self, agent_name: str, resources: Dict[str, float]) -> None:
        """
        Release resources from an agent.

        Args:
            agent_name: Name of the agent releasing resources
            resources: Dictionary of resources to release
        """
        try:
            for resource, amount in resources.items():
                if resource in self.state.available_resources:
                    self.state.available_resources[resource] += amount

            self.logger.info(f"Resources released from {agent_name}: {resources}")
        except Exception as e:
            self.logger.error(f"Error releasing resources: {e}") 