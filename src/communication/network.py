"""
Network Manager

This module provides network management capabilities for the multi-agent system,
including connection handling, network monitoring, and security.
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import asyncio
import logging
import socket
import ssl
from datetime import datetime

@dataclass
class NetworkStats:
    """Represents network statistics."""
    bytes_sent: int
    bytes_received: int
    connection_count: int
    error_count: int
    last_update: datetime

class NetworkManager:
    """
    Manages network operations for the multi-agent system.
    Handles connections, security, and network monitoring.
    """

    def __init__(self):
        """Initialize the network manager."""
        self.connections: Dict[str, Any] = {}
        self.stats = NetworkStats(
            bytes_sent=0,
            bytes_received=0,
            connection_count=0,
            error_count=0,
            last_update=datetime.now()
        )
        self.logger = logging.getLogger(__name__)
        self._monitor_task = None

    async def start(self) -> None:
        """Start network monitoring."""
        self._monitor_task = asyncio.create_task(self._monitor_network())
        self.logger.info("Network monitoring started")

    async def stop(self) -> None:
        """Stop network monitoring."""
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        self.logger.info("Network monitoring stopped")

    async def _monitor_network(self) -> None:
        """Periodically monitor network statistics."""
        while True:
            try:
                # Update network statistics
                self.stats.last_update = datetime.now()
                # Implementation would go here
                await asyncio.sleep(1)
            except Exception as e:
                self.logger.error(f"Error monitoring network: {e}")
                await asyncio.sleep(5)

    async def create_connection(self, host: str, port: int, 
                              protocol: str = "tcp", 
                              ssl_context: Optional[ssl.SSLContext] = None) -> str:
        """
        Create a new network connection.

        Args:
            host: Host to connect to
            port: Port to connect to
            protocol: Network protocol to use
            ssl_context: SSL context for secure connections

        Returns:
            Connection identifier

        Raises:
            ConnectionError: If connection fails
        """
        try:
            connection_id = f"{host}:{port}"
            if connection_id in self.connections:
                raise ConnectionError(f"Connection already exists: {connection_id}")

            # Create connection based on protocol
            if protocol == "tcp":
                reader, writer = await asyncio.open_connection(
                    host=host,
                    port=port,
                    ssl=ssl_context
                )
                self.connections[connection_id] = {
                    'reader': reader,
                    'writer': writer,
                    'protocol': protocol,
                    'created_at': datetime.now()
                }
            else:
                raise ValueError(f"Unsupported protocol: {protocol}")

            self.stats.connection_count += 1
            self.logger.info(f"Connection created: {connection_id}")
            return connection_id
        except Exception as e:
            self.stats.error_count += 1
            self.logger.error(f"Error creating connection: {e}")
            raise ConnectionError(f"Failed to create connection: {e}")

    async def close_connection(self, connection_id: str) -> None:
        """
        Close a network connection.

        Args:
            connection_id: Identifier of the connection to close
        """
        try:
            if connection_id in self.connections:
                connection = self.connections[connection_id]
                if 'writer' in connection:
                    connection['writer'].close()
                    await connection['writer'].wait_closed()
                del self.connections[connection_id]
                self.stats.connection_count -= 1
                self.logger.info(f"Connection closed: {connection_id}")
        except Exception as e:
            self.stats.error_count += 1
            self.logger.error(f"Error closing connection: {e}")

    async def send_data(self, connection_id: str, data: bytes) -> None:
        """
        Send data over a connection.

        Args:
            connection_id: Identifier of the connection
            data: Data to send

        Raises:
            ConnectionError: If connection is not found or sending fails
        """
        try:
            if connection_id not in self.connections:
                raise ConnectionError(f"Connection not found: {connection_id}")

            connection = self.connections[connection_id]
            writer = connection['writer']
            writer.write(data)
            await writer.drain()
            self.stats.bytes_sent += len(data)
        except Exception as e:
            self.stats.error_count += 1
            self.logger.error(f"Error sending data: {e}")
            raise ConnectionError(f"Failed to send data: {e}")

    async def receive_data(self, connection_id: str, buffer_size: int = 1024) -> bytes:
        """
        Receive data from a connection.

        Args:
            connection_id: Identifier of the connection
            buffer_size: Size of the receive buffer

        Returns:
            Received data

        Raises:
            ConnectionError: If connection is not found or receiving fails
        """
        try:
            if connection_id not in self.connections:
                raise ConnectionError(f"Connection not found: {connection_id}")

            connection = self.connections[connection_id]
            reader = connection['reader']
            data = await reader.read(buffer_size)
            self.stats.bytes_received += len(data)
            return data
        except Exception as e:
            self.stats.error_count += 1
            self.logger.error(f"Error receiving data: {e}")
            raise ConnectionError(f"Failed to receive data: {e}")

    def get_network_stats(self) -> NetworkStats:
        """
        Get current network statistics.

        Returns:
            Current network statistics
        """
        return self.stats

    def get_active_connections(self) -> List[str]:
        """
        Get list of active connections.

        Returns:
            List of connection identifiers
        """
        return list(self.connections.keys()) 