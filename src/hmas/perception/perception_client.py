"""Client for interacting with the perception module."""

import requests
import websockets
import asyncio
import json
import numpy as np
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import logging

@dataclass
class PerceptionConfig:
    """Configuration for perception module connection."""
    host: str = "localhost"
    port: int = 8100
    ws_port: int = 8101
    timeout: float = 2.0

class PerceptionClient:
    """Client for interacting with the perception module."""
    
    def __init__(self, config: Optional[PerceptionConfig] = None):
        self.config = config or PerceptionConfig()
        self.base_url = f"http://{self.config.host}:{self.config.port}"
        self.ws_url = f"ws://{self.config.host}:{self.config.ws_port}"
        self.ws_connection = None
        self.logger = logging.getLogger(__name__)
        
    async def connect(self) -> bool:
        """Establish WebSocket connection to perception module."""
        try:
            self.ws_connection = await websockets.connect(self.ws_url)
            return True
        except Exception as e:
            self.logger.error(f"Failed to connect to perception module: {e}")
            return False
            
    async def disconnect(self):
        """Close WebSocket connection."""
        if self.ws_connection:
            await self.ws_connection.close()
            self.ws_connection = None
            
    def check_health(self) -> Dict[str, Any]:
        """Check health status of perception module."""
        try:
            response = requests.get(
                f"{self.base_url}/health",
                timeout=self.config.timeout
            )
            return response.json()
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return {"status": "error", "message": str(e)}
            
    def get_perception_config(self) -> Dict[str, Any]:
        """Get current perception module configuration."""
        try:
            response = requests.get(
                f"{self.base_url}/config",
                timeout=self.config.timeout
            )
            return response.json()
        except Exception as e:
            self.logger.error(f"Failed to get config: {e}")
            return {}
            
    def update_perception_config(self, config: Dict[str, Any]) -> bool:
        """Update perception module configuration."""
        try:
            response = requests.post(
                f"{self.base_url}/config",
                json=config,
                timeout=self.config.timeout
            )
            return response.status_code == 200
        except Exception as e:
            self.logger.error(f"Failed to update config: {e}")
            return False
            
    async def get_sensor_data(self) -> Dict[str, Any]:
        """Get latest sensor data from perception module."""
        if not self.ws_connection:
            if not await self.connect():
                return {}
                
        try:
            await self.ws_connection.send(json.dumps({
                "type": "get_sensor_data"
            }))
            response = await self.ws_connection.recv()
            return json.loads(response)
        except Exception as e:
            self.logger.error(f"Failed to get sensor data: {e}")
            return {}
            
    async def get_object_detections(self) -> List[Dict[str, Any]]:
        """Get latest object detections."""
        if not self.ws_connection:
            if not await self.connect():
                return []
                
        try:
            await self.ws_connection.send(json.dumps({
                "type": "get_object_detections"
            }))
            response = await self.ws_connection.recv()
            return json.loads(response)
        except Exception as e:
            self.logger.error(f"Failed to get object detections: {e}")
            return []
            
    async def get_perception_metrics(self) -> Dict[str, Any]:
        """Get current perception performance metrics."""
        if not self.ws_connection:
            if not await self.connect():
                return {}
                
        try:
            await self.ws_connection.send(json.dumps({
                "type": "get_metrics"
            }))
            response = await self.ws_connection.recv()
            return json.loads(response)
        except Exception as e:
            self.logger.error(f"Failed to get metrics: {e}")
            return {}
            
    def process_image(self, image: np.ndarray) -> Dict[str, Any]:
        """Process a single image through perception module."""
        try:
            # Convert image to base64 for transmission
            import base64
            import cv2
            _, buffer = cv2.imencode('.jpg', image)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            
            response = requests.post(
                f"{self.base_url}/process_image",
                json={"image": img_base64},
                timeout=self.config.timeout
            )
            return response.json()
        except Exception as e:
            self.logger.error(f"Failed to process image: {e}")
            return {}
            
    def start_perception(self) -> bool:
        """Start perception processing."""
        try:
            response = requests.post(
                f"{self.base_url}/start",
                timeout=self.config.timeout
            )
            return response.status_code == 200
        except Exception as e:
            self.logger.error(f"Failed to start perception: {e}")
            return False
            
    def stop_perception(self) -> bool:
        """Stop perception processing."""
        try:
            response = requests.post(
                f"{self.base_url}/stop",
                timeout=self.config.timeout
            )
            return response.status_code == 200
        except Exception as e:
            self.logger.error(f"Failed to stop perception: {e}")
            return False
            
    async def subscribe_to_detections(self, callback) -> None:
        """Subscribe to real-time object detections."""
        if not self.ws_connection:
            if not await self.connect():
                return
                
        try:
            await self.ws_connection.send(json.dumps({
                "type": "subscribe",
                "stream": "detections"
            }))
            
            while True:
                message = await self.ws_connection.recv()
                data = json.loads(message)
                await callback(data)
        except Exception as e:
            self.logger.error(f"Detection subscription error: {e}")
            
    async def subscribe_to_metrics(self, callback) -> None:
        """Subscribe to real-time performance metrics."""
        if not self.ws_connection:
            if not await self.connect():
                return
                
        try:
            await self.ws_connection.send(json.dumps({
                "type": "subscribe",
                "stream": "metrics"
            }))
            
            while True:
                message = await self.ws_connection.recv()
                data = json.loads(message)
                await callback(data)
        except Exception as e:
            self.logger.error(f"Metrics subscription error: {e}") 