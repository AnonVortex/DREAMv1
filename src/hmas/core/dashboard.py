"""Web-based dashboard for monitoring H-MAS training."""

from fastapi import FastAPI, WebSocket, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import json
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from dataclasses import dataclass
import logging
from .distributed import DistributedTrainer, DistributedConfig
from .training import TrainingConfig
from .visualization import VisualizationConfig
from .environments import EnvironmentType

@dataclass
class DashboardConfig:
    """Configuration for dashboard settings."""
    host: str = "localhost"
    port: int = 8000
    update_interval: float = 1.0  # seconds
    max_history: int = 10000
    static_dir: str = "dashboard/static"
    template_dir: str = "dashboard/templates"

class Dashboard:
    """Web-based dashboard for monitoring training."""
    
    def __init__(
        self,
        config: DashboardConfig,
        trainer: DistributedTrainer
    ):
        """Initialize dashboard."""
        self.config = config
        self.trainer = trainer
        self.app = FastAPI()
        
        # Setup logging
        self.logger = logging.getLogger("dashboard")
        
        # Initialize metrics storage
        self.metrics_history = {
            env_type.value: {
                "timestamps": [],
                "rewards": [],
                "success_rates": [],
                "learning_progress": []
            }
            for env_type in EnvironmentType
        }
        
        # Active WebSocket connections
        self.active_connections: List[WebSocket] = []
        
        # Setup routes
        self.setup_routes()
        
        # Create static directories
        Path(config.static_dir).mkdir(parents=True, exist_ok=True)
        Path(config.template_dir).mkdir(parents=True, exist_ok=True)
        
        # Mount static files
        self.app.mount(
            "/static",
            StaticFiles(directory=config.static_dir),
            name="static"
        )
        
    def setup_routes(self) -> None:
        """Setup dashboard routes."""
        @self.app.get("/", response_class=HTMLResponse)
        async def root():
            return self.get_dashboard_html()
            
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            self.active_connections.append(websocket)
            try:
                while True:
                    data = await websocket.receive_json()
                    await self.handle_websocket_message(websocket, data)
            except:
                self.active_connections.remove(websocket)
                
        @self.app.get("/api/metrics")
        async def get_metrics():
            return self.get_current_metrics()
            
        @self.app.post("/api/config")
        async def update_config(config: Dict[str, Any]):
            return await self.update_training_config(config)
            
    def get_dashboard_html(self) -> str:
        """Generate dashboard HTML."""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>H-MAS Training Dashboard</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
            <link href="/static/dashboard.css" rel="stylesheet">
        </head>
        <body>
            <div class="container-fluid">
                <div class="row">
                    <div class="col-12">
                        <h1>H-MAS Training Dashboard</h1>
                    </div>
                </div>
                <div class="row">
                    <div class="col-md-6">
                        <div class="card">
                            <div class="card-header">
                                Training Progress
                            </div>
                            <div class="card-body">
                                <div id="training-progress"></div>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-6">
                        <div class="card">
                            <div class="card-header">
                                Environment Performance
                            </div>
                            <div class="card-body">
                                <div id="environment-performance"></div>
                            </div>
                        </div>
                    </div>
                </div>
                <div class="row mt-4">
                    <div class="col-md-4">
                        <div class="card">
                            <div class="card-header">
                                Worker Status
                            </div>
                            <div class="card-body">
                                <div id="worker-status"></div>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-4">
                        <div class="card">
                            <div class="card-header">
                                Learning Curves
                            </div>
                            <div class="card-body">
                                <div id="learning-curves"></div>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-4">
                        <div class="card">
                            <div class="card-header">
                                Configuration
                            </div>
                            <div class="card-body">
                                <form id="config-form">
                                    <!-- Config form will be dynamically populated -->
                                </form>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            <script src="/static/dashboard.js"></script>
        </body>
        </html>
        """
        
    async def update_metrics(self) -> None:
        """Update dashboard metrics."""
        while True:
            # Get current metrics from trainer
            metrics = self.trainer.performance_tracker.generate_performance_report()
            timestamp = datetime.now()
            
            # Update metrics history
            for env_type, env_metrics in metrics.items():
                history = self.metrics_history[env_type]
                history["timestamps"].append(timestamp)
                history["rewards"].append(env_metrics["average_reward"])
                history["success_rates"].append(env_metrics["success_rate"])
                history["learning_progress"].append(env_metrics["learning_progress"])
                
                # Trim history if needed
                if len(history["timestamps"]) > self.config.max_history:
                    for key in history:
                        history[key] = history[key][-self.config.max_history:]
                        
            # Broadcast updates to all connected clients
            await self.broadcast_metrics(metrics)
            
            # Wait for next update
            await asyncio.sleep(self.config.update_interval)
            
    async def broadcast_metrics(self, metrics: Dict[str, Any]) -> None:
        """Broadcast metrics to all connected clients."""
        if not self.active_connections:
            return
            
        # Prepare data for visualization
        visualization_data = self.prepare_visualization_data(metrics)
        
        # Broadcast to all connections
        for connection in self.active_connections:
            try:
                await connection.send_json({
                    "type": "metrics_update",
                    "data": visualization_data
                })
            except:
                self.active_connections.remove(connection)
                
    def prepare_visualization_data(
        self,
        metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Prepare metrics data for visualization."""
        # Training progress plot
        progress_data = {
            env_type: {
                "x": [ts.isoformat() for ts in history["timestamps"]],
                "y": history["learning_progress"],
                "name": env_type
            }
            for env_type, history in self.metrics_history.items()
        }
        
        # Environment performance plot
        performance_data = {
            env_type: {
                "x": [ts.isoformat() for ts in history["timestamps"]],
                "y": history["rewards"],
                "name": env_type
            }
            for env_type, history in self.metrics_history.items()
        }
        
        # Worker status
        worker_status = {
            f"Worker {i}": {
                "active": True,  # TODO: Implement worker health check
                "steps": metrics.get(f"worker_{i}_steps", 0),
                "success_rate": metrics.get(f"worker_{i}_success_rate", 0.0)
            }
            for i in range(self.trainer.config.num_workers)
        }
        
        # Learning curves
        learning_curves = {
            env_type: {
                "x": [ts.isoformat() for ts in history["timestamps"]],
                "y": history["success_rates"],
                "name": env_type
            }
            for env_type, history in self.metrics_history.items()
        }
        
        return {
            "progress": progress_data,
            "performance": performance_data,
            "worker_status": worker_status,
            "learning_curves": learning_curves
        }
        
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current training metrics."""
        return self.trainer.performance_tracker.generate_performance_report()
        
    async def update_training_config(
        self,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Update training configuration."""
        # TODO: Implement configuration update
        return {"status": "success"}
        
    async def handle_websocket_message(
        self,
        websocket: WebSocket,
        message: Dict[str, Any]
    ) -> None:
        """Handle incoming WebSocket messages."""
        message_type = message.get("type")
        
        if message_type == "config_update":
            await self.update_training_config(message["data"])
        elif message_type == "pause_training":
            # TODO: Implement training pause
            pass
        elif message_type == "resume_training":
            # TODO: Implement training resume
            pass
            
    def write_static_files(self) -> None:
        """Write static CSS and JavaScript files."""
        # Write CSS
        css_content = """
        body {
            padding: 20px;
            background-color: #f8f9fa;
        }
        
        .card {
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        
        .card-header {
            background-color: #007bff;
            color: white;
            font-weight: bold;
        }
        
        #config-form {
            max-height: 400px;
            overflow-y: auto;
        }
        """
        
        css_path = Path(self.config.static_dir) / "dashboard.css"
        css_path.write_text(css_content)
        
        # Write JavaScript
        js_content = """
        // WebSocket connection
        const ws = new WebSocket(`ws://${window.location.host}/ws`);
        
        // Plotly layout defaults
        const layoutDefaults = {
            margin: { t: 30, r: 30, b: 40, l: 50 },
            height: 300,
            showlegend: true,
            legend: { orientation: 'h', y: -0.2 }
        };
        
        // Initialize plots
        function initializePlots() {
            Plotly.newPlot('training-progress', [], {
                ...layoutDefaults,
                title: 'Training Progress',
                xaxis: { title: 'Time' },
                yaxis: { title: 'Progress' }
            });
            
            Plotly.newPlot('environment-performance', [], {
                ...layoutDefaults,
                title: 'Environment Performance',
                xaxis: { title: 'Time' },
                yaxis: { title: 'Reward' }
            });
            
            Plotly.newPlot('learning-curves', [], {
                ...layoutDefaults,
                title: 'Learning Curves',
                xaxis: { title: 'Time' },
                yaxis: { title: 'Success Rate' }
            });
        }
        
        // Update plots with new data
        function updatePlots(data) {
            // Update training progress
            const progressTraces = Object.entries(data.progress).map(([env, values]) => ({
                x: values.x,
                y: values.y,
                name: env,
                type: 'scatter',
                mode: 'lines'
            }));
            Plotly.react('training-progress', progressTraces, {
                ...layoutDefaults,
                title: 'Training Progress'
            });
            
            // Update environment performance
            const performanceTraces = Object.entries(data.performance).map(([env, values]) => ({
                x: values.x,
                y: values.y,
                name: env,
                type: 'scatter',
                mode: 'lines'
            }));
            Plotly.react('environment-performance', performanceTraces, {
                ...layoutDefaults,
                title: 'Environment Performance'
            });
            
            // Update learning curves
            const learningTraces = Object.entries(data.learning_curves).map(([env, values]) => ({
                x: values.x,
                y: values.y,
                name: env,
                type: 'scatter',
                mode: 'lines'
            }));
            Plotly.react('learning-curves', learningTraces, {
                ...layoutDefaults,
                title: 'Learning Curves'
            });
            
            // Update worker status
            updateWorkerStatus(data.worker_status);
        }
        
        // Update worker status display
        function updateWorkerStatus(status) {
            const statusHtml = Object.entries(status).map(([worker, info]) => `
                <div class="worker-status ${info.active ? 'active' : 'inactive'}">
                    <h5>${worker}</h5>
                    <p>Steps: ${info.steps}</p>
                    <p>Success Rate: ${(info.success_rate * 100).toFixed(1)}%</p>
                </div>
            `).join('');
            
            document.getElementById('worker-status').innerHTML = statusHtml;
        }
        
        // WebSocket message handling
        ws.onmessage = function(event) {
            const message = JSON.parse(event.data);
            if (message.type === 'metrics_update') {
                updatePlots(message.data);
            }
        };
        
        // Initialize on page load
        document.addEventListener('DOMContentLoaded', function() {
            initializePlots();
            
            // Setup configuration form
            fetch('/api/metrics')
                .then(response => response.json())
                .then(data => {
                    // TODO: Populate configuration form
                });
        });
        """
        
        js_path = Path(self.config.static_dir) / "dashboard.js"
        js_path.write_text(js_content)
        
    async def run(self) -> None:
        """Run the dashboard server."""
        # Write static files
        self.write_static_files()
        
        # Start metrics update task
        background_tasks = BackgroundTasks()
        background_tasks.add_task(self.update_metrics)
        
        # Start FastAPI server
        import uvicorn
        config = uvicorn.Config(
            self.app,
            host=self.config.host,
            port=self.config.port,
            log_level="info"
        )
        server = uvicorn.Server(config)
        await server.serve()
        
    async def shutdown(self) -> None:
        """Shutdown the dashboard server."""
        # Close all WebSocket connections
        for connection in self.active_connections:
            await connection.close()
        self.active_connections.clear() 