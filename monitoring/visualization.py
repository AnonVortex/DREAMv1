import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)

@dataclass
class VisualizationConfig:
    """Configuration for visualization settings."""
    update_interval: float = 1.0  # seconds
    max_history_points: int = 1000
    plot_style: str = "dark"
    coalition_graph_layout: str = "spring"
    interactive: bool = True
    save_plots: bool = True
    output_dir: str = "visualizations"
    plot_dimensions: Tuple[int, int] = (1200, 800)

class Visualizer:
    """Handles visualization of system metrics and coalition performance."""
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        """Initialize the visualizer with configuration."""
        self.config = config or VisualizationConfig()
        self.history: Dict[str, List[float]] = {}
        self.last_update = datetime.now()
        
        # Set up plotting style
        if self.config.interactive:
            plt.style.use("dark_background" if self.config.plot_style == "dark" else "default")
        
        # Initialize plotly figure objects
        self.system_fig = self._create_system_dashboard()
        self.coalition_fig = self._create_coalition_dashboard()
        self.learning_fig = self._create_learning_dashboard()
    
    def _create_system_dashboard(self) -> go.Figure:
        """Create the main system monitoring dashboard."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "System Resource Usage",
                "Network Activity",
                "Error Rates",
                "Coalition Performance"
            )
        )
        
        # Add initial empty traces
        fig.add_trace(
            go.Scatter(x=[], y=[], name="CPU Usage", line=dict(color="#00ff00")),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=[], y=[], name="Memory Usage", line=dict(color="#ff0000")),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=[], y=[], name="Messages/s", line=dict(color="#0000ff")),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(x=[], y=[], name="Bandwidth Usage", line=dict(color="#ff00ff")),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Scatter(x=[], y=[], name="Error Rate", line=dict(color="#ff0000")),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Bar(x=[], y=[], name="Coalition Success Rate"),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=self.config.plot_dimensions[1],
            width=self.config.plot_dimensions[0],
            showlegend=True,
            template="plotly_dark" if self.config.plot_style == "dark" else "plotly_white"
        )
        
        return fig
    
    def _create_coalition_dashboard(self) -> go.Figure:
        """Create the coalition relationship visualization dashboard."""
        fig = go.Figure()
        
        # Will be populated with network graph data
        fig.update_layout(
            title="Coalition Relationship Graph",
            showlegend=True,
            height=self.config.plot_dimensions[1],
            width=self.config.plot_dimensions[0],
            template="plotly_dark" if self.config.plot_style == "dark" else "plotly_white"
        )
        
        return fig
    
    def _create_learning_dashboard(self) -> go.Figure:
        """Create the learning progress visualization dashboard."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Learning Curves",
                "Policy Updates",
                "Reward Distribution",
                "Task Completion Times"
            )
        )
        
        # Add initial empty traces
        fig.add_trace(
            go.Scatter(x=[], y=[], name="Mean Reward", line=dict(color="#00ff00")),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=[], y=[], name="Policy Updates", line=dict(color="#0000ff")),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Histogram(x=[], name="Reward Distribution"),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Box(y=[], name="Completion Times"),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=self.config.plot_dimensions[1],
            width=self.config.plot_dimensions[0],
            showlegend=True,
            template="plotly_dark" if self.config.plot_style == "dark" else "plotly_white"
        )
        
        return fig
    
    def update_system_metrics(self, metrics: Dict[str, float]) -> None:
        """Update system metrics visualization."""
        try:
            current_time = datetime.now()
            
            # Update system resource traces
            self.system_fig.data[0].x = self.system_fig.data[0].x + (current_time,)
            self.system_fig.data[0].y = self.system_fig.data[0].y + (metrics["cpu_usage"],)
            
            self.system_fig.data[1].x = self.system_fig.data[1].x + (current_time,)
            self.system_fig.data[1].y = self.system_fig.data[1].y + (metrics["memory_usage"],)
            
            # Update network activity traces
            self.system_fig.data[2].x = self.system_fig.data[2].x + (current_time,)
            self.system_fig.data[2].y = self.system_fig.data[2].y + (metrics["messages_per_second"],)
            
            self.system_fig.data[3].x = self.system_fig.data[3].x + (current_time,)
            self.system_fig.data[3].y = self.system_fig.data[3].y + (metrics["bandwidth_usage"],)
            
            # Update error rate trace
            self.system_fig.data[4].x = self.system_fig.data[4].x + (current_time,)
            self.system_fig.data[4].y = self.system_fig.data[4].y + (metrics["error_rate"],)
            
            # Trim history if needed
            if len(self.system_fig.data[0].x) > self.config.max_history_points:
                for trace in self.system_fig.data:
                    trace.x = trace.x[-self.config.max_history_points:]
                    trace.y = trace.y[-self.config.max_history_points:]
            
            if self.config.save_plots:
                self.system_fig.write_html(f"{self.config.output_dir}/system_dashboard.html")
            
        except Exception as e:
            logger.error(f"Error updating system metrics visualization: {str(e)}")
    
    def update_coalition_graph(
        self,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]]
    ) -> None:
        """Update coalition relationship graph visualization."""
        try:
            # Create networkx graph
            G = nx.Graph()
            
            # Add nodes with attributes
            for node in nodes:
                G.add_node(
                    node["id"],
                    type=node["type"],
                    size=node.get("size", 20),
                    color=node.get("color", "#ffffff")
                )
            
            # Add edges with attributes
            for edge in edges:
                G.add_edge(
                    edge["source"],
                    edge["target"],
                    weight=edge.get("weight", 1.0),
                    type=edge.get("type", "default")
                )
            
            # Get node positions using specified layout
            pos = getattr(nx, f"{self.config.coalition_graph_layout}_layout")(G)
            
            # Create node trace
            node_x = []
            node_y = []
            node_colors = []
            node_sizes = []
            
            for node in G.nodes():
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                node_colors.append(G.nodes[node]["color"])
                node_sizes.append(G.nodes[node]["size"])
            
            node_trace = go.Scatter(
                x=node_x,
                y=node_y,
                mode="markers",
                hoverinfo="text",
                marker=dict(
                    color=node_colors,
                    size=node_sizes,
                    line=dict(width=2)
                )
            )
            
            # Create edge trace
            edge_x = []
            edge_y = []
            
            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
            
            edge_trace = go.Scatter(
                x=edge_x,
                y=edge_y,
                line=dict(width=0.5, color="#888"),
                hoverinfo="none",
                mode="lines"
            )
            
            # Update figure
            self.coalition_fig.data = []
            self.coalition_fig.add_trace(edge_trace)
            self.coalition_fig.add_trace(node_trace)
            
            if self.config.save_plots:
                self.coalition_fig.write_html(f"{self.config.output_dir}/coalition_graph.html")
            
        except Exception as e:
            logger.error(f"Error updating coalition graph visualization: {str(e)}")
    
    def update_learning_metrics(
        self,
        metrics: Dict[str, Any]
    ) -> None:
        """Update learning progress visualization."""
        try:
            current_time = datetime.now()
            
            # Update learning curves
            self.learning_fig.data[0].x = self.learning_fig.data[0].x + (current_time,)
            self.learning_fig.data[0].y = self.learning_fig.data[0].y + (metrics["mean_reward"],)
            
            # Update policy updates
            self.learning_fig.data[1].x = self.learning_fig.data[1].x + (current_time,)
            self.learning_fig.data[1].y = self.learning_fig.data[1].y + (metrics["policy_updates"],)
            
            # Update reward distribution
            self.learning_fig.data[2].x = metrics["reward_history"]
            
            # Update completion times
            self.learning_fig.data[3].y = metrics["completion_times"]
            
            if self.config.save_plots:
                self.learning_fig.write_html(f"{self.config.output_dir}/learning_dashboard.html")
            
        except Exception as e:
            logger.error(f"Error updating learning metrics visualization: {str(e)}")
    
    def save_all_visualizations(self) -> None:
        """Save all current visualizations to files."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            self.system_fig.write_html(
                f"{self.config.output_dir}/system_dashboard_{timestamp}.html"
            )
            self.coalition_fig.write_html(
                f"{self.config.output_dir}/coalition_graph_{timestamp}.html"
            )
            self.learning_fig.write_html(
                f"{self.config.output_dir}/learning_dashboard_{timestamp}.html"
            )
            
            logger.info(f"Saved all visualizations with timestamp {timestamp}")
            
        except Exception as e:
            logger.error(f"Error saving visualizations: {str(e)}") 