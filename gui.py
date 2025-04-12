import streamlit as st
import numpy as np
from src.hmas.environments.tasks_3d import (
    TaskType,
    TaskParameters,
    create_task,
    save_task,
    load_task,
    TaskExecutor,
    TaskResult,
    ExecutionError
)
from pathlib import Path
import plotly.graph_objects as go
from typing import Dict, Any, List, Tuple
import json
from src.hmas.perception.perception_client import PerceptionClient, PerceptionConfig
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading
from dataclasses import dataclass
from enum import Enum, auto
import pandas as pd
from datetime import datetime, timedelta
import os
from scipy import stats
import seaborn as sns
from sklearn.metrics import confusion_matrix
import copy
import requests
import plotly.express as px
import networkx as nx
import time

# This must be the very first Streamlit command.
st.set_page_config(
    page_title="HMAS Monitor",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

import subprocess
import requests
import time

# Optional auto-refresh library; install with: pip install streamlit-autorefresh
try:
    from streamlit_autorefresh import st_autorefresh
except ImportError:
    st_autorefresh = None

# -------------------- Custom CSS for Futuristic Look --------------------
st.markdown("""
    <style>
        body {
            background-color: #121212;
            color: #0aff0a;
            font-family: 'Orbitron', sans-serif;
        }
        .stButton>button {
            background-color: #0aff0a;
            color: #121212;
            border-radius: 5px;
            font-weight: bold;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #0be70b;
            transform: scale(1.05);
        }
        .sidebar .sidebarContent {
            background-color: #1e1e1e;
            color: #0aff0a;
        }
        .stMarkdown, .stText, .stTextArea {
            font-size: 16px;
        }
    </style>
    <link href="https://fonts.googleapis.com/css2?family=Orbitron&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)

# -------------------- Helper Functions --------------------
def run_command(cmd):
    """Execute a shell command and return its output or error."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        return f"Error: {e.stderr}"

def compose_up_detach():
    return run_command(["docker-compose", "up", "--build", "-d"])

def compose_down():
    return run_command(["docker-compose", "down"])

def compose_ps():
    return run_command(["docker-compose", "ps"])

def compose_logs(service=None):
    cmd = ["docker-compose", "logs"]
    if service:
        cmd.append(service)
    return run_command(cmd)

def compose_build(service):
    return run_command(["docker-compose", "build", service])

def compose_up_service(service):
    return run_command(["docker-compose", "up", "-d", service])

def check_health(url):
    try:
        r = requests.get(url, timeout=2)
        if r.status_code == 200:
            return "Healthy"
        else:
            return f"Error: {r.status_code}"
    except Exception as e:
        return f"Error: {e}"

# -------------------- Configuration --------------------
# Define module health endpoints
module_health = {
    "ingestion": "http://localhost:8000/health",
    "perception": "http://localhost:8100/health",
    "integration": "http://localhost:8200/health",
    "routing": "http://localhost:8300/health",
    "specialized": "http://localhost:8400/health",
    "meta": "http://localhost:8301/health",
    "memory": "http://localhost:8401/health",
    "aggregation": "http://localhost:8500/health",
    "feedback": "http://localhost:8600/health",
    "monitoring": "http://localhost:8700/health",
    "graph_rl": "http://localhost:8800/health",
    "comm_optimization": "http://localhost:8900/health",
    "security": "http://localhost:9100/health",
    "learning": "http://localhost:9200/health",
    "reasoning": "http://localhost:9300/health",
    "adaptation": "http://localhost:9400/health",
    "pipeline": "http://localhost:9000/health",
}

# Endpoint for triggering the full pipeline run
aggregator_run_url = "http://localhost:9000/run_pipeline"

# Configuration
API_BASE_URL = "http://localhost:8000"

# -------------------- Streamlit GUI --------------------
st.title("HMAS Monitor")

# Sidebar: Docker Compose Controls
st.sidebar.header("Docker Compose Controls")

if st.sidebar.button("Start All Containers (Detached)"):
    with st.spinner("Starting containers..."):
        output = compose_up_detach()
        st.sidebar.success("Containers started!")
        st.sidebar.text_area("Output", output, height=150)

if st.sidebar.button("Stop All Containers"):
    with st.spinner("Stopping containers..."):
        output = compose_down()
        st.sidebar.success("Containers stopped!")
        st.sidebar.text_area("Output", output, height=150)

if st.sidebar.button("Show Container Status"):
    output = compose_ps()
    st.sidebar.text_area("Status (docker-compose ps)", output, height=150)

# Sidebar: Fetch Logs for a Service
selected_log_service = st.sidebar.text_input("Service Name for Logs (optional)", "")
if st.sidebar.button("Fetch Logs"):
    with st.spinner("Fetching logs..."):
        output = compose_logs(selected_log_service if selected_log_service else None)
        st.sidebar.text_area("Logs Output", output, height=300)

# Sidebar: Rebuild & Restart a Specific Service
st.sidebar.markdown("---")
st.sidebar.header("Rebuild & Restart a Service")
service_options = list(module_health.keys())
selected_service = st.sidebar.selectbox("Select Service", service_options)
if st.sidebar.button("Rebuild & Restart Selected Service"):
    with st.spinner(f"Rebuilding {selected_service}..."):
        build_output = compose_build(selected_service.lower())
        st.sidebar.text_area("Build Output", build_output, height=150)
    with st.spinner(f"Restarting {selected_service}..."):
        up_output = compose_up_service(selected_service.lower())
        st.sidebar.text_area("Restart Output", up_output, height=150)
    st.sidebar.success(f"{selected_service} rebuilt and restarted!")

# Sidebar: Auto-Refresh Health (optional)
if st.sidebar.checkbox("Auto-Refresh Health (every 10 sec)", value=False) and st_autorefresh:
    st_autorefresh(interval=10000, limit=100, key="health_autorefresh")

# Sidebar navigation
st.sidebar.title("HMAS Monitor")
page = st.sidebar.radio(
    "Navigation",
    ["System Overview", "Agents", "Architecture", "Evolution"]
)

def fetch_metrics():
    try:
        response = requests.post(f"{API_BASE_URL}/resources/metrics")
        return response.json()
    except Exception as e:
        st.error(f"Error fetching metrics: {str(e)}")
        return None

def format_bytes(bytes_value):
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_value < 1024:
            return f"{bytes_value:.2f} {unit}"
        bytes_value /= 1024
    return f"{bytes_value:.2f} TB"

if page == "System Overview":
    st.title("System Overview")
    
    # Resource metrics
    col1, col2, col3, col4 = st.columns(4)
    
    metrics = fetch_metrics()
    if metrics:
        with col1:
            st.metric("CPU Usage", f"{metrics['cpu_usage']:.1f}%")
        with col2:
            st.metric("Memory Usage", f"{metrics['memory_usage']:.1f}%")
        with col3:
            st.metric("Disk Usage", f"{metrics['disk_usage']:.1f}%")
        with col4:
            st.metric("Network Usage", format_bytes(metrics['network_usage']))
    
    # Historical metrics chart
    st.subheader("Resource Usage History")
    if 'metrics_history' not in st.session_state:
        st.session_state.metrics_history = []
    
    if metrics:
        st.session_state.metrics_history.append({
            'timestamp': datetime.now(),
            **metrics
        })
        
        # Keep last hour of data
        cutoff = datetime.now() - timedelta(hours=1)
        st.session_state.metrics_history = [
            m for m in st.session_state.metrics_history
            if m['timestamp'] > cutoff
        ]
        
        df = pd.DataFrame(st.session_state.metrics_history)
        fig = px.line(
            df,
            x='timestamp',
            y=['cpu_usage', 'memory_usage', 'disk_usage'],
            labels={'value': 'Usage %', 'variable': 'Metric'},
            title='Resource Usage Trends'
        )
        st.plotly_chart(fig, use_container_width=True)

elif page == "Agents":
    st.title("Agent Management")
    
    # Agent list
    st.subheader("Active Agents")
    try:
        # Placeholder for agent data - replace with actual API call
        agents = [
            {"id": "agent1", "type": "worker", "status": "running", "load": 45},
            {"id": "agent2", "type": "coordinator", "status": "running", "load": 30},
            {"id": "agent3", "type": "specialist", "status": "idle", "load": 10}
        ]
        
        agent_df = pd.DataFrame(agents)
        st.dataframe(agent_df)
        
        # Agent load distribution
        st.subheader("Agent Load Distribution")
        fig = px.bar(
            agent_df,
            x='id',
            y='load',
            color='type',
            title='Agent Load Distribution'
        )
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error fetching agent data: {str(e)}")

elif page == "Architecture":
    st.title("System Architecture")
    
    # Architecture visualization
    st.subheader("Current Architecture")
    try:
        # Placeholder for architecture data - replace with actual API call
        components = [
            {"id": "comp1", "type": "worker", "resources": {"cpu": 1.0, "memory": 512}},
            {"id": "comp2", "type": "coordinator", "resources": {"cpu": 2.0, "memory": 1024}},
            {"id": "comp3", "type": "specialist", "resources": {"cpu": 1.5, "memory": 768}}
        ]
        
        connections = [
            {"source": "comp1", "target": "comp2", "type": "primary"},
            {"source": "comp2", "target": "comp3", "type": "primary"},
            {"source": "comp1", "target": "comp3", "type": "secondary"}
        ]
        
        # Create network graph
        G = nx.Graph()
        for comp in components:
            G.add_node(comp['id'], **comp)
        
        for conn in connections:
            G.add_edge(conn['source'], conn['target'], **conn)
        
        # Convert to plotly figure
        pos = nx.spring_layout(G)
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        node_x = [pos[node][0] for node in G.nodes()]
        node_y = [pos[node][1] for node in G.nodes()]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines'
        ))
        
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=[f"{node}\n{G.nodes[node]['type']}" for node in G.nodes()],
            textposition="bottom center",
            marker=dict(
                size=20,
                color=['#1f77b4' if G.nodes[node]['type'] == 'worker'
                       else '#2ca02c' if G.nodes[node]['type'] == 'coordinator'
                       else '#ff7f0e' for node in G.nodes()]
            )
        ))
        
        fig.update_layout(
            showlegend=False,
            hovermode='closest',
            margin=dict(b=0, l=0, r=0, t=0),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error visualizing architecture: {str(e)}")

elif page == "Evolution":
    st.title("Architecture Evolution")
    
    # Evolution controls
    st.subheader("Evolution Parameters")
    col1, col2 = st.columns(2)
    
    with col1:
        population_size = st.slider("Population Size", 5, 20, 10)
        generations = st.slider("Number of Generations", 1, 10, 5)
    
    with col2:
        fitness_threshold = st.slider("Fitness Threshold", 0.0, 1.0, 0.7)
        mutation_rate = st.slider("Mutation Rate", 0.0, 1.0, 0.1)
    
    if st.button("Start Evolution"):
        try:
            # Placeholder for evolution API call
            constraints = {
                "population_size": population_size,
                "generations": generations,
                "fitness_threshold": fitness_threshold,
                "mutation_rate": mutation_rate
            }
            
            with st.spinner("Evolving architecture..."):
                # Simulate evolution progress
                progress_bar = st.progress(0)
                for i in range(generations):
                    time.sleep(1)  # Simulate computation
                    progress = (i + 1) / generations
                    progress_bar.progress(progress)
                
                st.success("Evolution complete!")
                
                # Display evolution results
                st.subheader("Evolution Results")
                metrics = {
                    "Initial Fitness": 0.65,
                    "Final Fitness": 0.85,
                    "Improvement": "30.7%",
                    "Generation Count": generations,
                    "Best Architecture ID": "arch_123"
                }
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Initial Fitness", f"{metrics['Initial Fitness']:.2f}")
                col2.metric("Final Fitness", f"{metrics['Final Fitness']:.2f}")
                col3.metric("Improvement", metrics['Improvement'])
                
        except Exception as e:
            st.error(f"Error during evolution: {str(e)}")

# Auto-refresh every 5 seconds if on System Overview
if page == "System Overview":
    time.sleep(5)
    st.experimental_rerun()

class TaskState(Enum):
    """Task execution states."""
    READY = auto()
    RUNNING = auto()
    PAUSED = auto()
    COMPLETED = auto()
    FAILED = auto()

@dataclass
class TaskMetrics:
    """Task performance metrics."""
    progress: float = 0.0  # Overall task progress (0-1)
    success_rate: float = 0.0  # Success rate of subtasks
    completion_time: float = 0.0  # Time spent on task
    energy_efficiency: float = 0.0  # Energy efficiency metric
    learning_progress: float = 0.0  # Learning progress metric

@dataclass
class TaskExecutionResult:
    """Result of task execution."""
    success: bool
    completion_percentage: float
    execution_time: float
    error_message: str = None
    metrics: Dict[str, float] = None

class TaskRecording:
    """Class to store task execution recording data."""
    def __init__(self, task: Dict[str, Any], name: str):
        self.name = name
        self.task = task
        self.frames = []
        self.metrics_history = []
        self.perception_history = []
        self.timestamp = datetime.now()

def init_session_state():
    """Initialize session state variables."""
    if 'current_task' not in st.session_state:
        st.session_state.current_task = None
    if 'task_history' not in st.session_state:
        st.session_state.task_history = []
    if 'visualization_config' not in st.session_state:
        st.session_state.visualization_config = {
            'show_sensors': True,
            'show_objectives': True,
            'show_trajectories': False
        }
    if 'perception_client' not in st.session_state:
        st.session_state.perception_client = PerceptionClient()
    if 'perception_running' not in st.session_state:
        st.session_state.perception_running = False
    if 'perception_thread' not in st.session_state:
        st.session_state.perception_thread = None
    if 'task_state' not in st.session_state:
        st.session_state.task_state = TaskState.READY
    if 'task_metrics' not in st.session_state:
        st.session_state.task_metrics = TaskMetrics()
    if 'execution_thread' not in st.session_state:
        st.session_state.execution_thread = None
    if 'task_executor' not in st.session_state:
        st.session_state.task_executor = None
    if 'execution_results' not in st.session_state:
        st.session_state.execution_results = []
    if 'recording_active' not in st.session_state:
        st.session_state.recording_active = False
    if 'current_recording' not in st.session_state:
        st.session_state.current_recording = None
    if 'recordings' not in st.session_state:
        st.session_state.recordings = []
    if 'recording' not in st.session_state:
        st.session_state.recording = False
    if 'replaying' not in st.session_state:
        st.session_state.replaying = False
    if 'replay_playing' not in st.session_state:
        st.session_state.replay_playing = False
    if 'replay_index' not in st.session_state:
        st.session_state.replay_index = 0
    if 'selected_recording' not in st.session_state:
        st.session_state.selected_recording = None

def create_3d_scene(task: Dict[str, Any]) -> go.Figure:
    """Create 3D visualization of task environment."""
    fig = go.Figure()
    
    # Get workspace bounds
    workspace = task["parameters"]["workspace"]
    ws_min, ws_max = workspace[0], workspace[1]
    
    # Add objects to scene
    for obj in task["parameters"]["objects"]:
        position = obj["position"]
        scale = obj["scale"]
        color = obj.get("color", [0.7, 0.7, 0.7])
        
        # Create box/sphere based on object type
        if obj["type"] in ["AGENT", "DYNAMIC", "STATIC", "TOOL"]:
            # Create box
            x = [position[0] - scale[0]/2, position[0] + scale[0]/2]
            y = [position[1] - scale[1]/2, position[1] + scale[1]/2]
            z = [position[2] - scale[2]/2, position[2] + scale[2]/2]
            
            fig.add_trace(go.Mesh3d(
                x=[x[0], x[1], x[1], x[0], x[0], x[1], x[1], x[0]],
                y=[y[0], y[0], y[1], y[1], y[0], y[0], y[1], y[1]],
                z=[z[0], z[0], z[0], z[0], z[1], z[1], z[1], z[1]],
                i=[0, 1, 2, 3, 4, 5, 6, 7],
                j=[1, 2, 3, 0, 5, 6, 7, 4],
                k=[2, 3, 0, 1, 6, 7, 4, 5],
                color=f'rgb({int(color[0]*255)}, {int(color[1]*255)}, {int(color[2]*255)})',
                name=obj["id"]
            ))
            
            # Add sensors if enabled and object is agent
            if (st.session_state.visualization_config['show_sensors'] and 
                obj["type"] == "AGENT" and 
                "sensors" in obj):
                for sensor_name, sensor_spec in obj["sensors"].items():
                    if sensor_name == "range_sensor":
                        # Add range sensor cone
                        range_val = sensor_spec["range"]
                        angle = np.radians(sensor_spec["angle"])
                        radius = range_val * np.tan(angle/2)
                        
                        theta = np.linspace(0, 2*np.pi, 32)
                        r = np.linspace(0, radius, 2)
                        theta_grid, r_grid = np.meshgrid(theta, r)
                        
                        x_sensor = r_grid * np.cos(theta_grid) + position[0]
                        y_sensor = r_grid * np.sin(theta_grid) + position[1]
                        z_sensor = np.ones_like(x_sensor) * position[2]
                        
                        fig.add_trace(go.Surface(
                            x=x_sensor,
                            y=y_sensor,
                            z=z_sensor,
                            opacity=0.2,
                            showscale=False,
                            name=f"{obj['id']}_{sensor_name}"
                        ))
        
        elif obj["type"] == "TARGET":
            # Create target marker
            fig.add_trace(go.Scatter3d(
                x=[position[0]],
                y=[position[1]],
                z=[position[2]],
                mode='markers',
                marker=dict(
                    size=10,
                    symbol='diamond',
                    color=f'rgb({int(color[0]*255)}, {int(color[1]*255)}, {int(color[2]*255)})'
                ),
                name=obj["id"]
            ))
    
    # Add objectives visualization if enabled
    if st.session_state.visualization_config['show_objectives']:
        for obj in task["parameters"].get("objectives", []):
            if "target_position" in obj:
                pos = obj["target_position"]
                fig.add_trace(go.Scatter3d(
                    x=[pos[0]],
                    y=[pos[1]],
                    z=[pos[2]],
                    mode='markers+text',
                    marker=dict(size=8, symbol='x', color='red'),
                    text=[f"{obj['type']}"],
                    name=f"objective_{obj['type']}"
                ))
    
    # Set layout
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[ws_min, ws_max]),
            yaxis=dict(range=[ws_min, ws_max]),
            zaxis=dict(range=[0, ws_max]),
            aspectmode='cube'
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        showlegend=True
    )
    
    return fig

def task_configuration_ui() -> TaskParameters:
    """Create UI for task configuration."""
    st.sidebar.header("Task Configuration")
    
    difficulty = st.sidebar.slider(
        "Difficulty",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.1
    )
    
    workspace_size = st.sidebar.slider(
        "Workspace Size",
        min_value=5.0,
        max_value=20.0,
        value=10.0,
        step=1.0
    )
    
    num_obstacles = st.sidebar.slider(
        "Number of Obstacles",
        min_value=0,
        max_value=10,
        value=5,
        step=1
    )
    
    time_limit = st.sidebar.number_input(
        "Time Limit (s)",
        min_value=0.0,
        value=60.0,
        step=10.0
    )
    
    require_vision = st.sidebar.checkbox("Require Vision", value=True)
    allow_tool_use = st.sidebar.checkbox("Allow Tool Use", value=False)
    cooperative = st.sidebar.checkbox("Enable Cooperation", value=False)
    
    return TaskParameters(
        difficulty=difficulty,
        workspace_size=workspace_size,
        num_obstacles=num_obstacles,
        time_limit=time_limit,
        require_vision=require_vision,
        allow_tool_use=allow_tool_use,
        cooperative=cooperative
    )

def visualization_config_ui():
    """Create UI for visualization configuration."""
    st.sidebar.header("Visualization Settings")
    
    st.session_state.visualization_config['show_sensors'] = st.sidebar.checkbox(
        "Show Sensors",
        value=st.session_state.visualization_config['show_sensors']
    )
    
    st.session_state.visualization_config['show_objectives'] = st.sidebar.checkbox(
        "Show Objectives",
        value=st.session_state.visualization_config['show_objectives']
    )
    
    st.session_state.visualization_config['show_trajectories'] = st.sidebar.checkbox(
        "Show Trajectories",
        value=st.session_state.visualization_config['show_trajectories']
    )

def create_perception_view(task: Dict[str, Any], perception_data: Dict[str, Any]) -> Dict[str, go.Figure]:
    """Create visualization of perception data."""
    figures = {}
    
    # Camera view
    if "camera" in perception_data:
        camera_fig = go.Figure()
        camera_data = np.array(perception_data["camera"]["image"])
        
        camera_fig.add_trace(go.Heatmap(
            z=camera_data,
            colorscale="Viridis",
            showscale=False
        ))
        
        camera_fig.update_layout(
            title="Camera View",
            xaxis_title="Pixel X",
            yaxis_title="Pixel Y",
            margin=dict(l=0, r=0, b=0, t=30),
            height=300
        )
        
        figures["camera"] = camera_fig
    
    # Range sensor visualization
    if "range_sensor" in perception_data:
        range_fig = go.Figure()
        angles = np.array(perception_data["range_sensor"]["angles"])
        ranges = np.array(perception_data["range_sensor"]["ranges"])
        
        # Convert to cartesian coordinates
        x = ranges * np.cos(angles)
        y = ranges * np.sin(angles)
        
        range_fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode="markers",
            marker=dict(
                size=5,
                color=ranges,
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="Range (m)")
            ),
            name="Range Data"
        ))
        
        range_fig.update_layout(
            title="Range Sensor Data",
            xaxis_title="X (m)",
            yaxis_title="Y (m)",
            margin=dict(l=0, r=0, b=0, t=30),
            height=300,
            showlegend=False
        )
        
        figures["range"] = range_fig
    
    # Object detection visualization
    if "detected_objects" in perception_data:
        objects_fig = go.Figure()
        
        for obj in perception_data["detected_objects"]:
            objects_fig.add_trace(go.Scatter(
                x=[obj["position"][0]],
                y=[obj["position"][1]],
                mode="markers+text",
                marker=dict(
                    size=10,
                    symbol="circle",
                    color=obj.get("confidence", 1.0),
                    colorscale="RdYlGn",
                    showscale=True,
                    colorbar=dict(title="Confidence")
                ),
                text=[obj["class"]],
                textposition="top center",
                name=f"{obj['class']} ({obj.get('confidence', 1.0):.2f})"
            ))
        
        objects_fig.update_layout(
            title="Detected Objects",
            xaxis_title="X (m)",
            yaxis_title="Y (m)",
            margin=dict(l=0, r=0, b=0, t=30),
            height=300
        )
        
        figures["objects"] = objects_fig
    
    return figures

def perception_metrics_ui(perception_data: Dict[str, Any]):
    """Display perception metrics."""
    st.sidebar.header("Perception Metrics")
    
    if "metrics" in perception_data:
        metrics = perception_data["metrics"]
        
        # Detection metrics
        st.sidebar.subheader("Detection Performance")
        cols = st.sidebar.columns(2)
        cols[0].metric("Objects Detected", metrics.get("num_objects", 0))
        cols[1].metric("Avg Confidence", f"{metrics.get('avg_confidence', 0):.2f}")
        
        # Processing metrics
        st.sidebar.subheader("Processing Performance")
        cols = st.sidebar.columns(2)
        cols[0].metric("FPS", f"{metrics.get('fps', 0):.1f}")
        cols[1].metric("Latency (ms)", f"{metrics.get('latency', 0):.1f}")
        
        # Memory usage
        st.sidebar.subheader("Resource Usage")
        memory_usage = metrics.get("memory_usage", 0)
        st.sidebar.progress(memory_usage / 100.0)
        st.sidebar.text(f"Memory Usage: {memory_usage}%")

def perception_control_ui():
    """Create UI for perception control."""
    st.sidebar.header("Perception Control")
    
    # Check perception module health
    health = st.session_state.perception_client.check_health()
    status = health.get("status", "unknown")
    
    st.sidebar.text(f"Status: {status}")
    
    if status == "healthy":
        if not st.session_state.perception_running:
            if st.sidebar.button("Start Perception"):
                if st.session_state.perception_client.start_perception():
                    st.session_state.perception_running = True
                    start_perception_thread()
        else:
            if st.sidebar.button("Stop Perception"):
                if st.session_state.perception_client.stop_perception():
                    st.session_state.perception_running = False
                    stop_perception_thread()
    
    # Perception configuration
    with st.sidebar.expander("Perception Config"):
        config = st.session_state.perception_client.get_perception_config()
        
        new_config = {}
        for key, value in config.items():
            if isinstance(value, bool):
                new_config[key] = st.checkbox(key, value=value)
            elif isinstance(value, (int, float)):
                new_config[key] = st.number_input(key, value=value)
            elif isinstance(value, str):
                new_config[key] = st.text_input(key, value=value)
        
        if st.button("Update Config"):
            st.session_state.perception_client.update_perception_config(new_config)

def start_perception_thread():
    """Start background thread for perception updates."""
    if st.session_state.perception_thread is None:
        st.session_state.perception_thread = threading.Thread(
            target=run_perception_loop,
            daemon=True
        )
        st.session_state.perception_thread.start()

def stop_perception_thread():
    """Stop perception update thread."""
    st.session_state.perception_thread = None

def run_perception_loop():
    """Background loop for updating perception data."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    async def update_loop():
        while st.session_state.perception_running and st.session_state.perception_thread:
            # Get sensor data
            sensor_data = await st.session_state.perception_client.get_sensor_data()
            if "camera" in sensor_data:
                st.session_state.perception_data["camera"] = sensor_data["camera"]
            if "range_sensor" in sensor_data:
                st.session_state.perception_data["range_sensor"] = sensor_data["range_sensor"]
            
            # Get object detections
            detections = await st.session_state.perception_client.get_object_detections()
            st.session_state.perception_data["detected_objects"] = detections
            
            # Get metrics
            metrics = await st.session_state.perception_client.get_perception_metrics()
            st.session_state.perception_data["metrics"] = metrics
            
            await asyncio.sleep(0.1)  # Update at 10Hz
    
    loop.run_until_complete(update_loop())
    loop.close()

def task_execution_ui():
    """Create UI for task execution control."""
    st.sidebar.header("Task Execution")
    
    if st.session_state.current_task:
        state = st.session_state.task_state
        
        # Display current state with color coding
        state_colors = {
            TaskState.READY: "blue",
            TaskState.RUNNING: "green",
            TaskState.PAUSED: "orange",
            TaskState.COMPLETED: "green",
            TaskState.FAILED: "red"
        }
        st.sidebar.markdown(
            f"<p style='color: {state_colors[state]}'>Current State: {state.name}</p>",
            unsafe_allow_html=True
        )
        
        # Show error message if failed
        if state == TaskState.FAILED and st.session_state.execution_results:
            last_result = st.session_state.execution_results[-1]
            if last_result.error_message:
                st.sidebar.error(f"Error: {last_result.error_message}")
        
        # Control buttons based on current state
        if state == TaskState.READY:
            if st.sidebar.button("Start Task"):
                st.session_state.task_state = TaskState.RUNNING
                start_execution_thread()
                
        elif state == TaskState.RUNNING:
            col1, col2 = st.sidebar.columns(2)
            if col1.button("Pause"):
                st.session_state.task_state = TaskState.PAUSED
            if col2.button("Stop"):
                st.session_state.task_state = TaskState.READY
                stop_execution_thread()
                
        elif state == TaskState.PAUSED:
            col1, col2 = st.sidebar.columns(2)
            if col1.button("Resume"):
                st.session_state.task_state = TaskState.RUNNING
            if col2.button("Stop"):
                st.session_state.task_state = TaskState.READY
                stop_execution_thread()
        
        elif state in [TaskState.COMPLETED, TaskState.FAILED]:
            if st.sidebar.button("Reset Task"):
                st.session_state.task_state = TaskState.READY
                st.session_state.task_metrics = TaskMetrics()
                st.session_state.execution_results = []
        
        # Display metrics
        metrics = st.session_state.task_metrics
        st.sidebar.progress(metrics.progress, "Task Progress")
        
        cols = st.sidebar.columns(2)
        cols[0].metric("Success Rate", f"{metrics.success_rate*100:.1f}%")
        cols[1].metric("Time", f"{metrics.completion_time:.1f}s")
        
        cols = st.sidebar.columns(2)
        cols[0].metric("Energy", f"{metrics.energy_efficiency*100:.1f}%")
        cols[1].metric("Learning", f"{metrics.learning_progress*100:.1f}%")

def start_execution_thread():
    """Start background thread for task execution."""
    if st.session_state.execution_thread is None:
        st.session_state.execution_thread = threading.Thread(
            target=run_execution_loop,
            daemon=True
        )
        st.session_state.execution_thread.start()

def stop_execution_thread():
    """Stop task execution thread."""
    st.session_state.execution_thread = None
    # Reset metrics
    st.session_state.task_metrics = TaskMetrics()

def run_execution_loop():
    """Background loop for task execution."""
    start_time = time.time()
    
    try:
        # Initialize task executor if not exists
        if st.session_state.task_executor is None:
            st.session_state.task_executor = TaskExecutor(
                task=st.session_state.current_task,
                perception_client=st.session_state.perception_client
            )
        
        while (st.session_state.execution_thread and 
               st.session_state.task_state in [TaskState.RUNNING, TaskState.PAUSED]):
            
            if st.session_state.task_state == TaskState.RUNNING:
                try:
                    # Execute one step of the task
                    result = st.session_state.task_executor.step()
                    
                    # Update metrics
                    metrics = st.session_state.task_metrics
                    metrics.progress = result.completion_percentage
                    metrics.completion_time = time.time() - start_time
                    metrics.success_rate = result.success_rate
                    metrics.energy_efficiency = result.energy_efficiency
                    metrics.learning_progress = result.learning_progress
                    
                    # Store execution results
                    st.session_state.execution_results.append(
                        TaskExecutionResult(
                            success=result.success,
                            completion_percentage=result.completion_percentage,
                            execution_time=metrics.completion_time,
                            metrics={
                                'success_rate': metrics.success_rate,
                                'energy_efficiency': metrics.energy_efficiency,
                                'learning_progress': metrics.learning_progress
                            }
                        )
                    )
                    
                    # Check for task completion
                    if result.completion_percentage >= 1.0:
                        st.session_state.task_state = TaskState.COMPLETED
                        break
                    
                except ExecutionError as e:
                    st.session_state.task_state = TaskState.FAILED
                    st.session_state.execution_results.append(
                        TaskExecutionResult(
                            success=False,
                            completion_percentage=metrics.progress,
                            execution_time=metrics.completion_time,
                            error_message=str(e)
                        )
                    )
                    break
            
            time.sleep(0.1)  # Update at 10Hz
    
    except Exception as e:
        st.session_state.task_state = TaskState.FAILED
        st.session_state.execution_results.append(
            TaskExecutionResult(
                success=False,
                completion_percentage=0.0,
                execution_time=time.time() - start_time,
                error_message=f"Unexpected error: {str(e)}"
            )
        )
    finally:
        # Cleanup
        if st.session_state.task_executor is not None:
            st.session_state.task_executor.cleanup()
            st.session_state.task_executor = None

def record_execution_frame():
    """Record current frame of task execution."""
    if st.session_state.recording and st.session_state.current_recording:
        frame = {
            'task_state': st.session_state.task_state,
            'metrics': copy.deepcopy(st.session_state.task_metrics),
            'perception': copy.deepcopy(st.session_state.perception_data),
            'timestamp': time.time()
        }
        st.session_state.current_recording.frames.append(frame)

def recording_control_ui():
    """Create UI for recording control."""
    st.sidebar.header("Recording Controls")
    
    if st.session_state.current_task:
        if not st.session_state.recording:
            if st.sidebar.button("Start Recording"):
                recording_name = f"Recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                st.session_state.current_recording = TaskRecording(
                    st.session_state.current_task,
                    recording_name
                )
                st.session_state.recording = True
        else:
            if st.sidebar.button("Stop Recording"):
                if st.session_state.current_recording:
                    st.session_state.recordings.append(st.session_state.current_recording)
                    st.session_state.current_recording = None
                st.session_state.recording = False

def replay_control_ui():
    """Create UI for replay control."""
    st.sidebar.header("Replay Controls")
    
    if st.session_state.recordings:
        selected_recording = st.sidebar.selectbox(
            "Select Recording",
            options=[rec.name for rec in st.session_state.recordings],
            index=0
        )
        
        recording = next(rec for rec in st.session_state.recordings if rec.name == selected_recording)
        
        if not st.session_state.replaying:
            if st.sidebar.button("Start Replay"):
                st.session_state.replaying = True
                st.session_state.replay_index = 0
                st.session_state.current_task = recording.task
        else:
            col1, col2, col3 = st.sidebar.columns(3)
            
            if col1.button("⏪"):
                st.session_state.replay_index = max(0, st.session_state.replay_index - 1)
            
            if col2.button("⏸" if st.session_state.replay_playing else "▶"):
                st.session_state.replay_playing = not st.session_state.replay_playing
            
            if col3.button("⏩"):
                st.session_state.replay_index = min(
                    len(recording.frames) - 1,
                    st.session_state.replay_index + 1
                )
            
            # Replay progress
            st.sidebar.progress(
                st.session_state.replay_index / (len(recording.frames) - 1),
                "Replay Progress"
            )
            
            if st.sidebar.button("Stop Replay"):
                st.session_state.replaying = False
                st.session_state.replay_playing = False
                st.session_state.replay_index = 0

def update_replay():
    """Update visualization with replay frame."""
    if st.session_state.replaying and st.session_state.recordings:
        recording = next(
            rec for rec in st.session_state.recordings 
            if rec.name == st.session_state.selected_recording
        )
        
        if st.session_state.replay_playing:
            st.session_state.replay_index = min(
                len(recording.frames) - 1,
                st.session_state.replay_index + 1
            )
            
            if st.session_state.replay_index >= len(recording.frames) - 1:
                st.session_state.replay_playing = False
        
        frame = recording.frames[st.session_state.replay_index]
        st.session_state.task_state = frame['task_state']
        st.session_state.task_metrics = frame['metrics']
        st.session_state.perception_data = frame['perception']

def initialize_session_state():
    """Initialize session state variables."""
    if 'recordings' not in st.session_state:
        st.session_state.recordings = []
    if 'recording' not in st.session_state:
        st.session_state.recording = False
    if 'current_recording' not in st.session_state:
        st.session_state.current_recording = None
    if 'replaying' not in st.session_state:
        st.session_state.replaying = False
    if 'replay_playing' not in st.session_state:
        st.session_state.replay_playing = False
    if 'replay_index' not in st.session_state:
        st.session_state.replay_index = 0
    if 'selected_recording' not in st.session_state:
        st.session_state.selected_recording = None

def main():
    """Main application."""
    initialize_session_state()
    
    st.title("HMAS Task Visualization")
    
    # Create sidebar sections
    task_config_ui()
    visualization_config_ui()
    
    if not st.session_state.replaying:
        recording_control_ui()
    replay_control_ui()
    
    # Update replay if active
    if st.session_state.replaying:
        update_replay()
    
    # Record frame if recording
    if st.session_state.recording:
        record_execution_frame()
    
    # Create visualizations
    if st.session_state.current_task:
        # Create 3D scene
        scene_fig = create_3d_scene(
            st.session_state.current_task,
            st.session_state.visualization_config
        )
        st.plotly_chart(scene_fig, use_container_width=True)
        
        # Create perception visualizations
        if st.session_state.perception_data:
            perception_figs = create_perception_view(
                st.session_state.current_task,
                st.session_state.perception_data
            )
            
            for name, fig in perception_figs.items():
                st.plotly_chart(fig, use_container_width=True)
        
        # Show task execution controls if not replaying
        if not st.session_state.replaying:
            task_execution_ui()
            perception_control_ui()
            perception_metrics_ui(st.session_state.perception_data)
    
    # Task history at bottom
    if st.session_state.task_history:
        st.header("Task History")
        for i, task in enumerate(reversed(st.session_state.task_history[-5:])):
            with st.expander(f"Task {task['id']} ({task['type']})"):
                st.json(task)

if __name__ == "__main__":
    main()
