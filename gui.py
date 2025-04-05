import streamlit as st

# This must be the very first Streamlit command.
st.set_page_config(page_title="HMAS Pipeline Local Manager", layout="wide")

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
    "pipeline": "http://localhost:9000/health",
}

# Endpoint for triggering the full pipeline run
aggregator_run_url = "http://localhost:9000/run_pipeline"

# -------------------- Streamlit GUI --------------------
st.title("HMAS Pipeline Local Manager")

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

# Main Panel: Health Checks
st.header("Module Health Status")
cols = st.columns(2)
health_results = {}
for idx, (module, url) in enumerate(module_health.items()):
    health_results[module] = check_health(url)
    if idx % 2 == 0:
        cols[0].write(f"**{module}:** {health_results[module]}")
    else:
        cols[1].write(f"**{module}:** {health_results[module]}")

# Main Panel: Pipeline Aggregator Control
st.header("Pipeline Aggregator")
if st.button("Run Full Pipeline"):
    with st.spinner("Running pipeline aggregator..."):
        try:
            response = requests.post(aggregator_run_url, timeout=60)
            if response.status_code == 200:
                result = response.json()
                st.success("Pipeline executed successfully!")
                st.json(result)
            else:
                st.error(f"Pipeline run failed with status code {response.status_code}")
        except Exception as e:
            st.error(f"Error executing pipeline: {e}")

# Footer
st.markdown("---")
st.write("Use the sidebar to manage Docker containers and monitor module health. Enjoy your futuristic HMAS dashboard!")
st.markdown("© 2025 HMAS (AGI DREAM)")
