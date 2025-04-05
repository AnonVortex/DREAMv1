import os
from dotenv import load_dotenv

load_dotenv()

class PipelineSettings:
    INGESTION_URL = os.getenv("INGESTION_URL", "http://localhost:8000/ingest")
    PERCEPTION_URL = os.getenv("PERCEPTION_URL", "http://localhost:8100/perceive")
    INTEGRATION_URL = os.getenv("INTEGRATION_URL", "http://localhost:8200/integrate")
    ROUTING_URL = os.getenv("ROUTING_URL", "http://localhost:8300/route")
    SPECIALIZED_URL = os.getenv("SPECIALIZED_URL", "http://localhost:8400/specialize")
    META_URL = os.getenv("META_URL", "http://localhost:8301/meta")
    MEMORY_URL = os.getenv("MEMORY_URL", "http://localhost:8401/memory")
    AGGREGATION_URL = os.getenv("AGGREGATION_URL", "http://localhost:8500/aggregate")
    FEEDBACK_URL = os.getenv("FEEDBACK_URL", "http://localhost:8600/feedback")
    MONITORING_URL = os.getenv("MONITORING_URL", "http://localhost:8700/monitor")
    GRAPH_RL_URL = os.getenv("GRAPH_RL_URL", "http://localhost:8800/graph_rl")
    COMM_URL = os.getenv("COMM_URL", "http://localhost:8900/optimize")
    PIPELINE_PORT = int(os.getenv("PIPELINE_PORT", "9000"))

settings = PipelineSettings()
