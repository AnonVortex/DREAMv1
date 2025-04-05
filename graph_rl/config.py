import os
from dotenv import load_dotenv

load_dotenv()

class GraphRLSettings:
    HOST = os.getenv("GRAPH_RL_HOST", "0.0.0.0")
    PORT = int(os.getenv("GRAPH_RL_PORT", "8800"))
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

settings = GraphRLSettings()
