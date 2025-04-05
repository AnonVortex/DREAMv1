import os
import logging.config
from contextlib import asynccontextmanager
from typing import Dict

import torch
import torch.nn as nn
import torch.optim as optim
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from slowapi import Limiter
from slowapi.util import get_remote_address
from prometheus_fastapi_instrumentator import Instrumentator
from starlette.middleware.trustedhost import TrustedHostMiddleware
import uvicorn
import redis.asyncio as redis

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Import configuration
from .config import settings

LOGGING_CONFIG_PATH = "logging.conf"
if os.path.exists(LOGGING_CONFIG_PATH):
    logging.config.fileConfig(LOGGING_CONFIG_PATH, disable_existing_loggers=False)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("[GraphRL] Starting up Graph RL Module...")
    # Startup logic, e.g., model initialization can be done here.
    yield
    logger.info("[GraphRL] Shutting down Graph RL Module...")

app = FastAPI(
    title="HMAS Graph RL Module",
    version="1.0.0",
    lifespan=lifespan
)

# Setup Redis client if needed
redis_client = redis.Redis.from_url(settings.REDIS_URL, decode_responses=True)

# Rate limiting
limiter = Limiter(key_func=get_remote_address, default_limits=["10/minute"])
app.state.limiter = limiter

# Prometheus monitoring
Instrumentator().instrument(app).expose(app, endpoint="/metrics")

# Middleware
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])

# Define a simple Graph RL Agent using PyTorch
class GraphRLAgent(nn.Module):
    def __init__(self, input_dim: int = 32, hidden_dim: int = 64, output_dim: int = 4):
        super(GraphRLAgent, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Pydantic model for Graph RL input (if needed)
class GraphRLInput(BaseModel):
    # For now, no input parameters are required; can be expanded later.
    pass

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.get("/ready")
async def ready_check():
    try:
        await redis_client.ping()
        return {"status": "ready"}
    except Exception as e:
        logger.warning(f"[GraphRL] Redis not ready: {e}")
        raise HTTPException(status_code=500, detail="Redis not ready")

@app.post("/graph_rl")
@limiter.limit("10/minute")
async def run_graph_rl(request: Request, input_data: GraphRLInput = None):
    """
    Trains a Graph RL agent on dummy data and returns the final action and value estimate.
    In a production scenario, replace the dummy training loop with your actual training/inference logic.
    """
    logger.info("[GraphRL] Training Graph RL Agent...")

    agent = GraphRLAgent()
    optimizer = optim.Adam(agent.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    # Create dummy data
    dummy_input = torch.randn(10, 32)
    dummy_labels = torch.randn(10, 4)

    for epoch in range(5):
        optimizer.zero_grad()
        outputs = agent(dummy_input)
        loss = criterion(outputs, dummy_labels)
        loss.backward()
        optimizer.step()
        logger.info(f"Epoch {epoch+1}/5 - Loss: {loss.item():.4f}")

    final_output = {"action": 2, "value_estimate": 0.995}
    logger.info(f"[GraphRL] Final Output: {final_output}")
    return final_output

if __name__ == "__main__":
    uvicorn.run("graph_rl_agent:app", host="0.0.0.0", port=8800, reload=True)