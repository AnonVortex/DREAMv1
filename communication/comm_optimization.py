import os
import logging.config
from contextlib import asynccontextmanager

import random

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

# Import configuration from local config.py
from .config import settings

# Load logging configuration if available
LOGGING_CONFIG_PATH = "logging.conf"
if os.path.exists(LOGGING_CONFIG_PATH):
    logging.config.fileConfig(LOGGING_CONFIG_PATH, disable_existing_loggers=False)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("[CommOptimization] Starting up Communication Optimization Module...")
    yield
    logger.info("[CommOptimization] Shutting down Communication Optimization Module...")

app = FastAPI(
    title="HMAS Communication Optimization Module",
    version="1.0.0",
    lifespan=lifespan
)

# Setup Redis client for readiness check (if needed)
redis_client = redis.Redis.from_url(settings.REDIS_URL, decode_responses=True)

limiter = Limiter(key_func=get_remote_address, default_limits=["10/minute"])
app.state.limiter = limiter

# Prometheus metrics
Instrumentator().instrument(app).expose(app, endpoint="/metrics")

# Middleware
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])

# Pydantic model for input (expandable if needed)
class CommInput(BaseModel):
    # For now, no fields required; this endpoint can be triggered without payload
    pass

def optimize_communication():
    logger.info("[CommOptimization] Optimizing inter-agent communication...")
    metrics = {
        "message_latency": round(random.uniform(0.15, 0.3), 3),
        "message_success_rate": round(random.uniform(0.8, 0.95), 3)
    }
    strategy = random.choice(["broadcast", "unicast", "gossip"])
    logger.info(f"[CommOptimization] Chosen strategy: {strategy}")
    return {"metrics": metrics, "strategy": strategy}

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.get("/ready")
async def ready_check():
    try:
        await redis_client.ping()
        return {"status": "ready"}
    except Exception as e:
        logger.warning(f"[CommOptimization] Redis not ready: {e}")
        raise HTTPException(status_code=500, detail="Redis not ready")

@app.post("/optimize")
@limiter.limit("10/minute")
async def optimize_endpoint(request: Request, input_data: CommInput = None):
    """
    Endpoint to trigger communication optimization.
    Returns the chosen communication strategy and performance metrics.
    """
    result = optimize_communication()
    return result

if __name__ == "__main__":
    uvicorn.run("comm_optimization:app", host="0.0.0.0", port=8900, reload=True)
