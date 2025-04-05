import os
import logging.config
from contextlib import asynccontextmanager
from typing import Dict

import psutil  # Ensure psutil is installed
import time

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator
from slowapi import Limiter
from slowapi.util import get_remote_address
from starlette.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel
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
    logger.info("[Monitoring] Starting up Monitoring Module...")
    # Startup logic: e.g., connect to external monitoring services
    yield
    logger.info("[Monitoring] Shutting down Monitoring Module...")

app = FastAPI(
    title="HMAS Monitoring Module",
    version="1.0.0",
    lifespan=lifespan
)

# Redis client for readiness check (if needed)
redis_client = redis.Redis.from_url(settings.REDIS_URL, decode_responses=True)

# Rate limiting
limiter = Limiter(key_func=get_remote_address, default_limits=["10/minute"])
app.state.limiter = limiter

# Prometheus metrics
Instrumentator().instrument(app).expose(app, endpoint="/metrics")

# Middleware
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])

class ResourceUsage(BaseModel):
    cpu_usage: float
    memory_usage: float

def check_resource_usage() -> Dict[str, float]:
    try:
        cpu_usage = psutil.cpu_percent(interval=1)
        mem_usage = psutil.virtual_memory().percent
    except Exception as e:
        logger.warning(f"[Monitoring] psutil error: {e}")
        cpu_usage = 50.0
        mem_usage = 50.0
    return {"cpu_usage": cpu_usage, "memory_usage": mem_usage}

def run_diagnostics() -> str:
    time.sleep(0.5)  # Simulate diagnostic delay
    return "All systems operational."

def scale_if_needed(resource_usage: Dict[str, float]) -> str:
    if resource_usage["cpu_usage"] > 80 or resource_usage["memory_usage"] > 80:
        return "Scaling up required."
    return "No scaling required."

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.get("/ready")
async def ready_check():
    try:
        await redis_client.ping()
        return {"status": "ready"}
    except Exception as e:
        logger.warning(f"[Monitoring] Redis not ready: {e}")
        raise HTTPException(status_code=500, detail="Redis not ready")

@app.get("/monitor")
async def monitor_system():
    usage = check_resource_usage()
    diagnostics = run_diagnostics()
    scaling_decision = scale_if_needed(usage)
    
    result = {
        "resource_usage": usage,
        "diagnostics": diagnostics,
        "scaling_decision": scaling_decision
    }
    
    logger.info(f"[Monitoring] Summary: {result}")
    return result

if __name__ == "__main__":
    uvicorn.run("monitoring_main:app", host="0.0.0.0", port=8700, reload=True)
