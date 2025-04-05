import os
import logging.config
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from slowapi import Limiter
from slowapi.util import get_remote_address
from prometheus_fastapi_instrumentator import Instrumentator
from starlette.middleware.trustedhost import TrustedHostMiddleware
import uvicorn

# Import configuration from local config.py
from .config import settings

# Load logging configuration if available
LOGGING_CONFIG_PATH = "logging.conf"
if os.path.exists(LOGGING_CONFIG_PATH):
    logging.config.fileConfig(LOGGING_CONFIG_PATH, disable_existing_loggers=False)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("[Routing] Starting up Routing Module...")
    yield
    logger.info("[Routing] Shutting down Routing Module...")

app = FastAPI(
    title="HMAS Routing Module",
    version="1.0.0",
    lifespan=lifespan
)

# Rate limiting
limiter = Limiter(key_func=get_remote_address, default_limits=["10/minute"])
app.state.limiter = limiter

# Prometheus monitoring
Instrumentator().instrument(app).expose(app, endpoint="/metrics")

# Middleware to trust any host (adjust as needed)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])

# Define Pydantic model for integration input
class RoutingInput(BaseModel):
    fused_features: str

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.get("/ready")
async def ready_check():
    # No external dependencies to check here; could add if needed
    return {"status": "ready"}

@app.post("/route")
@limiter.limit("10/minute")
async def route_task(request: Request, input_data: RoutingInput):
    logger.info(f"[Routing] Received input: {input_data}")
    
    # Simple decision logic based on presence of "vision" in fused_features.
    if "vision" in input_data.fused_features.lower():
        routing_decision = {"agent": "VisionOptimizationAgent"}
    else:
        routing_decision = {"agent": "DefaultRoutingAgent"}
    
    logger.info(f"[Routing] Routing decision: {routing_decision}")
    return routing_decision

if __name__ == "__main__":
    uvicorn.run("routing_main:app", host="0.0.0.0", port=8300, reload=True)
