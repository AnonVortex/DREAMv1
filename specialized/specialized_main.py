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
import redis.asyncio as redis

# Import configuration from local config.py
from .config import settings

# Load logging configuration if available
LOGGING_CONFIG_PATH = "logging.conf"
if os.path.exists(LOGGING_CONFIG_PATH):
    logging.config.fileConfig(LOGGING_CONFIG_PATH, disable_existing_loggers=False)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("[Specialized] Starting up Specialized Module...")
    # (Add any startup logic here, e.g., initializing models)
    yield
    logger.info("[Specialized] Shutting down Specialized Module...")

app = FastAPI(
    title="HMAS Specialized Module",
    version="1.0.0",
    lifespan=lifespan
)

# Rate limiting
limiter = Limiter(key_func=get_remote_address, default_limits=["10/minute"])
app.state.limiter = limiter

# Prometheus monitoring
Instrumentator().instrument(app).expose(app, endpoint="/metrics")

# Middleware
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])

# Optionally, set up Redis for future use if needed for caching, etc.
redis_client = redis.Redis.from_url(settings.REDIS_URL, decode_responses=True)

# Define a Pydantic model for specialized input
class SpecializedInput(BaseModel):
    graph_optimization: Optional[str] = None
    # Add other fields for specialized processing as needed

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.get("/ready")
async def ready_check():
    # If you have external dependencies to check, do it here.
    return {"status": "ready"}

@app.post("/specialize")
@limiter.limit("10/minute")
async def specialize(request: Request, input_data: SpecializedInput):
    logger.info(f"[Specialized] Received input: {input_data}")
    
    # Specialized processing logic:
    if input_data.graph_optimization == "GraphOptimizationAgent":
        try:
            # Stubbed specialized processing logic (e.g., Graph RL optimization)
            specialized_output = {"graph_optimization_action": 1, "graph_optimization_value": 0.98}
        except Exception as e:
            logger.error(f"[Specialized] Processing error: {e}")
            raise HTTPException(status_code=500, detail="Specialized processing error")
    else:
        specialized_output = {"default_specialized_result": True}
    
    logger.info(f"[Specialized] Output: {specialized_output}")
    return specialized_output

if __name__ == "__main__":
    uvicorn.run("specialized_main:app", host="0.0.0.0", port=8400, reload=True)
