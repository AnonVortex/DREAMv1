import os
import logging.config
from contextlib import asynccontextmanager
from typing import Any, Dict, List

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

LOGGING_CONFIG_PATH = "logging.conf"
if os.path.exists(LOGGING_CONFIG_PATH):
    logging.config.fileConfig(LOGGING_CONFIG_PATH, disable_existing_loggers=False)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("[Aggregation] Starting up Aggregation Module...")
    yield
    logger.info("[Aggregation] Shutting down Aggregation Module...")

app = FastAPI(
    title="HMAS Aggregation Module",
    version="1.0.0",
    lifespan=lifespan
)

# Setup Redis client for readiness check (if needed)
redis_client = redis.Redis.from_url(settings.REDIS_URL, decode_responses=True)

# Rate limiting
limiter = Limiter(key_func=get_remote_address, default_limits=["10/minute"])
app.state.limiter = limiter

# Prometheus monitoring
Instrumentator().instrument(app).expose(app, endpoint="/metrics")

# Middleware
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])

# Define Pydantic model for Aggregation input
class AggregationInput(BaseModel):
    archive: List[Dict[str, Any]]
    query_result: Dict[str, Any]

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.get("/ready")
async def ready_check():
    try:
        await redis_client.ping()
        return {"status": "ready"}
    except Exception as e:
        logger.warning(f"[Aggregation] Redis not ready: {e}")
        raise HTTPException(status_code=500, detail="Redis not ready")

@app.post("/aggregate")
@limiter.limit("10/minute")
async def aggregate(request: Request, input_data: AggregationInput):
    logger.info(f"[Aggregation] Received memory data: {input_data}")

    if not input_data.archive:
        raise HTTPException(status_code=400, detail="Empty archive")
    
    # For demonstration, simply use the query_result as the final decision.
    final_decision = input_data.query_result

    logger.info(f"[Aggregation] Final Decision: {final_decision}")
    return {"final_decision": final_decision}

if __name__ == "__main__":
    uvicorn.run("aggregation_main:app", host="0.0.0.0", port=8500, reload=True)
