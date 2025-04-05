import os
import logging.config
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Tuple

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
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
    logger.info("[Memory] Starting up Memory Module...")
    # Startup logic: e.g., connect to databases, initialize cache, etc.
    yield
    logger.info("[Memory] Shutting down Memory Module...")

app = FastAPI(
    title="HMAS Memory Module",
    version="1.0.0",
    lifespan=lifespan
)

# Redis client for potential caching (if needed)
redis_client = redis.Redis.from_url(settings.REDIS_URL, decode_responses=True)

# Rate limiting
limiter = Limiter(key_func=get_remote_address, default_limits=["10/minute"])
app.state.limiter = limiter

# Prometheus monitoring
Instrumentator().instrument(app).expose(app, endpoint="/metrics")

# Middleware
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])

# Define Pydantic model for Memory input (Meta output)
class MemoryInput(BaseModel):
    meta_output: Dict[str, Any]

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.get("/ready")
async def ready_check():
    try:
        await redis_client.ping()
        return {"status": "ready"}
    except Exception as e:
        logger.warning(f"[Memory] Redis not ready: {e}")
        raise HTTPException(status_code=500, detail="Redis not ready")

@app.post("/memory")
@limiter.limit("10/minute")
async def archive_memory(request: Request, input_data: MemoryInput):
    logger.info(f"[Memory] Received meta output: {input_data.meta_output}")

    # Archive the meta output (stub implementation)
    archive: List[Dict[str, Any]] = [{"stage": "meta", "evaluation": input_data.meta_output}]
    logger.info(f"[Memory] Archive size: {len(archive)}")
    
    # Retrieve the latest archived record (for demonstration)
    query_result = archive[-1] if archive else None
    logger.info(f"[Memory] Query result: {query_result}")
    
    return {"archive": archive, "query_result": query_result}

if __name__ == "__main__":
    uvicorn.run("memory_main:app", host="0.0.0.0", port=8401, reload=True)
