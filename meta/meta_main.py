import os
import logging.config
from contextlib import asynccontextmanager
from typing import Any, Dict

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

# Import configuration
from .config import settings

LOGGING_CONFIG_PATH = "logging.conf"
if os.path.exists(LOGGING_CONFIG_PATH):
    logging.config.fileConfig(LOGGING_CONFIG_PATH, disable_existing_loggers=False)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("[Meta] Starting up Meta Module...")
    # Startup logic can include connecting to databases, pre-loading models, etc.
    yield
    logger.info("[Meta] Shutting down Meta Module...")

app = FastAPI(title="HMAS Meta Module", version="1.0.0", lifespan=lifespan)

# Redis client (if needed for caching or feedback integration)
redis_client = redis.Redis.from_url(settings.REDIS_URL, decode_responses=True)

# Rate limiting
limiter = Limiter(key_func=get_remote_address, default_limits=["10/minute"])
app.state.limiter = limiter

# Prometheus instrumentation
Instrumentator().instrument(app).expose(app, endpoint="/metrics")

# Middleware
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])

# Pydantic model for meta input
class MetaInput(BaseModel):
    specialized_output: Dict[str, Any]

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.get("/ready")
async def ready_check():
    try:
        await redis_client.ping()
        return {"status": "ready"}
    except Exception as e:
        logger.warning(f"[Meta] Redis not ready: {e}")
        raise HTTPException(status_code=500, detail="Redis not ready")

@app.post("/meta")
@limiter.limit("10/minute")
async def evaluate_meta(request: Request, input_data: MetaInput):
    logger.info(f"[Meta] Received specialized output: {input_data.specialized_output}")
    
    # Meta evaluation logic (currently a stub)
    report = {
        "Verification": "Outputs consistent",
        "Consensus": "Majority agreement reached",
        "SelfMonitoring": "Performance within acceptable limits",
        "Iteration": "No further iteration required"
    }
    
    logger.info(f"[Meta] Generated report: {report}")
    return report

if __name__ == "__main__":
    uvicorn.run("meta_main:app", host="0.0.0.0", port=8301, reload=True)
