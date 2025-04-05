import os
import logging.config
from contextlib import asynccontextmanager
from typing import Any, Dict

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
    logger.info("[Feedback] Starting up Feedback Module...")
    # Startup logic: connect to services, initialize models, etc.
    yield
    logger.info("[Feedback] Shutting down Feedback Module...")

app = FastAPI(
    title="HMAS Feedback Module",
    version="1.0.0",
    lifespan=lifespan
)

# Setup Redis client for readiness check if needed
redis_client = redis.Redis.from_url(settings.REDIS_URL, decode_responses=True)

limiter = Limiter(key_func=get_remote_address, default_limits=["10/minute"])
app.state.limiter = limiter

Instrumentator().instrument(app).expose(app, endpoint="/metrics")
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])

# Pydantic model for feedback input
class FeedbackInput(BaseModel):
    final_decision: str

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.get("/ready")
async def ready_check():
    try:
        await redis_client.ping()
        return {"status": "ready"}
    except Exception as e:
        logger.warning(f"[Feedback] Redis not ready: {e}")
        raise HTTPException(status_code=500, detail="Redis not ready")

@app.post("/feedback")
@limiter.limit("10/minute")
async def get_feedback(request: Request, input_data: FeedbackInput):
    logger.info(f"[Feedback] Received final decision: {input_data.final_decision}")
    
    # Feedback logic: stubbed for now
    metrics = {
        "accuracy": round(0.85 + 0.1 * 0.5, 3),  # example: random-ish fixed value
        "latency": round(0.2 + 0.3 * 0.5, 3),
        "error_rate": round(0.01 + 0.04 * 0.5, 3)
    }
    updated_params = {
        "learning_rate": round(0.0009 + 0.0001 * 0.5, 6),
        "batch_size": 24,  # example value
        "update_frequency": 3  # example value
    }
    summary = {
        "feedback": metrics,
        "updated_params": updated_params
    }
    
    logger.info(f"[Feedback] Summary: {summary}")
    return summary

if __name__ == "__main__":
    uvicorn.run("feedback_main:app", host="0.0.0.0", port=8600, reload=True)
