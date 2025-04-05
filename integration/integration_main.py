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

# If you have a local config.py in the same folder:
# from .config import settings
# Otherwise, if your config.py is at the root or differently placed, adjust the import path.
from integration.config import settings # Adjust as necessary

# If you want to keep environment loading here, do:
# from dotenv import load_dotenv
# load_dotenv()

LOGGING_CONFIG_PATH = "logging.conf"
if os.path.exists(LOGGING_CONFIG_PATH):
    logging.config.fileConfig(LOGGING_CONFIG_PATH, disable_existing_loggers=False)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context for startup/shutdown, avoiding @app.on_event (deprecated).
    """
    logger.info("[Integration] Starting up Integration Module...")
    # STARTUP LOGIC HERE (e.g. connect to a cache, load a model, etc.)

    yield

    # SHUTDOWN LOGIC
    logger.info("[Integration] Shutting down Integration Module...")

app = FastAPI(
    title="HMAS Integration Module",
    version="1.0.0",
    lifespan=lifespan
)

# --------------------------------------------------------------------------------
# Redis & Rate Limiting
# --------------------------------------------------------------------------------
# Use the async client from redis.asyncio
import redis.asyncio as redis
redis_client = redis.Redis.from_url(settings.REDIS_URL, decode_responses=True)

limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["10/minute"]  # Adjust as needed
)
app.state.limiter = limiter

# --------------------------------------------------------------------------------
# Prometheus Monitoring
# --------------------------------------------------------------------------------
Instrumentator().instrument(app).expose(app, endpoint="/metrics")

# --------------------------------------------------------------------------------
# Middleware
# --------------------------------------------------------------------------------
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])

# --------------------------------------------------------------------------------
# Pydantic Model for Fusion Input
# --------------------------------------------------------------------------------
class IntegrationInput(BaseModel):
    """
    Defines the input shape for integration.
    Expand fields as needed for more features (text_features, etc.).
    """
    vision_features: Optional[str] = None
    audio_features: Optional[str] = None

# --------------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------------
@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.get("/ready")
async def ready_check():
    """
    Readiness check to ensure the module is truly ready.
    Pings Redis asynchronously to confirm connectivity.
    """
    try:
        # Must await the async Redis call to avoid "coroutine never awaited" warnings
        await redis_client.ping()
        return {"status": "ready"}
    except Exception as e:
        logger.warning(f"[Integration] Redis not ready: {e}")
        raise HTTPException(status_code=500, detail="Redis not ready")

@app.post("/integrate")
@limiter.limit("10/minute")
async def integrate(request: Request, input_data: IntegrationInput):
    """
    Fuses multiple feature sets into a unified representation.

    Args:
      input_data (IntegrationInput): Pydantic model with optional vision/audio features.

    Returns:
      JSON with 'fused_features' indicating the integrated result.
    """
    logger.info(f"[Integration] Received input: {input_data}")

    if not (input_data.vision_features or input_data.audio_features):
        raise HTTPException(status_code=400, detail="No features provided")

    # Simple "fusion" example
    fused_str = "Fused("
    if input_data.vision_features:
        fused_str += f"{input_data.vision_features},"
    if input_data.audio_features:
        fused_str += f"{input_data.audio_features},"
    fused_str = fused_str.rstrip(",") + ")"

    logger.info(f"[Integration] Fused result: {fused_str}")

    # Return the fused features
    return {"fused_features": fused_str}

# --------------------------------------------------------------------------------
# Entry Point
# --------------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run("integration_main:app", host="0.0.0.0", port=8200, reload=True)