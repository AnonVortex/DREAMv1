import os
import logging.config
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from slowapi import Limiter
from slowapi.util import get_remote_address
from prometheus_fastapi_instrumentator import Instrumentator
from starlette.middleware.trustedhost import TrustedHostMiddleware
import uvicorn

# Optional: load env
from dotenv import load_dotenv
load_dotenv()

LOGGING_CONFIG = "logging.conf"
if os.path.exists(LOGGING_CONFIG):
    logging.config.fileConfig(LOGGING_CONFIG, disable_existing_loggers=False)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # STARTUP
    logger.info("[Perception] Startup complete.")
    yield
    # SHUTDOWN
    logger.info("[Perception] Shutting down...")

app = FastAPI(title="HMAS Perception Module", version="1.0", lifespan=lifespan)

# Rate Limiting (Optional)
limiter = Limiter(key_func=get_remote_address, default_limits=["5/minute"])
app.state.limiter = limiter

# Prometheus
Instrumentator().instrument(app).expose(app, endpoint="/metrics")

app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])

# Example Data Model for Input
class PerceptionInput(BaseModel):
    image_path: str = None
    audio_path: str = None

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.get("/ready")
async def ready_check():
    # Could test GPU, model loading, or other resources
    # For now, just return ready
    return {"status": "ready"}

@app.post("/perceive")
@limiter.limit("5/minute")
async def perceive(request: Request, input_data: PerceptionInput):
    """
    Stub endpoint that simulates processing an image/audio.
    In a real scenario, you might read from input_data.image_path or input_data.audio_path,
    load a model, run inference, and return embeddings or features.
    """
    logger.info(f"[Perception] Received data: {input_data.dict()}")

    if not input_data.image_path and not input_data.audio_path:
        raise HTTPException(status_code=400, detail="No image or audio path provided")

    # Basic stubs
    vision_features = None
    audio_features = None

    if input_data.image_path:
        # In real code: run YOLO or OpenCV or some vision model
        vision_features = "vision_feature_vector_stub"
    if input_data.audio_path:
        # In real code: run librosa or Whisper to extract audio features
        audio_features = "audio_feature_vector_stub"

    logger.info("[Perception] Features extracted.")
    return {
        "vision_features": vision_features,
        "audio_features": audio_features
    }

if __name__ == "__main__":
    uvicorn.run("perception_main:app", host="0.0.0.0", port=8100, reload=True)
