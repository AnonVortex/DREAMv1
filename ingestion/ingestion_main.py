import os
import logging.config
from contextlib import asynccontextmanager

import aiofiles
import redis.asyncio as redis
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.util import get_remote_address
from prometheus_fastapi_instrumentator import Instrumentator
from starlette.middleware.trustedhost import TrustedHostMiddleware
import uvicorn

# Optional: load env
from dotenv import load_dotenv
load_dotenv()

# --------------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------------
LOGGING_CONFIG = "logging.conf"
if os.path.exists(LOGGING_CONFIG):
    logging.config.fileConfig(LOGGING_CONFIG, disable_existing_loggers=False)
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------------
# Lifespan for Startup/Shutdown
# --------------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # STARTUP
    logger.info("Ingestion Module: startup complete.")
    yield
    # SHUTDOWN
    logger.info("Ingestion Module: shutting down...")

# --------------------------------------------------------------------------------
# Create FastAPI App
# --------------------------------------------------------------------------------
app = FastAPI(title="Ingestion Module", version="1.0", lifespan=lifespan)

# --------------------------------------------------------------------------------
# Redis & Rate Limiting
# --------------------------------------------------------------------------------
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)

limiter = Limiter(key_func=get_remote_address, default_limits=["5/minute"])
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
# Allowed Extensions
# --------------------------------------------------------------------------------
ALLOWED_EXTENSIONS = {".txt", ".csv", ".json", ".mp4", ".jpg", ".png", ".wav", ".mp3"}

# --------------------------------------------------------------------------------
# Health & Ready Endpoints
# --------------------------------------------------------------------------------
@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.get("/ready")
async def ready_check():
    try:
        await redis_client.ping()
        return {"status": "ready"}
    except:
        raise HTTPException(status_code=500, detail="Redis not ready")

# --------------------------------------------------------------------------------
# Ingestion Endpoint (Chunk-based)
# --------------------------------------------------------------------------------
@app.post("/ingest")
@limiter.limit("5/minute")
async def ingest_file(request: Request, file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename)[-1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")

    file_path = os.path.join("data", file.filename)
    os.makedirs("data", exist_ok=True)

    # chunk-based writing, 1MB
    async with aiofiles.open(file_path, "wb") as out_file:
        while chunk := await file.read(1_000_000):
            await out_file.write(chunk)

    logger.info(f"Ingested file {file.filename} to {file_path}")
    return {"message": "File successfully ingested", "filename": file.filename}

if __name__ == "__main__":
    uvicorn.run("ingestion_main:app", host="0.0.0.0", port=8000, reload=True)
