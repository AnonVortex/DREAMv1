import os
import logging.config
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any, List, Union, Set
from enum import Enum
from datetime import datetime
import json
import asyncio
import aiohttp
from scipy.spatial.distance import cosine
import torch
import torch.nn as nn
import hashlib
import time

from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from prometheus_fastapi_instrumentator import Instrumentator
from starlette.middleware.trustedhost import TrustedHostMiddleware
from prometheus_client import Counter, Histogram, start_http_server
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

class IntegrationError(Exception):
    """Base exception for integration errors."""
    pass

class FeatureValidationError(IntegrationError):
    """Raised when feature validation fails."""
    pass

class IntegrationInput(BaseModel):
    """Enhanced input model for integration."""
    vision_features: Optional[List[float]] = Field(None, description="Vision feature vector")
    audio_features: Optional[List[float]] = Field(None, description="Audio feature vector")
    text_features: Optional[List[float]] = Field(None, description="Text feature vector")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    weights: Optional[Dict[str, float]] = Field(
        None,
        description="Weights for each modality in fusion"
    )

class IntegrationOutput(BaseModel):
    """Output model for integration results."""
    fused_features: List[float]
    confidence_score: float
    modalities_used: List[str]
    fusion_method: str
    processing_time: float
    cache_hit: bool = False

class MultiModalFusion(nn.Module):
    """Neural network for multi-modal feature fusion."""
    
    def __init__(
        self,
        input_dims: Dict[str, int],
        fusion_dim: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Create projection layers for each modality
        self.projections = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(dim, fusion_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            for name, dim in input_dims.items()
        })
        
        # Attention mechanism for modal fusion
        self.attention = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=8,
            dropout=dropout
        )
        
        # Final fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim)
        )
        
    def forward(
        self,
        features: Dict[str, torch.Tensor],
        weights: Optional[Dict[str, float]] = None
    ) -> torch.Tensor:
        """Forward pass with optional modality weighting."""
        if not features:
            raise ValueError("No features provided")
            
        # Project each modality
        projected = {}
        for name, feat in features.items():
            if feat is not None:
                projected[name] = self.projections[name](feat)
                
        if not projected:
            raise ValueError("No valid features after projection")
            
        # Apply weights if provided
        if weights:
            for name in projected:
                if name in weights:
                    projected[name] = projected[name] * weights[name]
                    
        # Stack for attention
        stacked = torch.stack(list(projected.values()))
        
        # Self-attention fusion
        attended, _ = self.attention(stacked, stacked, stacked)
        
        # Mean pool and final fusion
        fused = self.fusion_layer(torch.mean(attended, dim=0))
        
        return fused

class IntegrationService:
    """Main service for feature integration."""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize fusion model
        self.fusion_model = MultiModalFusion({
            "vision": 2048,  # Adjust based on your feature dimensions
            "audio": 1024,
            "text": 768
        }).to(self.device)
        
        # Cache setup
        self.cache_ttl = int(os.getenv("CACHE_TTL", "3600"))  # 1 hour default
        self.max_cache_size = int(os.getenv("MAX_CACHE_SIZE", "1000"))
        self.cache = {}
        
    def _generate_cache_key(self, features: Dict[str, List[float]], weights: Optional[Dict[str, float]] = None) -> str:
        """Generate a unique cache key for the input features."""
        key_dict = {
            "features": features,
            "weights": weights or {}
        }
        return hashlib.sha256(json.dumps(key_dict, sort_keys=True).encode()).hexdigest()
        
    def _validate_features(self, features: Dict[str, List[float]]) -> None:
        """Validate feature vectors."""
        for modality, feat in features.items():
            if not isinstance(feat, list):
                raise FeatureValidationError(f"Features for {modality} must be a list")
            if not feat:
                raise FeatureValidationError(f"Empty feature vector for {modality}")
            if not all(isinstance(x, (int, float)) for x in feat):
                raise FeatureValidationError(f"Invalid feature values for {modality}")
                
    async def integrate_features(
        self,
        input_data: IntegrationInput,
        background_tasks: BackgroundTasks
    ) -> IntegrationOutput:
        """Integrate multiple feature modalities."""
        start_time = time.time()
        
        # Extract features
        features = {}
        if input_data.vision_features:
            features["vision"] = input_data.vision_features
        if input_data.audio_features:
            features["audio"] = input_data.audio_features
        if input_data.text_features:
            features["text"] = input_data.text_features
            
        if not features:
            raise HTTPException(
                status_code=400,
                detail="No features provided for integration"
            )
            
        try:
            self._validate_features(features)
        except FeatureValidationError as e:
            raise HTTPException(status_code=400, detail=str(e))
            
        # Check cache
        cache_key = self._generate_cache_key(features, input_data.weights)
        cached_result = self.cache.get(cache_key)
        
        if cached_result:
            logger.info("Cache hit for feature integration")
            cached_result["cache_hit"] = True
            cached_result["processing_time"] = time.time() - start_time
            return IntegrationOutput(**cached_result)
            
        # Convert to tensors
        tensor_features = {
            name: torch.tensor(feat, dtype=torch.float32).to(self.device)
            for name, feat in features.items()
        }
        
        # Integrate features
        try:
            with torch.no_grad():
                fused = self.fusion_model(
                    tensor_features,
                    input_data.weights
                )
        except Exception as e:
            logger.error(f"Error during feature fusion: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail="Feature fusion failed"
            )
            
        # Calculate confidence score
        confidence = self._calculate_confidence(
            tensor_features,
            fused
        )
        
        result = {
            "fused_features": fused.cpu().numpy().tolist(),
            "confidence_score": float(confidence),
            "modalities_used": list(features.keys()),
            "fusion_method": "attention_fusion",
            "processing_time": time.time() - start_time,
            "cache_hit": False
        }
        
        # Cache result
        if len(self.cache) >= self.max_cache_size:
            # Remove oldest entry
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        self.cache[cache_key] = result
        
        # Schedule cache cleanup
        background_tasks.add_task(
            self._cleanup_cache_entry,
            cache_key
        )
        
        return IntegrationOutput(**result)
        
    def _calculate_confidence(
        self,
        features: Dict[str, torch.Tensor],
        fused: torch.Tensor
    ) -> float:
        """Calculate confidence score for fusion result."""
        # Calculate cosine similarity between fused and individual features
        similarities = []
        for feat in features.values():
            sim = 1 - cosine(
                fused.cpu().numpy(),
                feat.cpu().numpy()
            )
            similarities.append(sim)
            
        # Confidence is average similarity weighted by feature norms
        weights = [float(torch.norm(feat)) for feat in features.values()]
        total_weight = sum(weights)
        if total_weight == 0:
            return 0.0
            
        confidence = sum(
            sim * (w / total_weight)
            for sim, w in zip(similarities, weights)
        )
        
        return float(confidence)
        
    async def _cleanup_cache_entry(self, key: str):
        """Remove cache entry after TTL."""
        await asyncio.sleep(self.cache_ttl)
        self.cache.pop(key, None)

# --------------------------------------------------------------------------------
# FastAPI Setup
# --------------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context for startup/shutdown."""
    logger.info("Initializing Integration Module...")
    
    # Initialize integration service
    app.state.integration_service = IntegrationService()
    logger.info("Integration service initialized")
    
    yield
    
    # Cleanup
    logger.info("Shutting down Integration Module...")
    app.state.integration_service = None

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
app.add_exception_handler(429, _rate_limit_exceeded_handler)

# --------------------------------------------------------------------------------
# Prometheus Monitoring
# --------------------------------------------------------------------------------
Instrumentator().instrument(app).expose(app, endpoint="/metrics")

# --------------------------------------------------------------------------------
# Middleware
# --------------------------------------------------------------------------------
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])

# --------------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------------
@app.get("/health")
async def health_check():
    """Basic health check."""
    return {"status": "ok"}

@app.get("/ready")
async def ready_check():
    """Readiness check with Redis connectivity test."""
    try:
        await redis_client.ping()
        return {"status": "ready"}
    except Exception as e:
        logger.warning(f"Redis not ready: {e}")
        raise HTTPException(status_code=500, detail="Redis not ready")

@app.post("/integrate", response_model=IntegrationOutput)
@limiter.limit("10/minute")
async def integrate(
    request: Request,
    input_data: IntegrationInput,
    background_tasks: BackgroundTasks
):
    """
    Integrate multiple feature modalities into a unified representation.
    
    Args:
        input_data: Feature vectors from different modalities
        background_tasks: FastAPI background tasks
        
    Returns:
        Integrated features with metadata
    """
    try:
        result = await request.app.state.integration_service.integrate_features(
            input_data,
            background_tasks
        )
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Integration error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Integration failed"
        )

# --------------------------------------------------------------------------------
# Entry Point
# --------------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run("integration_main:app", host="0.0.0.0", port=8200, reload=True)