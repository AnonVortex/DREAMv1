"""Advanced perception module for H-MAS."""

import os
import logging.config
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Any, Union, Tuple
import numpy as np
import torch
import torch.nn as nn
from transformers import (
    AutoFeatureExtractor,
    AutoModelForImageClassification,
    AutoProcessor,
    AutoModelForAudioClassification,
    AutoModelForVideoClassification,
    AutoTokenizer,
    AutoModel
)
from PIL import Image
import librosa
import cv2
from fastapi import FastAPI, HTTPException, Request, File, UploadFile, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.util import get_remote_address
from prometheus_fastapi_instrumentator import Instrumentator
from starlette.middleware.trustedhost import TrustedHostMiddleware
import uvicorn
from torch.cuda.amp import autocast
import gc
from concurrent.futures import ThreadPoolExecutor
import asyncio

# Optional: load env
from dotenv import load_dotenv
load_dotenv()

LOGGING_CONFIG = "logging.conf"
if os.path.exists(LOGGING_CONFIG):
    logging.config.fileConfig(LOGGING_CONFIG, disable_existing_loggers=False)
logger = logging.getLogger(__name__)

# Global thread pool for CPU-bound operations
thread_pool = ThreadPoolExecutor(max_workers=4)

class ModelLoadError(Exception):
    """Raised when a model fails to load."""
    pass

class ModalityConfig:
    """Configuration for different modalities."""
    VISION = {
        "model_name": "google/vit-base-patch16-224",
        "feature_dim": 768,
        "attention_heads": 12
    }
    AUDIO = {
        "model_name": "facebook/wav2vec2-base-960h",
        "feature_dim": 768,
        "sample_rate": 16000
    }
    TEXT = {
        "model_name": "bert-base-uncased",
        "feature_dim": 768,
        "max_length": 512
    }
    VIDEO = {
        "model_name": "MCG-NJU/videomae-base",
        "feature_dim": 768,
        "frames": 16
    }

class PerceptionInput(BaseModel):
    """Input data model for perception."""
    image_path: Optional[str] = Field(None, description="Path to image file")
    audio_path: Optional[str] = Field(None, description="Path to audio file")
    text: Optional[str] = Field(None, description="Text input")
    video_path: Optional[str] = Field(None, description="Path to video file")
    modalities: List[str] = Field(
        default=["vision", "audio", "text", "video"],
        description="List of modalities to process"
    )

class PerceptionOutput(BaseModel):
    """Output data model for perception."""
    features: Dict[str, List[float]]
    attention_weights: Dict[str, List[float]]
    confidence_scores: Dict[str, float]
    metadata: Dict[str, Any]

class MultiModalAttention(nn.Module):
    """Multi-modal attention mechanism."""
    
    def __init__(
        self,
        modality_dims: Dict[str, int],
        num_heads: int = 8
    ):
        super().__init__()
        self.modality_dims = modality_dims
        self.num_heads = num_heads
        
        # Create attention layers for each modality pair
        self.attention_layers = nn.ModuleDict()
        for mod1 in modality_dims:
            for mod2 in modality_dims:
                if mod1 != mod2:
                    key = f"{mod1}_{mod2}"
                    self.attention_layers[key] = nn.MultiheadAttention(
                        embed_dim=modality_dims[mod1],
                        num_heads=num_heads,
                        batch_first=True
                    )
                    
    def forward(
        self,
        features: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through attention mechanism."""
        attended_features = {}
        attention_weights = {}
        
        # Apply cross-modal attention
        for mod1, feat1 in features.items():
            attended = feat1
            for mod2, feat2 in features.items():
                if mod1 != mod2:
                    key = f"{mod1}_{mod2}"
                    attended_tmp, weights = self.attention_layers[key](
                        attended,
                        feat2,
                        feat2
                    )
                    attended = attended + attended_tmp
                    attention_weights[key] = weights
                    
            attended_features[mod1] = attended
            
        return attended_features, attention_weights

class PerceptionModule:
    """Advanced perception module implementation."""
    
    def __init__(self):
        """Initialize perception module."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models = {}
        self.processors = {}
        
        try:
            self._load_models()
        except Exception as e:
            logger.error(f"Failed to load models: {str(e)}")
            raise ModelLoadError(f"Failed to initialize perception module: {str(e)}")
            
        # Initialize attention mechanism
        self.attention = MultiModalAttention(
            modality_dims={
                "vision": ModalityConfig.VISION["feature_dim"],
                "audio": ModalityConfig.AUDIO["feature_dim"],
                "text": ModalityConfig.TEXT["feature_dim"],
                "video": ModalityConfig.VIDEO["feature_dim"]
            }
        ).to(self.device)
        
        # Batch processing settings
        self.batch_size = int(os.getenv("BATCH_SIZE", "32"))
        self.max_queue_size = int(os.getenv("MAX_QUEUE_SIZE", "100"))
        self.processing_queue = asyncio.Queue(maxsize=self.max_queue_size)
        
    def _load_models(self):
        """Load all models with proper error handling."""
        try:
            # Vision model
            self.models["vision"] = AutoModelForImageClassification.from_pretrained(
                ModalityConfig.VISION["model_name"]
            ).to(self.device)
            self.processors["vision"] = AutoFeatureExtractor.from_pretrained(
                ModalityConfig.VISION["model_name"]
            )
            logger.info("Vision model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load vision model: {str(e)}")
            raise
            
        try:
            # Audio model
            self.models["audio"] = AutoModelForAudioClassification.from_pretrained(
                ModalityConfig.AUDIO["model_name"]
            ).to(self.device)
            self.processors["audio"] = AutoProcessor.from_pretrained(
                ModalityConfig.AUDIO["model_name"]
            )
            logger.info("Audio model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load audio model: {str(e)}")
            raise
            
        try:
            # Text model
            self.models["text"] = AutoModel.from_pretrained(
                ModalityConfig.TEXT["model_name"]
            ).to(self.device)
            self.processors["text"] = AutoTokenizer.from_pretrained(
                ModalityConfig.TEXT["model_name"]
            )
            logger.info("Text model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load text model: {str(e)}")
            raise
            
        try:
            # Video model
            self.models["video"] = AutoModelForVideoClassification.from_pretrained(
                ModalityConfig.VIDEO["model_name"]
            ).to(self.device)
            self.processors["video"] = AutoFeatureExtractor.from_pretrained(
                ModalityConfig.VIDEO["model_name"]
            )
            logger.info("Video model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load video model: {str(e)}")
            raise
            
    async def cleanup(self):
        """Cleanup resources."""
        logger.info("Cleaning up perception module resources...")
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # Clear models
        for model in self.models.values():
            del model
        self.models.clear()
        
        # Clear processors
        self.processors.clear()
        
        # Clear attention mechanism
        del self.attention
        
        # Force garbage collection
        gc.collect()
        
        logger.info("Perception module cleanup completed")
        
    async def process_batch(
        self,
        batch: List[PerceptionInput]
    ) -> List[PerceptionOutput]:
        """Process a batch of inputs."""
        results = []
        
        # Group inputs by modality
        vision_inputs = []
        audio_inputs = []
        text_inputs = []
        video_inputs = []
        
        for input_data in batch:
            if "vision" in input_data.modalities and input_data.image_path:
                vision_inputs.append(input_data.image_path)
            if "audio" in input_data.modalities and input_data.audio_path:
                audio_inputs.append(input_data.audio_path)
            if "text" in input_data.modalities and input_data.text:
                text_inputs.append(input_data.text)
            if "video" in input_data.modalities and input_data.video_path:
                video_inputs.append(input_data.video_path)
                
        # Process each modality in parallel
        feature_futures = []
        
        if vision_inputs:
            feature_futures.append(
                asyncio.create_task(
                    self._extract_vision_features(vision_inputs)
                )
            )
            
        if audio_inputs:
            feature_futures.append(
                asyncio.create_task(
                    self._extract_audio_features(audio_inputs)
                )
            )
            
        if text_inputs:
            feature_futures.append(
                asyncio.create_task(
                    self._extract_text_features(text_inputs)
                )
            )
            
        if video_inputs:
            feature_futures.append(
                asyncio.create_task(
                    self._extract_video_features(video_inputs)
                )
            )
            
        # Wait for all features
        feature_results = await asyncio.gather(*feature_futures)
        
        # Combine results
        for i, input_data in enumerate(batch):
            features = {}
            attention_weights = {}
            confidence_scores = {}
            
            for modality_features in feature_results:
                if i < len(modality_features):
                    mod_name = modality_features[i]["modality"]
                    features[mod_name] = modality_features[i]["features"]
                    confidence_scores[mod_name] = modality_features[i]["confidence"]
                    
            # Apply cross-modal attention if multiple modalities
            if len(features) > 1:
                tensor_features = {
                    mod: torch.tensor(feat).to(self.device)
                    for mod, feat in features.items()
                }
                
                with torch.no_grad():
                    attended_features, weights = self.attention(tensor_features)
                    
                # Update features with attended versions
                features = {
                    mod: feat.cpu().numpy().tolist()
                    for mod, feat in attended_features.items()
                }
                attention_weights = {
                    key: weight.cpu().numpy().tolist()
                    for key, weight in weights.items()
                }
                
            results.append(
                PerceptionOutput(
                    features=features,
                    attention_weights=attention_weights,
                    confidence_scores=confidence_scores,
                    metadata={
                        "device": str(self.device),
                        "batch_size": len(batch),
                        "modalities_processed": list(features.keys())
                    }
                )
            )
            
        return results
        
    async def _extract_vision_features(
        self,
        image_paths: List[str]
    ) -> List[Dict[str, Any]]:
        """Extract features from images."""
        results = []
        
        def process_image(path: str) -> np.ndarray:
            image = Image.open(path).convert("RGB")
            inputs = self.processors["vision"](
                image,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad(), autocast():
                outputs = self.models["vision"](**inputs)
                features = outputs.logits.mean(dim=1)
                
            return {
                "modality": "vision",
                "features": features.cpu().numpy().tolist(),
                "confidence": float(torch.sigmoid(outputs.logits).max().item())
            }
            
        # Process images in thread pool
        loop = asyncio.get_event_loop()
        futures = [
            loop.run_in_executor(thread_pool, process_image, path)
            for path in image_paths
        ]
        results = await asyncio.gather(*futures)
        
        return results
        
    async def _extract_audio_features(
        self,
        audio_paths: List[str]
    ) -> List[Dict[str, Any]]:
        """Extract features from audio files."""
        results = []
        
        def process_audio(path: str) -> np.ndarray:
            # Load and preprocess audio
            audio, sr = librosa.load(
                path,
                sr=ModalityConfig.AUDIO["sample_rate"]
            )
            inputs = self.processors["audio"](
                audio,
                sampling_rate=sr,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad(), autocast():
                outputs = self.models["audio"](**inputs)
                features = outputs.logits.mean(dim=1)
                
            return {
                "modality": "audio",
                "features": features.cpu().numpy().tolist(),
                "confidence": float(torch.sigmoid(outputs.logits).max().item())
            }
            
        # Process audio in thread pool
        loop = asyncio.get_event_loop()
        futures = [
            loop.run_in_executor(thread_pool, process_audio, path)
            for path in audio_paths
        ]
        results = await asyncio.gather(*futures)
        
        return results
        
    async def _extract_text_features(
        self,
        texts: List[str]
    ) -> List[Dict[str, Any]]:
        """Extract features from text."""
        results = []
        
        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            
            # Tokenize
            inputs = self.processors["text"](
                batch_texts,
                padding=True,
                truncation=True,
                max_length=ModalityConfig.TEXT["max_length"],
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad(), autocast():
                outputs = self.models["text"](**inputs)
                features = outputs.last_hidden_state.mean(dim=1)
                
            # Calculate confidence using attention weights
            attention_weights = outputs.attentions[-1].mean(dim=1)
            confidence_scores = attention_weights.max(dim=-1).values.mean(dim=-1)
            
            for j, (feat, conf) in enumerate(zip(features, confidence_scores)):
                results.append({
                    "modality": "text",
                    "features": feat.cpu().numpy().tolist(),
                    "confidence": float(conf.item())
                })
                
        return results
        
    async def _extract_video_features(
        self,
        video_paths: List[str]
    ) -> List[Dict[str, Any]]:
        """Extract features from videos."""
        results = []
        
        def process_video(path: str) -> np.ndarray:
            # Load video frames
            cap = cv2.VideoCapture(path)
            frames = []
            while len(frames) < ModalityConfig.VIDEO["frames"]:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            cap.release()
            
            # Pad if necessary
            while len(frames) < ModalityConfig.VIDEO["frames"]:
                frames.append(frames[-1])
                
            # Process frames
            inputs = self.processors["video"](
                frames,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad(), autocast():
                outputs = self.models["video"](**inputs)
                features = outputs.logits.mean(dim=1)
                
            return {
                "modality": "video",
                "features": features.cpu().numpy().tolist(),
                "confidence": float(torch.sigmoid(outputs.logits).max().item())
            }
            
        # Process videos in thread pool
        loop = asyncio.get_event_loop()
        futures = [
            loop.run_in_executor(thread_pool, process_video, path)
            for path in video_paths
        ]
        results = await asyncio.gather(*futures)
        
        return results

# Initialize FastAPI app
@asynccontextmanager
async def lifespan(app: FastAPI):
    # STARTUP
    logger.info("Initializing perception module...")
    try:
        perception_module = PerceptionModule()
        app.state.perception_module = perception_module
        logger.info("Perception module initialized successfully")
    except ModelLoadError as e:
        logger.error(f"Failed to initialize perception module: {str(e)}")
        raise
        
    yield
    
    # SHUTDOWN
    logger.info("Shutting down perception module...")
    try:
        await app.state.perception_module.cleanup()
        logger.info("Perception module shutdown complete")
    except Exception as e:
        logger.error(f"Error during perception module shutdown: {str(e)}")

app = FastAPI(title="HMAS Perception Module", version="1.0", lifespan=lifespan)

# Rate Limiting
limiter = Limiter(key_func=get_remote_address, default_limits=["5/minute"])
app.state.limiter = limiter

# Prometheus
Instrumentator().instrument(app).expose(app, endpoint="/metrics")

app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.get("/ready")
async def ready_check():
    """Check if models are loaded and GPU is available."""
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return {
            "status": "ready",
            "device": str(device),
            "gpu_available": torch.cuda.is_available()
        }
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Service not ready: {str(e)}"
        )

@app.post("/perceive", response_model=PerceptionOutput)
@limiter.limit("5/minute")
async def perceive(request: Request, input_data: PerceptionInput):
    """Extract features from multi-modal input."""
    try:
        perception_module = request.app.state.perception_module
        output = await perception_module.extract_features(input_data)
        return output
    except Exception as e:
        logger.error(f"[Perception] Error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Perception failed: {str(e)}"
        )

if __name__ == "__main__":
    uvicorn.run("perception_main:app", host="0.0.0.0", port=8100, reload=True)
