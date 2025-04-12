import os
import logging.config
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Any, Union, Tuple
from enum import Enum
from datetime import datetime
import json
import asyncio
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.datastructures import UploadFile
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.util import get_remote_address
from prometheus_fastapi_instrumentator import Instrumentator
from PIL import Image
import cv2
import librosa

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Setup logging
LOGGING_CONFIG_PATH = "logging.conf"
if os.path.exists(LOGGING_CONFIG_PATH):
    logging.config.fileConfig(LOGGING_CONFIG_PATH, disable_existing_loggers=False)
logger = logging.getLogger(__name__)

class InputType(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    SENSOR = "sensor"
    MULTIMODAL = "multimodal"

class FeatureType(str, Enum):
    EMBEDDINGS = "embeddings"
    VISUAL = "visual"
    ACOUSTIC = "acoustic"
    TEMPORAL = "temporal"
    SPATIAL = "spatial"
    SEMANTIC = "semantic"

class ProcessingStage(str, Enum):
    PREPROCESSING = "preprocessing"
    FEATURE_EXTRACTION = "feature_extraction"
    PATTERN_RECOGNITION = "pattern_recognition"
    FUSION = "fusion"
    POSTPROCESSING = "postprocessing"

class PerceptionConfig(BaseModel):
    input_type: InputType
    feature_types: List[FeatureType]
    processing_stages: List[ProcessingStage]
    model_configs: Optional[Dict[str, Any]] = None
    batch_size: int = 32
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class PerceptionResult(BaseModel):
    input_id: str
    features: Dict[str, Any]
    patterns: Optional[Dict[str, Any]] = None
    confidence: float
    metadata: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.now)

class TextProcessor:
    def __init__(self, model_name: str = "bert-base-uncased"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
    def process(self, text: str) -> Dict[str, Any]:
        """Process text input and extract features."""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # Get embeddings and attention
        embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        attention = outputs.attentions[-1].mean(dim=1).cpu().numpy() if outputs.attentions else None
        
        return {
            "embeddings": embeddings.tolist(),
            "attention": attention.tolist() if attention is not None else None,
            "tokens": self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        }

class ImageProcessor:
    def __init__(self):
        # Load pre-trained CNN
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        
        # Image preprocessing
        self.preprocess = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True).transforms
        
    def process(self, image: Union[str, np.ndarray]) -> Dict[str, Any]:
        """Process image input and extract features."""
        if isinstance(image, str):
            # Load image from path
            img = Image.open(image)
        else:
            img = Image.fromarray(image)
            
        # Preprocess image
        img_tensor = self.preprocess(img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            features = self.model.features(img_tensor)
            
        # Extract visual features
        spatial_features = features.mean(dim=[2, 3]).cpu().numpy()
        
        # Basic image analysis
        img_array = np.array(img)
        color_hist = cv2.calcHist([img_array], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        
        return {
            "visual_features": spatial_features.tolist(),
            "color_histogram": color_hist.flatten().tolist(),
            "image_size": img.size,
            "channels": len(img.getbands())
        }

class AudioProcessor:
    def __init__(self, sample_rate: int = 22050):
        self.sample_rate = sample_rate
        
    def process(self, audio: Union[str, np.ndarray]) -> Dict[str, Any]:
        """Process audio input and extract features."""
        if isinstance(audio, str):
            # Load audio from path
            y, sr = librosa.load(audio, sr=self.sample_rate)
        else:
            y = audio
            sr = self.sample_rate
            
        # Extract various audio features
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        
        return {
            "mfcc": mfcc.tolist(),
            "spectral_centroid": spectral_centroid.tolist(),
            "chroma": chroma.tolist(),
            "duration": len(y) / sr,
            "sample_rate": sr
        }

class PatternRecognizer:
    def __init__(self):
        self.feature_extractors = {
            "spatial": self._extract_spatial_patterns,
            "temporal": self._extract_temporal_patterns,
            "semantic": self._extract_semantic_patterns
        }
        
    def recognize_patterns(
        self,
        features: Dict[str, Any],
        feature_types: List[FeatureType]
    ) -> Dict[str, Any]:
        """Recognize patterns in extracted features."""
        patterns = {}
        
        for feature_type in feature_types:
            if feature_type == FeatureType.SPATIAL:
                patterns["spatial"] = self._extract_spatial_patterns(features)
            elif feature_type == FeatureType.TEMPORAL:
                patterns["temporal"] = self._extract_temporal_patterns(features)
            elif feature_type == FeatureType.SEMANTIC:
                patterns["semantic"] = self._extract_semantic_patterns(features)
                
        return patterns
        
    def _extract_spatial_patterns(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Extract spatial patterns from features."""
        patterns = {}
        
        if "visual_features" in features:
            # Analyze visual patterns
            visual_features = np.array(features["visual_features"])
            patterns["regions_of_interest"] = self._find_regions_of_interest(visual_features)
            patterns["feature_importance"] = self._calculate_feature_importance(visual_features)
            
        return patterns
        
    def _extract_temporal_patterns(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Extract temporal patterns from features."""
        patterns = {}
        
        if "mfcc" in features:
            # Analyze audio patterns
            mfcc = np.array(features["mfcc"])
            patterns["temporal_segments"] = self._segment_temporal_features(mfcc)
            patterns["rhythm_patterns"] = self._analyze_rhythm(mfcc)
            
        return patterns
        
    def _extract_semantic_patterns(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Extract semantic patterns from features."""
        patterns = {}
        
        if "embeddings" in features:
            # Analyze semantic patterns
            embeddings = np.array(features["embeddings"])
            patterns["semantic_clusters"] = self._cluster_embeddings(embeddings)
            patterns["semantic_similarity"] = self._calculate_semantic_similarity(embeddings)
            
        return patterns
        
    def _find_regions_of_interest(self, features: np.ndarray) -> List[Dict[str, Any]]:
        """Find regions of interest in visual features."""
        # Implement region detection logic
        return []
        
    def _calculate_feature_importance(self, features: np.ndarray) -> List[float]:
        """Calculate importance scores for features."""
        return np.abs(features).mean(axis=0).tolist()
        
    def _segment_temporal_features(self, features: np.ndarray) -> List[Dict[str, Any]]:
        """Segment temporal features into meaningful units."""
        # Implement temporal segmentation logic
        return []
        
    def _analyze_rhythm(self, features: np.ndarray) -> Dict[str, Any]:
        """Analyze rhythmic patterns in temporal features."""
        # Implement rhythm analysis logic
        return {}
        
    def _cluster_embeddings(self, embeddings: np.ndarray) -> List[Dict[str, Any]]:
        """Cluster embeddings to find semantic patterns."""
        # Implement clustering logic
        return []
        
    def _calculate_semantic_similarity(self, embeddings: np.ndarray) -> float:
        """Calculate semantic similarity scores."""
        # Implement similarity calculation logic
        return 0.0

class PerceptionManager:
    def __init__(self):
        self.text_processor = TextProcessor()
        self.image_processor = ImageProcessor()
        self.audio_processor = AudioProcessor()
        self.pattern_recognizer = PatternRecognizer()
        self.configs: Dict[str, PerceptionConfig] = {}
        
    def register_config(self, config_id: str, config: PerceptionConfig):
        """Register a perception configuration."""
        self.configs[config_id] = config
        
    async def process_input(
        self,
        config_id: str,
        input_data: Any,
        input_type: InputType,
        background_tasks: BackgroundTasks
    ) -> PerceptionResult:
        """Process input data according to configuration."""
        if config_id not in self.configs:
            raise HTTPException(
                status_code=404,
                detail=f"Configuration {config_id} not found"
            )
            
        config = self.configs[config_id]
        features = {}
        
        try:
            # Extract features based on input type
            if input_type == InputType.TEXT:
                features = self.text_processor.process(input_data)
            elif input_type == InputType.IMAGE:
                features = self.image_processor.process(input_data)
            elif input_type == InputType.AUDIO:
                features = self.audio_processor.process(input_data)
            elif input_type == InputType.MULTIMODAL:
                # Process each modality
                for modality, data in input_data.items():
                    if modality == "text":
                        features["text"] = self.text_processor.process(data)
                    elif modality == "image":
                        features["image"] = self.image_processor.process(data)
                    elif modality == "audio":
                        features["audio"] = self.audio_processor.process(data)
                        
            # Recognize patterns
            patterns = self.pattern_recognizer.recognize_patterns(
                features,
                config.feature_types
            )
            
            # Calculate confidence
            confidence = self._calculate_confidence(features, patterns)
            
            return PerceptionResult(
                input_id=datetime.now().isoformat(),
                features=features,
                patterns=patterns,
                confidence=confidence,
                metadata={
                    "input_type": input_type,
                    "feature_types": [ft.value for ft in config.feature_types],
                    "processing_stages": [ps.value for ps in config.processing_stages]
                }
            )
            
        except Exception as e:
            logger.error(f"Error processing input: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
            
    def _calculate_confidence(
        self,
        features: Dict[str, Any],
        patterns: Dict[str, Any]
    ) -> float:
        """Calculate confidence score for perception results."""
        # Implement confidence calculation logic
        # This is a placeholder implementation
        return 0.8

# Initialize FastAPI app
@asynccontextmanager
async def lifespan(app: FastAPI):
    # STARTUP
    logger.info("Initializing perception service...")
    try:
        perception_manager = PerceptionManager()
        app.state.perception_manager = perception_manager
        logger.info("Perception service initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize perception service: {str(e)}")
        raise
        
    yield
    
    # SHUTDOWN
    logger.info("Shutting down perception service...")

app = FastAPI(title="HMAS Perception Service", lifespan=lifespan)

# Rate limiting
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

# Prometheus metrics
Instrumentator().instrument(app).expose(app)

@app.post("/config/{config_id}")
@limiter.limit("20/minute")
async def register_config(
    request: Request,
    config_id: str,
    config: PerceptionConfig
):
    """Register a perception configuration."""
    try:
        request.app.state.perception_manager.register_config(config_id, config)
        return {"status": "success", "config_id": config_id}
    except Exception as e:
        logger.error(f"Error registering configuration: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/perceive/{config_id}")
@limiter.limit("50/minute")
async def process_input(
    request: Request,
    config_id: str,
    input_type: InputType,
    input_data: Any,
    background_tasks: BackgroundTasks
):
    """Process input data using specified configuration."""
    try:
        result = await request.app.state.perception_manager.process_input(
            config_id,
            input_data,
            input_type,
            background_tasks
        )
        return result.dict()
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error processing input: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/perceive/file/{config_id}")
@limiter.limit("30/minute")
async def process_file(
    request: Request,
    config_id: str,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """Process file input using specified configuration."""
    try:
        # Determine input type from file extension
        ext = file.filename.split(".")[-1].lower()
        input_type = None
        
        if ext in ["txt", "json"]:
            input_type = InputType.TEXT
            content = await file.read()
            input_data = content.decode()
        elif ext in ["jpg", "jpeg", "png"]:
            input_type = InputType.IMAGE
            content = await file.read()
            input_data = np.frombuffer(content, np.uint8)
            input_data = cv2.imdecode(input_data, cv2.IMREAD_COLOR)
        elif ext in ["wav", "mp3"]:
            input_type = InputType.AUDIO
            input_data = await file.read()
            
        if input_type is None:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {ext}"
            )
            
        result = await request.app.state.perception_manager.process_input(
            config_id,
            input_data,
            input_type,
            background_tasks
        )
        return result.dict()
        
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8100) 