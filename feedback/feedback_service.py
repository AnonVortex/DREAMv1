import os
import logging.config
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from datetime import datetime
import numpy as np
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.util import get_remote_address
from prometheus_fastapi_instrumentator import Instrumentator

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Setup logging
LOGGING_CONFIG_PATH = "logging.conf"
if os.path.exists(LOGGING_CONFIG_PATH):
    logging.config.fileConfig(LOGGING_CONFIG_PATH, disable_existing_loggers=False)
logger = logging.getLogger(__name__)

class FeedbackType(str, Enum):
    PERFORMANCE = "performance"
    REWARD = "reward"
    ERROR = "error"
    ADAPTATION = "adaptation"
    USER = "user"

class FeedbackSource(str, Enum):
    AGENT = "agent"
    SYSTEM = "system"
    USER = "user"
    ENVIRONMENT = "environment"

class FeedbackSignal(BaseModel):
    id: str = Field(default_factory=lambda: datetime.now().isoformat())
    type: FeedbackType
    source: FeedbackSource
    target_id: str
    value: float
    metadata: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.now)

class PerformanceMetrics(BaseModel):
    accuracy: float
    latency: float
    resource_usage: float
    reliability: float
    adaptability: float

class FeedbackAnalyzer:
    def __init__(self):
        self.performance_history: Dict[str, List[PerformanceMetrics]] = {}
        self.feedback_weights = {
            "accuracy": 0.3,
            "latency": 0.2,
            "resource_usage": 0.2,
            "reliability": 0.2,
            "adaptability": 0.1
        }
        
    def calculate_reward(self, metrics: PerformanceMetrics) -> float:
        """Calculate reward based on performance metrics."""
        reward = (
            self.feedback_weights["accuracy"] * metrics.accuracy +
            self.feedback_weights["latency"] * (1 - metrics.latency) +  # Lower is better
            self.feedback_weights["resource_usage"] * (1 - metrics.resource_usage) +  # Lower is better
            self.feedback_weights["reliability"] * metrics.reliability +
            self.feedback_weights["adaptability"] * metrics.adaptability
        )
        return max(0.0, min(1.0, reward))
        
    def analyze_trend(self, target_id: str, window_size: int = 10) -> Dict[str, Any]:
        """Analyze performance trends for a target."""
        if target_id not in self.performance_history:
            return {"trend": "unknown", "stability": 0.0, "improvement_rate": 0.0}
            
        history = self.performance_history[target_id][-window_size:]
        if not history:
            return {"trend": "unknown", "stability": 0.0, "improvement_rate": 0.0}
            
        # Calculate trends for each metric
        trends = {}
        for metric in ["accuracy", "latency", "resource_usage", "reliability", "adaptability"]:
            values = [getattr(m, metric) for m in history]
            trend = np.polyfit(range(len(values)), values, 1)[0]
            stability = 1.0 - np.std(values)
            trends[metric] = {
                "trend": "improving" if trend > 0 else "declining",
                "stability": stability,
                "improvement_rate": trend
            }
            
        return trends
        
    def generate_adaptation_signal(
        self,
        target_id: str,
        current_metrics: PerformanceMetrics
    ) -> Optional[FeedbackSignal]:
        """Generate adaptation signal based on performance analysis."""
        trends = self.analyze_trend(target_id)
        
        # Check if adaptation is needed
        adaptation_needed = False
        adaptation_type = None
        
        if current_metrics.latency > 0.8:  # High latency
            adaptation_needed = True
            adaptation_type = "performance"
        elif current_metrics.resource_usage > 0.9:  # High resource usage
            adaptation_needed = True
            adaptation_type = "resource"
        elif current_metrics.accuracy < 0.6:  # Low accuracy
            adaptation_needed = True
            adaptation_type = "accuracy"
            
        if adaptation_needed:
            return FeedbackSignal(
                type=FeedbackType.ADAPTATION,
                source=FeedbackSource.SYSTEM,
                target_id=target_id,
                value=1.0,
                metadata={
                    "adaptation_type": adaptation_type,
                    "current_metrics": current_metrics.dict(),
                    "trends": trends
                }
            )
        return None

class FeedbackManager:
    def __init__(self):
        self.analyzer = FeedbackAnalyzer()
        self.feedback_history: Dict[str, List[FeedbackSignal]] = {}
        
    async def process_feedback(
        self,
        feedback: FeedbackSignal,
        background_tasks: BackgroundTasks
    ) -> Dict[str, Any]:
        """Process incoming feedback and generate appropriate responses."""
        # Store feedback
        if feedback.target_id not in self.feedback_history:
            self.feedback_history[feedback.target_id] = []
        self.feedback_history[feedback.target_id].append(feedback)
        
        response = {
            "status": "success",
            "feedback_id": feedback.id
        }
        
        # Handle different feedback types
        if feedback.type == FeedbackType.PERFORMANCE:
            metrics = PerformanceMetrics(**feedback.metadata["metrics"])
            
            # Update performance history
            if feedback.target_id not in self.analyzer.performance_history:
                self.analyzer.performance_history[feedback.target_id] = []
            self.analyzer.performance_history[feedback.target_id].append(metrics)
            
            # Calculate reward
            reward = self.analyzer.calculate_reward(metrics)
            
            # Generate reward feedback
            reward_feedback = FeedbackSignal(
                type=FeedbackType.REWARD,
                source=FeedbackSource.SYSTEM,
                target_id=feedback.target_id,
                value=reward,
                metadata={"performance_metrics": metrics.dict()}
            )
            self.feedback_history[feedback.target_id].append(reward_feedback)
            
            # Check for adaptation needs
            adaptation_signal = self.analyzer.generate_adaptation_signal(
                feedback.target_id,
                metrics
            )
            if adaptation_signal:
                self.feedback_history[feedback.target_id].append(adaptation_signal)
                response["adaptation_needed"] = True
                response["adaptation_signal"] = adaptation_signal.dict()
                
            response["reward"] = reward
            
        elif feedback.type == FeedbackType.ERROR:
            # Trigger immediate adaptation signal
            adaptation_signal = FeedbackSignal(
                type=FeedbackType.ADAPTATION,
                source=FeedbackSource.SYSTEM,
                target_id=feedback.target_id,
                value=1.0,
                metadata={
                    "error_type": feedback.metadata.get("error_type", "unknown"),
                    "severity": feedback.metadata.get("severity", "high")
                }
            )
            self.feedback_history[feedback.target_id].append(adaptation_signal)
            response["adaptation_needed"] = True
            response["adaptation_signal"] = adaptation_signal.dict()
            
        return response
        
    async def get_feedback_history(
        self,
        target_id: str,
        feedback_type: Optional[FeedbackType] = None,
        limit: int = 100
    ) -> List[FeedbackSignal]:
        """Retrieve feedback history for a target."""
        if target_id not in self.feedback_history:
            return []
            
        history = self.feedback_history[target_id]
        if feedback_type:
            history = [f for f in history if f.type == feedback_type]
            
        return sorted(
            history,
            key=lambda x: x.timestamp,
            reverse=True
        )[:limit]
        
    async def get_performance_analysis(
        self,
        target_id: str
    ) -> Dict[str, Any]:
        """Get comprehensive performance analysis for a target."""
        if target_id not in self.analyzer.performance_history:
            raise HTTPException(
                status_code=404,
                detail=f"No performance history found for target {target_id}"
            )
            
        trends = self.analyzer.analyze_trend(target_id)
        
        # Get recent metrics
        recent_metrics = self.analyzer.performance_history[target_id][-1]
        current_reward = self.analyzer.calculate_reward(recent_metrics)
        
        return {
            "current_metrics": recent_metrics.dict(),
            "current_reward": current_reward,
            "trends": trends,
            "feedback_counts": {
                f_type.value: len([
                    f for f in self.feedback_history.get(target_id, [])
                    if f.type == f_type
                ])
                for f_type in FeedbackType
            }
        }

# Initialize FastAPI app
@asynccontextmanager
async def lifespan(app: FastAPI):
    # STARTUP
    logger.info("Initializing feedback service...")
    try:
        feedback_manager = FeedbackManager()
        app.state.feedback_manager = feedback_manager
        logger.info("Feedback service initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize feedback service: {str(e)}")
        raise
        
    yield
    
    # SHUTDOWN
    logger.info("Shutting down feedback service...")

app = FastAPI(title="HMAS Feedback Service", lifespan=lifespan)

# Rate limiting
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

# Prometheus metrics
Instrumentator().instrument(app).expose(app)

@app.post("/feedback")
@limiter.limit("100/minute")
async def submit_feedback(
    request: Request,
    feedback: FeedbackSignal,
    background_tasks: BackgroundTasks
):
    """Submit feedback for processing."""
    try:
        response = await request.app.state.feedback_manager.process_feedback(
            feedback,
            background_tasks
        )
        return response
    except Exception as e:
        logger.error(f"Error processing feedback: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/feedback/{target_id}")
@limiter.limit("50/minute")
async def get_feedback_history(
    request: Request,
    target_id: str,
    feedback_type: Optional[FeedbackType] = None,
    limit: int = 100
):
    """Get feedback history for a target."""
    try:
        history = await request.app.state.feedback_manager.get_feedback_history(
            target_id,
            feedback_type,
            limit
        )
        return {"feedback_history": [f.dict() for f in history]}
    except Exception as e:
        logger.error(f"Error retrieving feedback history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analysis/{target_id}")
@limiter.limit("30/minute")
async def get_performance_analysis(
    request: Request,
    target_id: str
):
    """Get performance analysis for a target."""
    try:
        analysis = await request.app.state.feedback_manager.get_performance_analysis(
            target_id
        )
        return analysis
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error generating performance analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8400) 