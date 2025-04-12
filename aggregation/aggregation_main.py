import os
import logging.config
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import numpy as np
from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import json
import asyncio

from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from pydantic import BaseModel, Field
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

class EvaluationResult(BaseModel):
    """Model for evaluation results."""
    stage: str
    evaluation: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.now)
    confidence: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

class AggregationInput(BaseModel):
    """Enhanced input model with metadata."""
    archive: List[Dict[str, Any]]
    query_result: Dict[str, Any]
    context: Optional[Dict[str, Any]] = None
    optimization_criteria: Optional[Dict[str, float]] = Field(
        None,
        description="Criteria for decision optimization with weights"
    )

class AggregationResult(BaseModel):
    """Enhanced output model with confidence and explanations."""
    final_decision: Dict[str, Any]
    confidence_score: float
    supporting_evidence: List[Dict[str, Any]]
    trends: Dict[str, Any]
    anomalies: List[Dict[str, Any]]
    optimization_score: float
    timestamp: datetime = Field(default_factory=datetime.now)

class AnomalyDetector:
    """Detects anomalies in evaluation metrics."""
    
    def __init__(self, sensitivity: float = 2.0):
        self.sensitivity = sensitivity
        self.scaler = StandardScaler()
        
    def detect_anomalies(
        self,
        metrics: List[Dict[str, float]],
        window_size: int = 10
    ) -> List[Dict[str, Any]]:
        """Detect anomalies using statistical and clustering methods."""
        if not metrics or len(metrics) < window_size:
            return []
            
        anomalies = []
        recent_metrics = metrics[-window_size:]
        
        # Convert to feature matrix
        feature_names = list(recent_metrics[0].keys())
        feature_matrix = np.array([
            [m[f] for f in feature_names]
            for m in recent_metrics
        ])
        
        # Z-score based detection
        z_scores = zscore(feature_matrix, axis=0)
        for i, metric_values in enumerate(z_scores):
            for j, z_value in enumerate(metric_values):
                if abs(z_value) > self.sensitivity:
                    anomalies.append({
                        "metric": feature_names[j],
                        "timestamp": metrics[-window_size + i]["timestamp"],
                        "value": metrics[-window_size + i][feature_names[j]],
                        "z_score": float(z_value),
                        "type": "statistical"
                    })
                    
        # Clustering based detection
        scaled_features = self.scaler.fit_transform(feature_matrix)
        clusterer = DBSCAN(eps=0.5, min_samples=3)
        labels = clusterer.fit_predict(scaled_features)
        
        # Points labeled as -1 are considered anomalies
        for i, label in enumerate(labels):
            if label == -1:
                anomalies.append({
                    "timestamp": metrics[-window_size + i]["timestamp"],
                    "metrics": {
                        name: metrics[-window_size + i][name]
                        for name in feature_names
                    },
                    "type": "clustering"
                })
                
        return anomalies

class DecisionOptimizer:
    """Optimizes decisions based on multiple criteria."""
    
    def __init__(self):
        self.default_weights = {
            "confidence": 0.4,
            "consistency": 0.3,
            "performance": 0.3
        }
        
    def optimize_decision(
        self,
        candidates: List[Dict[str, Any]],
        criteria: Optional[Dict[str, float]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, Any], float]:
        """Optimize decision based on weighted criteria."""
        if not candidates:
            raise ValueError("No decision candidates provided")
            
        weights = criteria if criteria else self.default_weights
        
        # Normalize weights
        total_weight = sum(weights.values())
        normalized_weights = {
            k: v / total_weight for k, v in weights.items()
        }
        
        # Score candidates
        scores = []
        for candidate in candidates:
            score = 0.0
            
            # Confidence score
            if "confidence" in normalized_weights:
                score += normalized_weights["confidence"] * candidate.get(
                    "confidence",
                    0.5
                )
                
            # Consistency score
            if "consistency" in normalized_weights and context:
                consistency = self._evaluate_consistency(
                    candidate,
                    context.get("historical_decisions", [])
                )
                score += normalized_weights["consistency"] * consistency
                
            # Performance score
            if "performance" in normalized_weights:
                performance = self._evaluate_performance(candidate)
                score += normalized_weights["performance"] * performance
                
            scores.append((score, candidate))
            
        # Select best candidate
        best_score, best_candidate = max(scores, key=lambda x: x[0])
        return best_candidate, best_score
        
    def _evaluate_consistency(
        self,
        candidate: Dict[str, Any],
        history: List[Dict[str, Any]]
    ) -> float:
        """Evaluate decision consistency with historical decisions."""
        if not history:
            return 1.0
            
        similarities = []
        for hist_decision in history[-5:]:  # Consider last 5 decisions
            sim = self._calculate_decision_similarity(
                candidate,
                hist_decision
            )
            similarities.append(sim)
            
        return np.mean(similarities) if similarities else 1.0
        
    def _evaluate_performance(self, candidate: Dict[str, Any]) -> float:
        """Evaluate expected performance impact of decision."""
        # Implement performance evaluation logic
        # For now, return a default score
        return 0.8
        
    def _calculate_decision_similarity(
        self,
        decision1: Dict[str, Any],
        decision2: Dict[str, Any]
    ) -> float:
        """Calculate similarity between two decisions."""
        try:
            # Extract comparable attributes
            attrs1 = self._extract_comparable_attributes(decision1)
            attrs2 = self._extract_comparable_attributes(decision2)
            
            # Calculate Jaccard similarity for sets of attributes
            intersection = len(set(attrs1) & set(attrs2))
            union = len(set(attrs1) | set(attrs2))
            
            return intersection / union if union > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating decision similarity: {str(e)}")
            return 0.0
            
    def _extract_comparable_attributes(
        self,
        decision: Dict[str, Any]
    ) -> List[str]:
        """Extract attributes for comparison."""
        attributes = []
        
        def extract_recursive(obj: Any, prefix: str = ""):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    new_prefix = f"{prefix}.{k}" if prefix else k
                    if isinstance(v, (str, int, float, bool)):
                        attributes.append(f"{new_prefix}:{str(v)}")
                    else:
                        extract_recursive(v, new_prefix)
            elif isinstance(obj, list):
                for i, v in enumerate(obj):
                    extract_recursive(v, f"{prefix}[{i}]")
                    
        extract_recursive(decision)
        return attributes

class AggregationEngine:
    """Handles sophisticated aggregation logic."""
    
    def __init__(self):
        self.recent_decisions = []
        self.confidence_threshold = 0.7
        self.max_history = 100
        self.anomaly_detector = AnomalyDetector()
        self.decision_optimizer = DecisionOptimizer()
        
    def calculate_confidence(
        self,
        evaluation: Dict[str, Any],
        archive: List[Dict[str, Any]]
    ) -> float:
        """Calculate confidence score based on historical consistency and current evaluation."""
        try:
            # Base confidence from current evaluation
            base_confidence = evaluation.get("confidence", 0.5)
            
            # Historical consistency
            if archive:
                recent_evaluations = [
                    entry["evaluation"] 
                    for entry in archive[-10:]  # Look at last 10 entries
                    if entry["stage"] == evaluation["stage"]
                ]
                
                if recent_evaluations:
                    # Calculate similarity with recent evaluations
                    similarities = []
                    for hist_eval in recent_evaluations:
                        sim_score = self._calculate_similarity(
                            evaluation,
                            hist_eval
                        )
                        similarities.append(sim_score)
                    
                    hist_confidence = np.mean(similarities)
                    
                    # Weight current and historical confidence
                    confidence = 0.7 * base_confidence + 0.3 * hist_confidence
                else:
                    confidence = base_confidence
            else:
                confidence = base_confidence
                
            return min(1.0, max(0.0, confidence))
            
        except Exception as e:
            logger.error(f"Error calculating confidence: {str(e)}")
            return 0.5
            
    def _calculate_similarity(
        self,
        eval1: Dict[str, Any],
        eval2: Dict[str, Any]
    ) -> float:
        """Calculate similarity between two evaluations."""
        try:
            # Extract key metrics for comparison
            metrics1 = self._extract_metrics(eval1)
            metrics2 = self._extract_metrics(eval2)
            
            # Calculate normalized difference
            diff_sum = 0
            count = 0
            
            for key in set(metrics1.keys()) & set(metrics2.keys()):
                diff = abs(metrics1[key] - metrics2[key])
                diff_sum += min(
                    1.0,
                    diff / max(abs(metrics1[key]), abs(metrics2[key]), 1e-6)
                )
                count += 1
                
            return 1.0 - (diff_sum / count if count > 0 else 0)
            
        except Exception as e:
            logger.error(f"Error calculating similarity: {str(e)}")
            return 0.0
            
    def _extract_metrics(self, evaluation: Dict[str, Any]) -> Dict[str, float]:
        """Extract numerical metrics from evaluation."""
        metrics = {}
        
        def extract_recursive(obj: Any, prefix: str = ""):
            if isinstance(obj, (int, float)):
                metrics[prefix] = float(obj)
            elif isinstance(obj, dict):
                for k, v in obj.items():
                    new_prefix = f"{prefix}.{k}" if prefix else k
                    extract_recursive(v, new_prefix)
            elif isinstance(obj, list):
                for i, v in enumerate(obj):
                    extract_recursive(v, f"{prefix}[{i}]")
                    
        extract_recursive(evaluation)
        return metrics
        
    def detect_trends(
        self,
        archive: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Detect trends in the evaluation archive."""
        try:
            if not archive:
                return {}
                
            # Group by evaluation type
            grouped_evals = {}
            for entry in archive:
                eval_type = entry["stage"]
                if eval_type not in grouped_evals:
                    grouped_evals[eval_type] = []
                grouped_evals[eval_type].append(entry)
                
            trends = {}
            for eval_type, evals in grouped_evals.items():
                if len(evals) < 2:
                    continue
                    
                # Calculate trend for each metric
                metric_trends = {}
                metrics = self._extract_metrics(evals[-1]["evaluation"])
                
                for metric in metrics:
                    values = [
                        self._extract_metrics(e["evaluation"]).get(metric, 0)
                        for e in evals[-5:]  # Look at last 5 entries
                    ]
                    
                    if len(values) >= 2:
                        # Fit polynomial for trend analysis
                        coeffs = np.polyfit(range(len(values)), values, 2)
                        slope = coeffs[0]  # Quadratic coefficient
                        
                        # Calculate trend strength and volatility
                        trend_strength = abs(slope)
                        volatility = np.std(values) / (np.mean(values) + 1e-6)
                        
                        metric_trends[metric] = {
                            "slope": float(slope),
                            "direction": "accelerating" if slope > 0.01
                                       else "decelerating" if slope < -0.01
                                       else "stable",
                            "strength": float(trend_strength),
                            "volatility": float(volatility)
                        }
                        
                trends[eval_type] = metric_trends
                
            return trends
            
        except Exception as e:
            logger.error(f"Error detecting trends: {str(e)}")
            return {}
            
    def aggregate(
        self,
        input_data: AggregationInput
    ) -> AggregationResult:
        """Perform sophisticated aggregation of evaluation results."""
        try:
            # Convert query result to EvaluationResult
            current_eval = EvaluationResult(
                stage=input_data.query_result["stage"],
                evaluation=input_data.query_result["evaluation"]
            )
            
            # Calculate confidence
            confidence = self.calculate_confidence(
                current_eval.evaluation,
                input_data.archive
            )
            
            # Detect trends
            trends = self.detect_trends(input_data.archive)
            
            # Detect anomalies
            metrics_history = [
                {
                    **self._extract_metrics(entry["evaluation"]),
                    "timestamp": entry.get(
                        "timestamp",
                        datetime.now() - timedelta(
                            seconds=len(input_data.archive) - i
                        )
                    )
                }
                for i, entry in enumerate(input_data.archive)
            ]
            
            anomalies = self.anomaly_detector.detect_anomalies(
                metrics_history
            )
            
            # Generate decision candidates
            candidates = self._generate_decision_candidates(
                current_eval,
                input_data.archive,
                trends,
                anomalies
            )
            
            # Optimize decision
            final_decision, optimization_score = self.decision_optimizer.optimize_decision(
                candidates,
                input_data.optimization_criteria,
                {"historical_decisions": self.recent_decisions}
            )
            
            # Update recent decisions
            self.recent_decisions.append(final_decision)
            if len(self.recent_decisions) > self.max_history:
                self.recent_decisions.pop(0)
                
            # Gather supporting evidence
            supporting_evidence = [
                {
                    "stage": entry["stage"],
                    "timestamp": entry.get("timestamp", datetime.now()),
                    "relevance": self._calculate_similarity(
                        entry["evaluation"],
                        current_eval.evaluation
                    )
                }
                for entry in input_data.archive[-5:]  # Last 5 entries as evidence
            ]
            
            return AggregationResult(
                final_decision=final_decision,
                confidence_score=confidence,
                supporting_evidence=supporting_evidence,
                trends=trends,
                anomalies=anomalies,
                optimization_score=optimization_score
            )
            
        except Exception as e:
            logger.error(f"Error in aggregation: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Aggregation failed: {str(e)}"
            )
            
    def _generate_decision_candidates(
        self,
        current_eval: EvaluationResult,
        archive: List[Dict[str, Any]],
        trends: Dict[str, Any],
        anomalies: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate decision candidates based on current state."""
        candidates = []
        
        # Base candidate from current evaluation
        base_candidate = {
            "decision": current_eval.evaluation,
            "confidence": current_eval.confidence or 0.5,
            "timestamp": current_eval.timestamp
        }
        candidates.append(base_candidate)
        
        # Generate alternative candidates based on trends
        for eval_type, metric_trends in trends.items():
            for metric, trend_info in metric_trends.items():
                if trend_info["strength"] > 0.1:  # Significant trend
                    # Create adjusted candidate
                    adjusted = base_candidate.copy()
                    adjusted["decision"] = current_eval.evaluation.copy()
                    
                    # Adjust based on trend
                    if "." in metric:
                        keys = metric.split(".")
                        target = adjusted["decision"]
                        for key in keys[:-1]:
                            if key not in target:
                                target[key] = {}
                            target = target[key]
                        if keys[-1] in target and isinstance(target[keys[-1]], (int, float)):
                            target[keys[-1]] *= (1 + trend_info["slope"])
                            
                    candidates.append(adjusted)
                    
        # Generate candidates considering anomalies
        if anomalies:
            # Create conservative candidate
            conservative = base_candidate.copy()
            conservative["decision"] = current_eval.evaluation.copy()
            conservative["confidence"] *= 0.8  # Reduce confidence due to anomalies
            candidates.append(conservative)
            
        return candidates

# --------------------------------------------------------------------------------
# FastAPI Setup
# --------------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context for startup/shutdown."""
    logger.info("Initializing Aggregation Module...")
    
    # Initialize aggregation engine
    app.state.aggregation_engine = AggregationEngine()
    logger.info("Aggregation engine initialized")
    
    yield
    
    # Cleanup
    logger.info("Shutting down Aggregation Module...")
    app.state.aggregation_engine = None

app = FastAPI(
    title="HMAS Aggregation Module",
    version="1.0.0",
    lifespan=lifespan
)

# Redis & Rate Limiting
redis_client = redis.Redis.from_url(settings.REDIS_URL, decode_responses=True)

limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["10/minute"]
)
app.state.limiter = limiter

# Prometheus Monitoring
Instrumentator().instrument(app).expose(app, endpoint="/metrics")

# Middleware
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])

# --------------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------------
@app.get("/health")
async def health_check():
    """Basic health check."""
    return {"status": "ok"}

@app.get("/ready")
async def readiness_check():
    """Readiness check with Redis connectivity test."""
    try:
        await redis_client.ping()
        return {"status": "ready"}
    except Exception as e:
        logger.warning(f"Redis not ready: {e}")
        raise HTTPException(status_code=500, detail="Redis not ready")

@app.post("/aggregate", response_model=AggregationResult)
@limiter.limit("10/minute")
async def aggregate(request: Request, input_data: AggregationInput):
    """
    Aggregate evaluation results into a final decision.
    
    Args:
        input_data: Evaluation archive and current query result
        
    Returns:
        Aggregated result with confidence and supporting evidence
    """
    try:
        result = request.app.state.aggregation_engine.aggregate(input_data)
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Aggregation error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Aggregation failed"
        )

if __name__ == "__main__":
    uvicorn.run(
        "aggregation_main:app",
        host="0.0.0.0",
        port=8300,
        reload=True
    )
