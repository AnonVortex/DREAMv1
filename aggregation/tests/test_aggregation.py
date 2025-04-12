import pytest
from datetime import datetime, timedelta
from typing import Dict, Any, List
import numpy as np
from fastapi.testclient import TestClient

from ..aggregation_main import (
    app,
    AggregationEngine,
    AggregationInput,
    AggregationResult,
    EvaluationResult
)

@pytest.fixture
def test_client():
    """Create a test client."""
    return TestClient(app)

@pytest.fixture
def aggregation_engine():
    """Create an aggregation engine instance."""
    return AggregationEngine()

@pytest.fixture
def sample_evaluation():
    """Create a sample evaluation."""
    return {
        "stage": "meta",
        "evaluation": {
            "accuracy": 0.85,
            "confidence": 0.75,
            "latency": 120,
            "resource_usage": 0.65
        }
    }

@pytest.fixture
def sample_archive(sample_evaluation):
    """Create a sample archive of evaluations."""
    base_time = datetime.now()
    archive = []
    
    for i in range(5):
        eval_copy = sample_evaluation.copy()
        eval_copy["evaluation"] = eval_copy["evaluation"].copy()
        eval_copy["evaluation"].update({
            "accuracy": 0.85 + np.random.normal(0, 0.05),
            "confidence": 0.75 + np.random.normal(0, 0.05),
            "latency": 120 + np.random.normal(0, 10),
            "resource_usage": 0.65 + np.random.normal(0, 0.05)
        })
        eval_copy["timestamp"] = base_time - timedelta(minutes=i*5)
        archive.append(eval_copy)
    
    return archive

def test_health_check(test_client):
    """Test health check endpoint."""
    response = test_client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_calculate_confidence(aggregation_engine, sample_evaluation, sample_archive):
    """Test confidence calculation."""
    confidence = aggregation_engine.calculate_confidence(
        sample_evaluation["evaluation"],
        sample_archive
    )
    assert 0 <= confidence <= 1
    assert isinstance(confidence, float)

def test_calculate_similarity(aggregation_engine):
    """Test similarity calculation between evaluations."""
    eval1 = {
        "accuracy": 0.8,
        "confidence": 0.7,
        "latency": 100
    }
    eval2 = {
        "accuracy": 0.85,
        "confidence": 0.75,
        "latency": 110
    }
    
    similarity = aggregation_engine._calculate_similarity(eval1, eval2)
    assert 0 <= similarity <= 1
    assert isinstance(similarity, float)

def test_detect_trends(aggregation_engine, sample_archive):
    """Test trend detection in evaluation archive."""
    trends = aggregation_engine.detect_trends(sample_archive)
    
    assert isinstance(trends, dict)
    assert "meta" in trends
    
    meta_trends = trends["meta"]
    assert "accuracy" in meta_trends
    assert "direction" in meta_trends["accuracy"]
    assert meta_trends["accuracy"]["direction"] in ["increasing", "decreasing", "stable"]

def test_aggregate_empty_archive(test_client):
    """Test aggregation with empty archive."""
    input_data = {
        "archive": [],
        "query_result": {
            "stage": "meta",
            "evaluation": {
                "accuracy": 0.8,
                "confidence": 0.7
            }
        }
    }
    
    response = test_client.post("/aggregate", json=input_data)
    assert response.status_code == 400
    assert "Empty archive" in response.json()["detail"]

def test_aggregate_success(test_client, sample_evaluation, sample_archive):
    """Test successful aggregation."""
    input_data = {
        "archive": sample_archive,
        "query_result": sample_evaluation
    }
    
    response = test_client.post("/aggregate", json=input_data)
    assert response.status_code == 200
    
    result = response.json()
    assert "final_decision" in result
    assert "confidence_score" in result
    assert "supporting_evidence" in result
    assert "trends" in result
    
    assert isinstance(result["confidence_score"], float)
    assert 0 <= result["confidence_score"] <= 1
    
    assert len(result["supporting_evidence"]) <= 5
    for evidence in result["supporting_evidence"]:
        assert "stage" in evidence
        assert "relevance" in evidence
        assert 0 <= evidence["relevance"] <= 1

def test_trend_detection_empty_archive(aggregation_engine):
    """Test trend detection with empty archive."""
    trends = aggregation_engine.detect_trends([])
    assert trends == {}

def test_trend_detection_single_entry(aggregation_engine, sample_evaluation):
    """Test trend detection with single entry."""
    trends = aggregation_engine.detect_trends([sample_evaluation])
    assert isinstance(trends, dict)
    assert len(trends) == 0  # No trends with single entry

def test_extract_metrics(aggregation_engine):
    """Test metric extraction from evaluation."""
    evaluation = {
        "accuracy": 0.8,
        "confidence": 0.7,
        "latency": 100,
        "name": "test",  # Should be ignored
        "active": True   # Should be ignored
    }
    
    metrics = aggregation_engine._extract_metrics(evaluation)
    assert set(metrics.keys()) == {"accuracy", "confidence", "latency"}
    assert all(isinstance(v, float) for v in metrics.values())

def test_history_management(aggregation_engine, sample_evaluation, sample_archive):
    """Test decision history management."""
    input_data = AggregationInput(
        archive=sample_archive,
        query_result=sample_evaluation
    )
    
    # Make multiple aggregations
    for _ in range(150):  # More than max_history
        aggregation_engine.aggregate(input_data)
    
    assert len(aggregation_engine.recent_decisions) <= aggregation_engine.max_history

def test_invalid_input(test_client):
    """Test aggregation with invalid input."""
    invalid_input = {
        "archive": "not_a_list",  # Invalid type
        "query_result": {}
    }
    
    response = test_client.post("/aggregate", json=invalid_input)
    assert response.status_code == 422  # Validation error

def test_missing_query_result(test_client, sample_archive):
    """Test aggregation with missing query result."""
    input_data = {
        "archive": sample_archive,
        "query_result": {}  # Empty query result
    }
    
    response = test_client.post("/aggregate", json=input_data)
    assert response.status_code == 422  # Validation error

def test_rate_limiting(test_client, sample_evaluation, sample_archive):
    """Test rate limiting."""
    input_data = {
        "archive": sample_archive,
        "query_result": sample_evaluation
    }
    
    # Make multiple requests
    responses = []
    for _ in range(12):  # More than rate limit
        responses.append(test_client.post("/aggregate", json=input_data))
    
    # Check that some requests were rate limited
    assert any(r.status_code == 429 for r in responses)  # Too many requests
