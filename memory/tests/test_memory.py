import pytest
from fastapi.testclient import TestClient
from memory.memory_main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_ready_check():
    response = client.get("/ready")
    # Allow 200 or 500 if Redis isn't running locally
    assert response.status_code in [200, 500]
    if response.status_code == 200:
        assert response.json() == {"status": "ready"}

def test_memory_endpoint_valid():
    payload = {
        "meta_output": {
            "Verification": "Outputs consistent",
            "Consensus": "Majority agreement reached",
            "SelfMonitoring": "Performance within acceptable limits",
            "Iteration": "No further iteration required"
        }
    }
    response = client.post("/memory", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "archive" in data
    assert "query_result" in data

def test_memory_endpoint_invalid():
    # Missing the required field should return a validation error
    response = client.post("/memory", json={})
    assert response.status_code == 422
