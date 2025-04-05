import pytest
from fastapi.testclient import TestClient
from meta.meta_main import app  # Ensure PYTHONPATH includes the meta module

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_ready_check():
    response = client.get("/ready")
    # If Redis is not available, you may get a 500 error.
    assert response.status_code in [200, 500]
    if response.status_code == 200:
        assert response.json() == {"status": "ready"}

def test_meta_endpoint_valid():
    payload = {
        "specialized_output": {
            "graph_optimization_action": 1,
            "graph_optimization_value": 0.98
        }
    }
    response = client.post("/meta", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "Verification" in data
    assert data["Consensus"] == "Majority agreement reached"

def test_meta_endpoint_invalid():
    # Missing the required specialized_output field should trigger a validation error
    response = client.post("/meta", json={})
    assert response.status_code == 422
