import pytest
from fastapi.testclient import TestClient
from communication.comm_optimization import app

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

def test_optimize_endpoint():
    response = client.post("/optimize", json={})
    assert response.status_code == 200
    data = response.json()
    assert "metrics" in data
    assert "strategy" in data
    assert data["strategy"] in ["broadcast", "unicast", "gossip"]
