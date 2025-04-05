import pytest
from fastapi.testclient import TestClient
from graph_rl.graph_rl_agent import app  # Ensure PYTHONPATH includes the graph_rl package

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

def test_graph_rl_endpoint():
    response = client.post("/graph_rl")
    assert response.status_code == 200
    data = response.json()
    assert "action" in data
    assert "value_estimate" in data
