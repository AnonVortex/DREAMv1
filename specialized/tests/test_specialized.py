import pytest
from fastapi.testclient import TestClient
from specialized.specialized_main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_ready_check():
    response = client.get("/ready")
    assert response.status_code == 200
    assert response.json() == {"status": "ready"}

def test_specialize_default():
    # When no graph_optimization field is provided, expect default output.
    response = client.post("/specialize", json={})
    assert response.status_code == 200
    data = response.json()
    assert data.get("default_specialized_result") == True

def test_specialize_graph():
    response = client.post("/specialize", json={"graph_optimization": "GraphOptimizationAgent"})
    assert response.status_code == 200
    data = response.json()
    assert "graph_optimization_action" in data
    assert "graph_optimization_value" in data

def test_specialize_invalid_input():
    # If input is not valid JSON, FastAPI will return a 422 error.
    response = client.post("/specialize", data="not a json")
    assert response.status_code == 422
