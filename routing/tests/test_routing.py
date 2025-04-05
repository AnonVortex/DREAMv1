import pytest
from fastapi.testclient import TestClient
from routing.routing_main import app  # Correct# If running from project root, ensure PYTHONPATH is set correctly

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_ready_check():
    response = client.get("/ready")
    assert response.status_code == 200
    assert response.json() == {"status": "ready"}

def test_route_no_input():
    response = client.post("/route", json={})
    # Pydantic will raise validation error for missing required field
    assert response.status_code == 422

def test_route_with_vision():
    data = {"fused_features": "Fused(vision_data,audio_data)"}
    response = client.post("/route", json=data)
    assert response.status_code == 200
    json_resp = response.json()
    assert "agent" in json_resp
    assert json_resp["agent"] == "VisionOptimizationAgent"

def test_route_default():
    data = {"fused_features": "Fused(audio_data)"}
    response = client.post("/route", json=data)
    assert response.status_code == 200
    json_resp = response.json()
    assert json_resp["agent"] == "DefaultRoutingAgent"
