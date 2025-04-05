import pytest
from fastapi.testclient import TestClient
from feedback.feedback_main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_ready_check():
    response = client.get("/ready")
    assert response.status_code in [200, 500]
    if response.status_code == 200:
        assert response.json() == {"status": "ready"}

def test_feedback_valid():
    payload = {"final_decision": "AGGREGATED_DECISION"}
    response = client.post("/feedback", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "feedback" in data
    assert "updated_params" in data

def test_feedback_invalid():
    # Missing required field
    response = client.post("/feedback", json={})
    assert response.status_code == 422
