import pytest
from fastapi.testclient import TestClient
from aggregation.aggregation_main import app  # Ensure PYTHONPATH includes the aggregation package

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_ready_check():
    response = client.get("/ready")
    # Allow 200 or 500 if Redis is not running locally
    assert response.status_code in [200, 500]
    if response.status_code == 200:
        assert response.json() == {"status": "ready"}

def test_aggregate_valid_input():
    payload = {
        "archive": [
            {"stage": "meta", "evaluation": {"Verification": "Consistent"}}
        ],
        "query_result": {"stage": "meta", "evaluation": {"Verification": "Consistent"}}
    }
    response = client.post("/aggregate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "final_decision" in data

def test_aggregate_empty_archive():
    payload = {
        "archive": [],
        "query_result": {"stage": "meta", "evaluation": {"Verification": "Consistent"}}
    }
    response = client.post("/aggregate", json=payload)
    assert response.status_code == 400
    assert response.json()["detail"] == "Empty archive"

def test_aggregate_invalid_payload():
    response = client.post("/aggregate", json={})
    assert response.status_code == 422
