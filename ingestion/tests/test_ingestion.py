import pytest
from fastapi.testclient import TestClient
from ingestion.ingestion_main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_ready_check_redis_failure(monkeypatch):
    def mock_ping():
        raise Exception("Redis error")
    monkeypatch.setattr("ingestion.ingestion_main.redis_client.ping", mock_ping)

    response = client.get("/ready")
    assert response.status_code == 500
    assert response.json()["detail"] == "Redis not ready"
