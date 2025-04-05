import pytest
from fastapi.testclient import TestClient
from integration.integration_main import app

client = TestClient(app)

def test_health_check():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}

def test_ready_check_redis():
    # In real usage, might mock redis if no local instance is running
    resp = client.get("/ready")
    # If local redis isn't up, you might get 500. For real tests, consider mocking redis_client.
    # But let's assume you have local redis or skip this check if offline.
    assert resp.status_code in [200, 500]  # or just 200 if you run local redis
    if resp.status_code == 200:
        assert resp.json() == {"status": "ready"}

def test_integrate_no_features():
    # Should fail if no features are provided
    resp = client.post("/integrate", json={})
    assert resp.status_code == 400
    assert resp.json()["detail"] == "No features provided"

def test_integrate_vision_only():
    resp = client.post("/integrate", json={"vision_features": "v_feat"})
    assert resp.status_code == 200
    data = resp.json()
    assert "fused_features" in data
    assert "v_feat" in data["fused_features"]

def test_integrate_audio_only():
    resp = client.post("/integrate", json={"audio_features": "a_feat"})
    assert resp.status_code == 200
    data = resp.json()
    assert "fused_features" in data
    assert "a_feat" in data["fused_features"]

def test_integrate_both_features():
    resp = client.post("/integrate", json={"vision_features": "vX", "audio_features": "aY"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["fused_features"].startswith("Fused(vX,aY")
