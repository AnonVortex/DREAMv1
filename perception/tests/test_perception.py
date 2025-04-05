import pytest
from fastapi.testclient import TestClient
from perception.perception_main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_ready_check():
    response = client.get("/ready")
    assert response.status_code == 200
    assert response.json() == {"status": "ready"}

def test_perceive_no_paths():
    response = client.post("/perceive", json={})
    assert response.status_code == 400
    assert response.json()["detail"] == "No image or audio path provided"

def test_perceive_with_image():
    response = client.post("/perceive", json={"image_path": "sample.jpg"})
    assert response.status_code == 200
    data = response.json()
    assert data["vision_features"] == "vision_feature_vector_stub"
    assert data["audio_features"] is None

def test_perceive_with_audio():
    response = client.post("/perceive", json={"audio_path": "sample.wav"})
    assert response.status_code == 200
    data = response.json()
    assert data["audio_features"] == "audio_feature_vector_stub"
    assert data["vision_features"] is None
