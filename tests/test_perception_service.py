import pytest
from fastapi.testclient import TestClient
from perception.perception_service import app, PerceptionManager
import numpy as np
import base64
import json
from PIL import Image
import io
import librosa

@pytest.fixture
def test_client():
    return TestClient(app)

@pytest.fixture
def perception_manager():
    return PerceptionManager()

def create_test_image():
    # Create a simple test image
    img = Image.new('RGB', (100, 100), color='red')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    return base64.b64encode(img_byte_arr).decode()

def create_test_audio():
    # Create a simple test audio signal
    sample_rate = 22050
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
    
    # Convert to bytes
    audio_bytes = io.BytesIO()
    librosa.output.write_wav(audio_bytes, audio, sample_rate)
    return base64.b64encode(audio_bytes.getvalue()).decode()

def test_process_text(test_client):
    text_data = {
        "text": "This is a test sentence for natural language processing.",
        "analysis_types": ["sentiment", "entities", "keywords"]
    }
    
    response = test_client.post("/process/text", json=text_data)
    assert response.status_code == 200
    data = response.json()
    
    assert "sentiment" in data
    assert "entities" in data
    assert "keywords" in data
    assert isinstance(data["sentiment"], dict)
    assert isinstance(data["entities"], list)
    assert isinstance(data["keywords"], list)

def test_process_image(test_client):
    image_data = {
        "image": create_test_image(),
        "analysis_types": ["objects", "colors", "faces"]
    }
    
    response = test_client.post("/process/image", json=image_data)
    assert response.status_code == 200
    data = response.json()
    
    assert "objects" in data
    assert "colors" in data
    assert "faces" in data
    assert isinstance(data["objects"], list)
    assert isinstance(data["colors"], list)
    assert isinstance(data["faces"], list)

def test_process_audio(test_client):
    audio_data = {
        "audio": create_test_audio(),
        "analysis_types": ["speech_to_text", "audio_features"]
    }
    
    response = test_client.post("/process/audio", json=audio_data)
    assert response.status_code == 200
    data = response.json()
    
    assert "speech_to_text" in data
    assert "audio_features" in data
    assert isinstance(data["speech_to_text"], str)
    assert isinstance(data["audio_features"], dict)

def test_invalid_text_input(test_client):
    invalid_data = {
        "text": "",  # Empty text
        "analysis_types": ["sentiment"]
    }
    
    response = test_client.post("/process/text", json=invalid_data)
    assert response.status_code == 400

def test_invalid_image_input(test_client):
    invalid_data = {
        "image": "invalid_base64",  # Invalid base64
        "analysis_types": ["objects"]
    }
    
    response = test_client.post("/process/image", json=invalid_data)
    assert response.status_code == 400

def test_invalid_audio_input(test_client):
    invalid_data = {
        "audio": "invalid_base64",  # Invalid base64
        "analysis_types": ["speech_to_text"]
    }
    
    response = test_client.post("/process/audio", json=invalid_data)
    assert response.status_code == 400

def test_unsupported_analysis_type(test_client):
    text_data = {
        "text": "Test text",
        "analysis_types": ["unsupported_type"]
    }
    
    response = test_client.post("/process/text", json=text_data)
    assert response.status_code == 400

def test_batch_processing(test_client):
    batch_data = {
        "inputs": [
            {
                "type": "text",
                "content": "Test text",
                "analysis_types": ["sentiment"]
            },
            {
                "type": "image",
                "content": create_test_image(),
                "analysis_types": ["objects"]
            }
        ]
    }
    
    response = test_client.post("/process/batch", json=batch_data)
    assert response.status_code == 200
    data = response.json()
    
    assert len(data["results"]) == 2
    assert "sentiment" in data["results"][0]
    assert "objects" in data["results"][1]

def test_feature_extraction(test_client):
    text_data = {
        "text": "Test text for feature extraction",
        "extraction_type": "embeddings"
    }
    
    response = test_client.post("/extract_features", json=text_data)
    assert response.status_code == 200
    data = response.json()
    
    assert "features" in data
    assert isinstance(data["features"], list)
    assert len(data["features"]) > 0

def test_pattern_recognition(test_client):
    pattern_data = {
        "text": "Test text with pattern 123-456-789",
        "pattern_type": "phone_numbers"
    }
    
    response = test_client.post("/recognize_patterns", json=pattern_data)
    assert response.status_code == 200
    data = response.json()
    
    assert "patterns" in data
    assert isinstance(data["patterns"], list)
    assert len(data["patterns"]) > 0

def test_perception_config(test_client):
    config_data = {
        "text_analysis": {
            "sentiment_threshold": 0.7,
            "entity_types": ["person", "organization"]
        },
        "image_analysis": {
            "object_confidence": 0.8,
            "face_detection": True
        },
        "audio_analysis": {
            "sample_rate": 16000,
            "language": "en-US"
        }
    }
    
    response = test_client.post("/config", json=config_data)
    assert response.status_code == 200
    data = response.json()
    
    assert data["status"] == "success"
    assert "config_id" in data

def test_get_perception_config(test_client):
    response = test_client.get("/config")
    assert response.status_code == 200
    data = response.json()
    
    assert "text_analysis" in data
    assert "image_analysis" in data
    assert "audio_analysis" in data

def test_health_check(test_client):
    response = test_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    
    assert data["status"] == "healthy"
    assert "version" in data
    assert "timestamp" in data

def test_metrics(test_client):
    response = test_client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    
    assert "request_count" in data
    assert "processing_times" in data
    assert "error_count" in data 