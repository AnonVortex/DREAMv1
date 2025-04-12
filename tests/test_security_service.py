import pytest
from fastapi.testclient import TestClient
from security.security_service import app, SecurityManager
from datetime import datetime, timedelta
import jwt

@pytest.fixture
def test_client():
    return TestClient(app)

@pytest.fixture
def security_manager():
    return SecurityManager()

def test_create_user(test_client):
    response = test_client.post(
        "/users",
        json={
            "username": "testuser",
            "password": "testpass123",
            "role": "user",
            "security_level": "standard"
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "user_id" in data

def test_authenticate_user(test_client):
    # First create a user
    test_client.post(
        "/users",
        json={
            "username": "authuser",
            "password": "authpass123",
            "role": "user",
            "security_level": "standard"
        }
    )
    
    # Then authenticate
    response = test_client.post(
        "/token",
        data={
            "username": "authuser",
            "password": "authpass123"
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"

def test_invalid_authentication(test_client):
    response = test_client.post(
        "/token",
        data={
            "username": "wronguser",
            "password": "wrongpass"
        }
    )
    assert response.status_code == 401

def test_create_security_event(test_client):
    # First authenticate
    auth_response = test_client.post(
        "/token",
        data={
            "username": "authuser",
            "password": "authpass123"
        }
    )
    token = auth_response.json()["access_token"]
    
    # Create security event
    response = test_client.post(
        "/events",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "event_type": "login_attempt",
            "source_ip": "192.168.1.1",
            "user_id": "testuser",
            "details": {"success": True}
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "event_id" in data

def test_get_security_events(test_client):
    # First authenticate
    auth_response = test_client.post(
        "/token",
        data={
            "username": "authuser",
            "password": "authpass123"
        }
    )
    token = auth_response.json()["access_token"]
    
    # Get events
    response = test_client.get(
        "/events",
        headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)

def test_create_threat_alert(test_client):
    # First authenticate
    auth_response = test_client.post(
        "/token",
        data={
            "username": "authuser",
            "password": "authpass123"
        }
    )
    token = auth_response.json()["access_token"]
    
    # Create threat alert
    response = test_client.post(
        "/threats",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "threat_type": "brute_force",
            "source_ip": "192.168.1.1",
            "severity": "high",
            "details": {"attempts": 5}
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "alert_id" in data

def test_get_threat_alerts(test_client):
    # First authenticate
    auth_response = test_client.post(
        "/token",
        data={
            "username": "authuser",
            "password": "authpass123"
        }
    )
    token = auth_response.json()["access_token"]
    
    # Get alerts
    response = test_client.get(
        "/threats",
        headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)

def test_token_expiration(security_manager):
    # Create token with short expiration
    token = security_manager.create_access_token(
        data={"sub": "testuser"},
        expires_delta=timedelta(seconds=1)
    )
    
    # Wait for expiration
    import time
    time.sleep(2)
    
    # Verify token is expired
    with pytest.raises(jwt.ExpiredSignatureError):
        security_manager.verify_token(token)

def test_token_blacklist(security_manager):
    # Create token
    token = security_manager.create_access_token(
        data={"sub": "testuser"}
    )
    
    # Blacklist token
    security_manager.blacklist_token(token)
    
    # Verify token is blacklisted
    with pytest.raises(ValueError):
        security_manager.verify_token(token)

def test_rate_limiting(test_client):
    # Make multiple requests quickly
    responses = []
    for _ in range(60):
        response = test_client.post(
            "/token",
            data={
                "username": "authuser",
                "password": "authpass123"
            }
        )
        responses.append(response)
    
    # Verify rate limiting
    assert any(r.status_code == 429 for r in responses)

def test_health_check(test_client):
    response = test_client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy" 