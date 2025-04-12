import pytest
from fastapi.testclient import TestClient
from security.security_service import app, create_access_token, encrypt_data, decrypt_data, hash_sensitive_data
from security.config import SECRET_KEY, ENCRYPTION_KEY

client = TestClient(app)

def test_create_access_token():
    data = {"sub": "test_user", "permissions": ["read", "write"]}
    token = create_access_token(data)
    assert token is not None
    assert isinstance(token, str)

def test_encrypt_decrypt_data():
    test_data = "sensitive information"
    encrypted = encrypt_data(test_data)
    assert encrypted is not None
    assert encrypted != test_data
    
    decrypted = decrypt_data(encrypted)
    assert decrypted == test_data

def test_hash_sensitive_data():
    test_data = "password123"
    hashed = hash_sensitive_data(test_data)
    assert hashed is not None
    assert hashed != test_data
    assert len(hashed) > 0

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_token_endpoint():
    response = client.post(
        "/token",
        json={"username": "test_user", "password": "test_password"}
    )
    assert response.status_code == 200
    assert "access_token" in response.json()

def test_encrypt_endpoint():
    response = client.post(
        "/encrypt",
        json={"data": "sensitive data"}
    )
    assert response.status_code == 200
    assert "encrypted_data" in response.json()

def test_decrypt_endpoint():
    # First encrypt some data
    encrypt_response = client.post(
        "/encrypt",
        json={"data": "sensitive data"}
    )
    encrypted_data = encrypt_response.json()["encrypted_data"]
    
    # Then try to decrypt it
    response = client.post(
        "/decrypt",
        json={"encrypted_data": encrypted_data}
    )
    assert response.status_code == 200
    assert response.json()["decrypted_data"] == "sensitive data"

def test_hash_endpoint():
    response = client.post(
        "/hash",
        json={"data": "password123"}
    )
    assert response.status_code == 200
    assert "hashed_data" in response.json()

def test_invalid_token():
    response = client.get(
        "/protected",
        headers={"Authorization": "Bearer invalid_token"}
    )
    assert response.status_code == 401

def test_rate_limiting():
    # Make multiple requests to test rate limiting
    for _ in range(10):
        response = client.get("/health")
        assert response.status_code == 200
    
    # This should be rate limited
    response = client.get("/health")
    assert response.status_code == 429 