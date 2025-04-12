import pytest
from fastapi.testclient import TestClient
from security.security_service import app
from security.config import SECRET_KEY, ENCRYPTION_KEY

@pytest.fixture
def client():
    return TestClient(app)

@pytest.fixture
def test_token():
    from security.security_service import create_access_token
    return create_access_token({"sub": "test_user", "permissions": ["read", "write"]})

@pytest.fixture
def test_encrypted_data():
    from security.security_service import encrypt_data
    return encrypt_data("test data")

@pytest.fixture
def test_hashed_data():
    from security.security_service import hash_sensitive_data
    return hash_sensitive_data("test password") 