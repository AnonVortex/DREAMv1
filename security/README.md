# HMAS Security Module

## Overview
The Security Module provides comprehensive security services for the Hierarchical Multi-Agent System, including authentication, encryption, and access control mechanisms.

## Features
- JWT-based authentication
- Data encryption/decryption
- Access control and permissions management
- Rate limiting
- Audit logging
- Security headers

## Configuration
The module can be configured through environment variables and the `config.py` file:

```bash
SECRET_KEY=your-secret-key
ENCRYPTION_KEY=your-encryption-key
```

## API Endpoints
- `POST /token`: Generate access tokens
- `POST /encrypt`: Encrypt sensitive data
- `POST /decrypt`: Decrypt encrypted data
- `POST /hash`: Hash sensitive data

## Dependencies
- FastAPI
- Python-JOSE
- Cryptography
- Passlib
- Python-dotenv

## Usage
```python
from security import SecurityConfig, create_access_token

# Configure security
config = SecurityConfig(
    encryption_enabled=True,
    access_control_enabled=True
)

# Generate token
token = create_access_token(
    data={"sub": "user_id", "permissions": ["read", "write"]}
)
```

## Security Best Practices
1. Always use HTTPS in production
2. Rotate encryption keys regularly
3. Implement proper access control
4. Monitor audit logs
5. Keep dependencies updated

## Contributing
See the main project's contribution guidelines.

## License
Same as the main project. 