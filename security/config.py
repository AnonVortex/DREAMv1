import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Security configurations
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-here")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Encryption settings
ENCRYPTION_KEY = os.getenv("ENCRYPTION_KEY", "your-encryption-key-here")

# Access control settings
DEFAULT_PERMISSIONS = {
    "admin": ["read", "write", "execute", "manage"],
    "agent": ["read", "execute"],
    "user": ["read"]
}

# Rate limiting settings
RATE_LIMIT_REQUESTS = 100
RATE_LIMIT_PERIOD = 60  # seconds

# Audit logging settings
AUDIT_LOG_ENABLED = True
AUDIT_LOG_FILE = "security_audit.log"

# Security headers
SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "X-XSS-Protection": "1; mode=block",
    "Strict-Transport-Security": "max-age=31536000; includeSubDomains"
} 