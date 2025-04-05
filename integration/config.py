import os
from dotenv import load_dotenv

load_dotenv()

class IntegrationSettings:
    HOST = os.getenv("INTEGRATION_HOST", "0.0.0.0")
    PORT = int(os.getenv("INTEGRATION_PORT", "8200"))
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

settings = IntegrationSettings()
