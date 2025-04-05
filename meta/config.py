import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env if present

class MetaSettings:
    HOST = os.getenv("META_HOST", "0.0.0.0")
    PORT = int(os.getenv("META_PORT", "8301"))
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

settings = MetaSettings()
