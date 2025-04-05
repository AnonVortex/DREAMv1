import os
from dotenv import load_dotenv

load_dotenv()

class SpecializedSettings:
    HOST = os.getenv("SPECIALIZED_HOST", "0.0.0.0")
    PORT = int(os.getenv("SPECIALIZED_PORT", "8400"))
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")  # Added attribute

settings = SpecializedSettings()
