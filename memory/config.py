import os
from dotenv import load_dotenv

load_dotenv()

class MemorySettings:
    HOST = os.getenv("MEMORY_HOST", "0.0.0.0")
    PORT = int(os.getenv("MEMORY_PORT", "8401"))
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

settings = MemorySettings()
