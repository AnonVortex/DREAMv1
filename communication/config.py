import os
from dotenv import load_dotenv

load_dotenv()

class CommSettings:
    HOST = os.getenv("COMM_HOST", "0.0.0.0")
    PORT = int(os.getenv("COMM_PORT", "8900"))
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

settings = CommSettings()
