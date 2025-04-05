import os
from dotenv import load_dotenv

load_dotenv()

class MonitoringSettings:
    HOST = os.getenv("MONITORING_HOST", "0.0.0.0")
    PORT = int(os.getenv("MONITORING_PORT", "8700"))
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

settings = MonitoringSettings()
