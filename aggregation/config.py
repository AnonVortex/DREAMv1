import os
from dotenv import load_dotenv

load_dotenv()

class AggregationSettings:
    HOST = os.getenv("AGGREGATION_HOST", "0.0.0.0")
    PORT = int(os.getenv("AGGREGATION_PORT", "8500"))
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

settings = AggregationSettings()
