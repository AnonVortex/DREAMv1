import os
from dotenv import load_dotenv

load_dotenv()

class FeedbackSettings:
    HOST = os.getenv("FEEDBACK_HOST", "0.0.0.0")
    PORT = int(os.getenv("FEEDBACK_PORT", "8600"))
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

settings = FeedbackSettings()
