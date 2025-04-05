import os
from dotenv import load_dotenv

load_dotenv()

class PerceptionSettings:
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", 8100))

settings = PerceptionSettings()
