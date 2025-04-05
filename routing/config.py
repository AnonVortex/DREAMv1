import os
from dotenv import load_dotenv

load_dotenv()  # Load variables from .env if present

class RoutingSettings:
    HOST = os.getenv("ROUTING_HOST", "0.0.0.0")
    PORT = int(os.getenv("ROUTING_PORT", "8300"))

settings = RoutingSettings()
