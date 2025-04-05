# HMAS Communication Optimization Module - Technical Documentation

## 1. Role in the Pipeline
The Communication module optimizes inter-agent communication within the HMAS pipeline. It selects the optimal communication strategy (e.g., broadcast, unicast, gossip) based on simulated performance metrics (message latency, success rate) and provides a standardized report for use in subsequent decision-making stages.

## 2. Design & Implementation
- **FastAPI Microservice**: Built with an async lifespan context to ensure a clean startup and shutdown process.
- **Rate Limiting**: Uses slowapi to protect the service from overload (default 10 requests per minute).
- **Monitoring**: Exposes Prometheus metrics at `/metrics` for real-time monitoring.
- **Health & Readiness**: Provides `/health` and `/ready` endpoints (the latter pings Redis for dependency checks).
- **Configuration**: Environment variables (e.g., COMM_HOST, COMM_PORT, REDIS_URL) are managed via `config.py`, with optional overrides from `.env`.
- **Logging**: Consistent logging is configured via `logging.conf`.

## 3. Data Flow & Processing
- **Input**: The module does not require a complex input payload; the `/optimize` endpoint can be triggered with an empty POST (or extended later).
- **Processing**:
  - The module simulates communication optimization by randomly generating metrics:
    - **Message Latency** (e.g., between 0.15 and 0.3 seconds).
    - **Message Success Rate** (e.g., between 0.8 and 0.95).
  - It then randomly selects a communication strategy (broadcast, unicast, or gossip).
- **Output**:
  ```json
  {
      "metrics": {
          "message_latency": 0.25,
          "message_success_rate": 0.89
      },
      "strategy": "unicast"
  }
