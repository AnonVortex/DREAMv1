# HMAS Feedback Module - Technical Documentation

## 1. Role in the Pipeline
The Feedback module collects performance metrics and evaluation data from the final decision output (from Aggregation) and produces a comprehensive feedback summary. This feedback is used to fine-tune system parameters and improve overall performance.

## 2. Design & Implementation
- **FastAPI Microservice**: Built using an async lifespan context for efficient startup and shutdown.
- **Rate Limiting**: Utilizes slowapi to limit incoming requests (default 10 requests/minute).
- **Monitoring**: Exposes Prometheus metrics at `/metrics`.
- **Health & Readiness Endpoints**: Provides `/health` and `/ready` endpoints, with Redis readiness check.
- **Configuration**: Uses environment variables loaded via `config.py` (overridable with `.env`).
- **Logging**: Standardized logging is configured via `logging.conf`.

## 3. Data Flow
- **Input**: Receives a JSON payload containing the final decision. Example:
  ```json
  {
      "final_decision": "AGGREGATED_DECISION"
  }
