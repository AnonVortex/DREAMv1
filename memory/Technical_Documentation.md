# HMAS Memory Module - Technical Documentation

## 1. Role in the Pipeline
The Memory module stores the evaluated outputs from the Meta module into long-term memory. It archives data for historical reference and provides a query mechanism to retrieve the latest stored result, serving as a working memory for downstream processing.

## 2. Design & Implementation
- **FastAPI Microservice**: Uses an async lifespan context for startup/shutdown.
- **Rate Limiting**: Uses slowapi to limit requests (default 10/minute).
- **Monitoring**: Exposes Prometheus metrics at `/metrics`.
- **Health & Readiness**: Provides `/health` and `/ready` endpoints (with a Redis ping).
- **Configuration**: Environment variables are loaded via `config.py` (overridable by a `.env` file).
- **Logging**: Uses standardized logging as defined in `logging.conf`.

## 3. Data Flow
- **Input**: A JSON payload from the Meta module with its output, for example:
  ```json
  {
      "meta_output": {
          "Verification": "Outputs consistent",
          "Consensus": "Majority agreement reached",
          "SelfMonitoring": "Performance within acceptable limits",
          "Iteration": "No further iteration required"
      }
  }
