# HMAS Aggregation Module - Technical Documentation

## 1. Role in the Pipeline
The Aggregation module consolidates outputs from the Memory/Meta module into a final decision. It synthesizes historical data and the latest evaluation into a coherent output for downstream processing or final decision-making.

## 2. Design & Implementation
- **FastAPI Microservice**: Built with an async lifespan context to manage startup and shutdown without deprecated event handlers.
- **Rate Limiting**: Implements slowapi to limit incoming requests to prevent overload.
- **Monitoring**: Exposes Prometheus metrics at `/metrics` for integration with Grafana or similar tools.
- **Health & Readiness**: Provides `/health` and `/ready` endpoints (with a Redis ping for readiness).
- **Configuration**: Uses environment variables loaded via `config.py`, with optional overrides from a `.env` file.
- **Logging**: Consistent logging via a standardized `logging.conf`.

## 3. Data Flow
- **Input**: Expects a JSON payload structured as:
  ```json
  {
      "archive": [
          {"stage": "meta", "evaluation": {...}},
          ...
      ],
      "query_result": {"stage": "meta", "evaluation": {...}}
  }
