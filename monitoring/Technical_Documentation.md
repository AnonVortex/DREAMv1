# HMAS Monitoring Module - Technical Documentation

## 1. Role in the Pipeline
The Monitoring module continuously tracks the system’s resource usage (CPU, memory, etc.), runs diagnostic tests, and makes scalability decisions. It provides a comprehensive overview of system health, which is crucial for maintaining performance and reliability as the HMAS pipeline scales.

## 2. Design & Implementation
- **FastAPI Microservice**: Utilizes an async lifespan context to manage startup and shutdown without deprecated event handlers.
- **Rate Limiting**: Uses slowapi to prevent request overload (default 10 requests/minute).
- **Monitoring**: Exposes Prometheus metrics at `/metrics` for integration with visualization tools like Grafana.
- **Health & Readiness**: Provides `/health` and `/ready` endpoints (with Redis connectivity check).
- **Configuration**: Loads environment variables from `config.py` (with optional `.env` overrides).
- **Logging**: Standardized logging is configured via `logging.conf`.

## 3. Data Flow & Processing
- **Resource Usage Check**: Uses `psutil` to check CPU and memory usage.
- **Diagnostics**: Simulates a diagnostic delay and returns a summary message.
- **Scaling Decision**: Based on resource usage thresholds (e.g., >80% CPU or memory), the module decides whether scaling is required.
- **Output**: Returns a summary that includes:
  - Resource usage metrics (CPU, memory)
  - Diagnostics result
  - Scaling decision

## 4. Environment & Configuration
- **Variables**:
  - `MONITORING_HOST` (default: `0.0.0.0`)
  - `MONITORING_PORT` (default: `8700`)
  - `REDIS_URL` (default: `redis://localhost:6379`)
- These variables are loaded from `config.py` and can be overridden by a `.env` file.

## 5. Future Enhancements
- **Detailed Diagnostics**: Expand checks to include disk I/O, network performance, etc.
- **Automated Alerts & Scaling**: Integrate with cloud services for automated scaling and alerting.
- **Integration with Centralized Monitoring**: Use tools like Grafana for real-time dashboards.

## 6. Additional Notes & Future References
- The module is fully containerized and ready for deployment in the HMAS pipeline.
- Test coverage is provided in `tests/test_monitoring.py`.
- As the pipeline evolves, revisit this module to adapt to increased load and advanced diagnostics requirements.
