# HMAS - Routing Module (Technical Documentation)

## 1. Role in the Pipeline
The Routing module receives fused features (e.g., from Integration) and determines which specialized processing agent should handle the next step. It essentially decomposes tasks based on the input data characteristics.

## 2. Design & Implementation
- **FastAPI Microservice**: Uses an async lifespan context for startup/shutdown, eliminating deprecated event decorators.
- **Rate Limiting**: Implements slowapi to limit requests (default 10/minute).
- **Monitoring**: Exposes Prometheus metrics at `/metrics`.
- **Health & Readiness**: Provides `/health` and `/ready` endpoints.
- **Configuration**: Environment variables (e.g., ROUTING_HOST, ROUTING_PORT) are loaded from `.env` via a local config module (`config.py`).

## 3. Data Flow
- **Input**: A JSON payload containing a key `fused_features` (e.g., `{"fused_features": "Fused(vision_data,audio_data)"}`).
- **Processing**: The module checks for specific keywords in the fused features to decide routing. For instance, if "vision" is present, it routes to a VisionOptimizationAgent; otherwise, it defaults to a standard agent.
- **Output**: Returns a JSON response with the routing decision, e.g., `{"agent": "VisionOptimizationAgent"}`.

## 4. Environment & Configuration
- **Variables**:
  - `ROUTING_HOST` (default: `0.0.0.0`)
  - `ROUTING_PORT` (default: `8300`)
- These are managed via the `config.py` file and can be overridden in production deployments.

## 5. Future Enhancements
- **Advanced Routing Logic**: Implement dynamic decision-making or ML-based routing.
- **Distributed Messaging**: Incorporate a message broker (e.g., Kafka) to handle routing in real time.
- **Enhanced Health Checks**: Integrate with other modules for end-to-end pipeline readiness.

## 6. Additional Notes & Future References
- This module is fully containerized and ready to be deployed.
- Test coverage is provided in `tests/test_routing.py`.
- As the pipeline evolves, revisit the routing logic to adapt to new types of fused features or routing strategies.
