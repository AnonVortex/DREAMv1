# HMAS Pipeline Aggregator - Technical Documentation

## 1. Role in the Pipeline
The Pipeline Aggregator serves as the central coordinator for the HMAS pipeline. It sequentially calls the API endpoints of each module (Ingestion, Perception, Integration, etc.) and aggregates their outputs into a final, consolidated response.

## 2. Design & Implementation
- **Asynchronous Orchestration**: Utilizes httpx’s AsyncClient to make non-blocking API calls.
- **Error Handling**: Checks response status at each stage and aborts if any module fails.
- **Configuration**: Endpoints and ports are managed via environment variables in `config.py`.
- **Logging & Monitoring**: Standardized logging via `logging.conf` is used to trace each pipeline stage.
- **Containerization**: The aggregator is container-ready via its Dockerfile.

## 3. Data Flow
1. **Ingestion**: Uploads a dummy file and returns a file path.
2. **Perception**: Processes the file (using the file path) to extract features.
3. **Integration**: Fuses the features into a consolidated representation.
4. **Routing**: Determines which specialized processing agent to use.
5. **Specialized**: Processes the routed data.
6. **Meta**: Evaluates the specialized output.
7. **Memory**: Archives the meta evaluation.
8. **Aggregation**: Produces a final decision.
9. **Feedback**: Provides system feedback based on the final decision.
10. **Monitoring**: Checks system health and resource usage.
11. **Graph RL**: Optionally triggers Graph RL training/inference.
12. **Communication**: Optimizes inter-agent communication.

## 4. Environment & Configuration
- **Variables** (loaded via `config.py` and optionally overridden via `.env`):
  - `INGESTION_URL`, `PERCEPTION_URL`, `INTEGRATION_URL`, `ROUTING_URL`, `SPECIALIZED_URL`, `META_URL`, `MEMORY_URL`, `AGGREGATION_URL`, `FEEDBACK_URL`, `MONITORING_URL`, `GRAPH_RL_URL`, `COMM_URL`, and `PIPELINE_PORT`.

## 5. Future Enhancements
- **Parallel Execution**: Implement parallel processing for independent modules.
- **Retries & Timeouts**: Enhance robustness with retry mechanisms.
- **Detailed Metrics**: Integrate more detailed metrics for end-to-end performance monitoring.
- **Cloud Integration**: Adapt the aggregator for a cloud-based orchestration environment.

## 6. Additional Notes
- The aggregator is fully containerized and designed to be part of a larger Docker Compose or Kubernetes deployment.
- Extensive logging provides traceability for debugging and performance analysis.
- This module currently uses dummy inputs (for file ingestion, etc.) and stubs for demonstration purposes. Real data inputs and advanced processing logic should be integrated as the pipeline evolves.
