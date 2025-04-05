# HMAS Pipeline Aggregator

## Overview
The Pipeline Aggregator orchestrates the entire HMAS pipeline by sequentially calling each module’s API endpoint and collecting their outputs. This module acts as the central coordinator, ensuring data flows from Ingestion → Perception → Integration → Routing → Specialized → Meta → Memory → Aggregation → Feedback → Monitoring → Graph RL → Communication.

## Key Features
- **Asynchronous Orchestration**: Uses httpx to call endpoints asynchronously.
- **Centralized Error Handling**: Checks response status codes at each stage.
- **Environment-Based Configuration**: Endpoints and settings are managed via a `.env` file and `config.py`.
- **Logging**: Standardized logging for tracing the pipeline execution.
- **Containerized**: Ready for Docker deployment.
- **Testable**: Can be tested end-to-end with dummy inputs.

## Setup Instructions

### Local Development
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
