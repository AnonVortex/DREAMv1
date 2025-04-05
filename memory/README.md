# HMAS Memory Module

## Overview
The Memory module archives the outputs from the Meta module into long-term memory storage and retrieves the latest archived data. It ensures that the system’s evaluations and decisions are stored for historical reference and further analysis.

## Key Features
- **FastAPI Microservice** with async lifespan context for clean startup/shutdown.
- **Rate Limiting**: Uses slowapi to limit requests.
- **Prometheus Metrics**: Exposed at `/metrics` for monitoring.
- **Health & Readiness Endpoints**: For service monitoring.
- **Redis Integration**: Optionally uses Redis for caching or storing state.

## Endpoints
- `GET /health`: Basic health check.
- `GET /ready`: Readiness check (pings Redis).
- `POST /memory`: Accepts a JSON payload with Meta module output and returns an archive and the latest (query) result.

## Setup Instructions

### Local Development
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
