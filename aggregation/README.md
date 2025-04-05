# HMAS Aggregation Module

## Overview
The Aggregation module is responsible for synthesizing the outputs from the Memory (or Meta) module into a final decision. It aggregates historical evaluations into a coherent output that can be used for final decision-making.

## Key Features
- **FastAPI Microservice** with async lifespan for startup/shutdown.
- **Rate Limiting**: Uses slowapi to limit requests (default 10 requests/minute).
- **Prometheus Metrics**: Exposed at `/metrics` for monitoring.
- **Health & Readiness Endpoints**: `/health` and `/ready` for service diagnostics.
- **Configuration**: Environment variables managed via `config.py` (overridable with `.env`).

## Endpoints
- `GET /health`: Returns basic health status.
- `GET /ready`: Returns readiness status (pings Redis).
- `POST /aggregate`: Accepts a JSON payload with memory archive and query result, and returns a final decision.

## Setup Instructions

### Local Development
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
