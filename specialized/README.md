# HMAS - Specialized Module

## Overview
The Specialized module processes domain-specific tasks after Routing. It is responsible for executing specialized algorithms—such as Graph Reinforcement Learning optimization—and producing tailored outputs for subsequent pipeline stages (e.g., Meta).

## Key Features
- **FastAPI Microservice** with async lifespan context for clean startup and shutdown.
- **Rate Limiting** using slowapi.
- **Prometheus Metrics** exposed at `/metrics` for monitoring.
- **Health** and **Readiness** endpoints.
- Environment-based configuration via a local `config.py` and optional `.env`.

## Endpoints
- `GET /health` - Basic health check.
- `GET /ready` - Readiness check.
- `POST /specialize` - Processes specialized input and returns a specialized output.

## Setup Instructions
1. **Install Dependencies**  
   ```bash
   pip install -r requirements.txt
