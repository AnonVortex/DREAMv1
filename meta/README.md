# HMAS Meta Module

## Overview
The Meta module evaluates the specialized output from the previous stage and produces a comprehensive report on system performance and output quality. It is built using FastAPI with a lifespan context, includes rate limiting, Prometheus metrics, and environment-based configuration.

## Key Endpoints
- `GET /health`: Basic health check.
- `GET /ready`: Readiness check.
- `POST /meta`: Accepts a JSON payload with specialized output and returns a meta-evaluation report.

## Setup Instructions

### Local Development
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
