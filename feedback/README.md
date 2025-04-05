# HMAS Feedback Module

## Overview
The Feedback module gathers system feedback based on the final decision from the Aggregation module. It produces metrics such as accuracy, latency, error rates, and suggests updated parameters for the system. This service is built as a FastAPI microservice with an async lifespan context, rate limiting, Prometheus metrics, and health/readiness endpoints.

## Key Endpoints
- **GET /health**: Returns basic health status.
- **GET /ready**: Checks readiness (pings Redis).
- **POST /feedback**: Accepts a JSON payload with the final decision and returns a feedback summary.

## Setup Instructions

### Local Development
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
