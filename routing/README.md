# HMAS - Routing Module

## Overview
The Routing module is responsible for decomposing tasks by examining the fused feature set received from the Integration module and determining which specialized processing agent should handle the task. This module is built using FastAPI, features asynchronous startup/shutdown via a lifespan context, and incorporates rate limiting, Prometheus metrics, and health/readiness endpoints.

## Key Endpoints
- **GET /health**: Basic health check.
- **GET /ready**: Readiness check.
- **POST /route**: Accepts JSON input containing `fused_features` and returns a routing decision (e.g., routes to VisionOptimizationAgent if vision data is detected).

## Setup
1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
