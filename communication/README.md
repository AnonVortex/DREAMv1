# HMAS Communication Optimization Module

## Overview
The Communication module optimizes inter-agent communication by selecting a communication strategy (broadcast, unicast, or gossip) and reporting key metrics such as message latency and success rate. This module is a FastAPI microservice built with an async lifespan context, rate limiting, Prometheus monitoring, and environment-based configuration.

## Key Endpoints
- **GET /health**: Returns basic health status.
- **GET /ready**: Checks readiness (pings Redis).
- **POST /optimize**: Triggers communication optimization and returns the chosen strategy along with metrics.

## Setup Instructions

### Local Development
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
