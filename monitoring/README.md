# HMAS Monitoring Module

## Overview
The Monitoring module is responsible for continuously tracking system resource usage (CPU, memory), running diagnostic tests, and making scaling decisions. It provides a summary of system health and resource usage to inform maintenance and scaling decisions within the HMAS pipeline.

## Key Endpoints
- **GET /health**: Returns basic health status.
- **GET /ready**: Checks system readiness by, for example, pinging Redis.
- **GET /metrics**: Exposes Prometheus metrics for monitoring.
- (Optional) Additional endpoints may be added as needed for diagnostics.

## Setup Instructions

### Local Development
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
