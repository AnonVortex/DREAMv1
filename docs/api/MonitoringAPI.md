# Monitoring Module API Reference

## Base URL
`http://localhost:8700`

## Endpoints

### GET /monitor
- **Description:** Provides a summary of system resource usage, diagnostics, and scaling decisions.
- **Response (JSON):**
  ```json
  {
    "resource_usage": {
      "cpu_usage": 35.0,
      "memory_usage": 50.0
    },
    "diagnostics": "All systems operational.",
    "scaling_decision": "No scaling required."
  }

Errors:
Returns appropriate HTTP status if internal checks fail.
GET /health
Description: Health check.
Response: {"status": "ok"}
GET /ready
Description: Readiness check (Redis connectivity).
Response: {"status": "ready"}