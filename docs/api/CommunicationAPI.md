# Communication Optimization Module API Reference

## Base URL
`http://localhost:8900`

## Endpoints

### POST /optimize
- **Description:** Triggers the communication optimization routine, which selects a communication strategy and returns performance metrics.
- **Request:**  
  - An optional empty JSON payload (reserved for future parameters).
- **Response (JSON):**
  ```json
  {
    "metrics": {
      "message_latency": 0.25,
      "message_success_rate": 0.89
    },
    "strategy": "unicast"
  }

Errors:
422 Unprocessable Entity for invalid input.
GET /health
Description: Health check.
Response: {"status": "ok"}
GET /ready
Description: Readiness check.
Response: {"status": "ready"}