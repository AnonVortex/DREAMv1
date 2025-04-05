# Feedback Module API Reference

## Base URL
`http://localhost:8600`

## Endpoints

### POST /feedback
- **Description:** Generates a feedback summary based on the final decision.
- **Request (JSON):**
  ```json
  {
    "final_decision": "AGGREGATED_DECISION"
  }

Response (JSON):
json
Copy
{
  "feedback": {
    "accuracy": 0.92,
    "latency": 0.35,
    "error_rate": 0.03
  },
  "updated_params": {
    "learning_rate": 0.001000,
    "batch_size": 24,
    "update_frequency": 3
  }
}
Errors:
422 Unprocessable Entity if required fields are missing.
GET /health
Description: Health check.
Response: {"status": "ok"}
GET /ready
Description: Readiness check.
Response: {"status": "ready"}