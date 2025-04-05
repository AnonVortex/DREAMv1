# Meta Module API Reference

## Base URL
`http://localhost:8301`

## Endpoints

### POST /meta
- **Description:** Evaluates the output from the Specialized module and produces a meta-evaluation report.
- **Request (JSON):**
  ```json
  {
    "specialized_output": {
      "graph_optimization_action": 1,
      "graph_optimization_value": 0.98
    }
  }

Response (JSON):
json
Copy
{
  "Verification": "Outputs consistent",
  "Consensus": "Majority agreement reached",
  "SelfMonitoring": "Performance within acceptable limits",
  "Iteration": "No further iteration required"
}
Errors:
422 Unprocessable Entity if required fields are missing.
GET /health
Description: Health check.
Response: {"status": "ok"}
GET /ready
Description: Readiness check.
Response: {"status": "ready"}