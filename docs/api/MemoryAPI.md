# Memory Module API Reference

## Base URL
`http://localhost:8401`

## Endpoints

### POST /memory
- **Description:** Archives the meta output and retrieves the latest archived record.
- **Request (JSON):**
  ```json
  {
    "meta_output": {
      "Verification": "Outputs consistent",
      "Consensus": "Majority agreement reached",
      "SelfMonitoring": "Performance within acceptable limits",
      "Iteration": "No further iteration required"
    }
  }

Response (JSON):
json
Copy
{
  "archive": [
    {"stage": "meta", "evaluation": { ... }}
  ],
  "query_result": {"stage": "meta", "evaluation": { ... }}
}
Errors:
422 Unprocessable Entity if required data is missing.
GET /health
Description: Health check.
Response: {"status": "ok"}
GET /ready
Description: Readiness check (Redis connectivity).
Response: {"status": "ready"}
yaml
Copy
