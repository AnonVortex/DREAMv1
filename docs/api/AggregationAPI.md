# Aggregation Module API Reference

## Base URL
`http://localhost:8500`

## Endpoints

### POST /aggregate
- **Description:** Aggregates memory outputs into a final decision.
- **Request (JSON):**
  ```json
  {
    "archive": [
      {"stage": "meta", "evaluation": { ... }}
    ],
    "query_result": {"stage": "meta", "evaluation": { ... }}
  }

Response (JSON):
json
Copy
{
  "final_decision": {
    "stage": "meta",
    "evaluation": { ... }
  }
}
Errors:
400 Bad Request if the archive is empty.
422 Unprocessable Entity for invalid payloads.
GET /health
Description: Health check.
Response: {"status": "ok"}
GET /ready
Description: Readiness check.
Response: {"status": "ready"}