# Specialized Module API Reference

## Base URL
`http://localhost:8400`

## Endpoints

### POST /specialize
- **Description:** Processes routed data using domain-specific algorithms (e.g., Graph RL) and returns specialized output.
- **Request (JSON):**
  ```json
  {
    "graph_optimization": "GraphOptimizationAgent"
  }

If the field equals "GraphOptimizationAgent", specialized processing is triggered.

Response (JSON):
json
Copy
{
  "graph_optimization_action": 1,
  "graph_optimization_value": 0.98
}
Errors:
422 Unprocessable Entity if the input is invalid.
GET /health
Description: Health check.
Response: {"status": "ok"}
GET /ready
Description: Readiness check.
Response: {"status": "ready"}