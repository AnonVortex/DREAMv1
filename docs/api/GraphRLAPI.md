# Graph RL Module API Reference

## Base URL
`http://localhost:8800`

## Endpoints

### POST /graph_rl
- **Description:** Triggers training/inference of the Graph RL agent using dummy data.
- **Request:**  
  - Optionally, a JSON payload (currently not used, but reserved for future parameters).
- **Response (JSON):**
  ```json
  {
    "action": 2,
    "value_estimate": 0.995
  }

Errors:
422 Unprocessable Entity if input is invalid.
GET /health
Description: Health check.
Response: {"status": "ok"}
GET /ready
Description: Readiness check.
Response: {"status": "ready"}
yaml
Copy
