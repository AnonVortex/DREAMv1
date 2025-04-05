# Routing Module API Reference

## Base URL
`http://localhost:8300`

## Endpoints

### POST /route
- **Description:** Determines which specialized processing agent should handle the fused data.
- **Request (JSON):**
  ```json
  {
    "fused_features": "Fused(vision_result,audio_result)"
  }

Response (JSON):
json
Copy
{
  "agent": "VisionOptimizationAgent"
}
If "vision" is detected, routes to VisionOptimizationAgent; otherwise, defaults to a standard agent.

Errors:
422 Unprocessable Entity for validation errors.
GET /health
Description: Health check endpoint.
Response: {"status": "ok"}
GET /ready
Description: Readiness check.
Response: {"status": "ready"}