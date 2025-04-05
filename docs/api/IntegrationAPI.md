# Integration Module API Reference

## Base URL
`http://localhost:8200`

## Endpoints

### POST /integrate
- **Description:** Fuses extracted features from the Perception module into a unified representation.
- **Request (JSON):**
  ```json
  {
    "vision_features": "vision_result",
    "audio_features": "audio_result"
  }

Response (JSON):
json
Copy
{
  "fused_features": "Fused(vision_result,audio_result)"
}
Errors:
400 Bad Request if no features are provided.
GET /health
Description: Basic health check.
Response: {"status": "ok"}
GET /ready
Description: Checks readiness (e.g., Redis connectivity).
Response: {"status": "ready"}