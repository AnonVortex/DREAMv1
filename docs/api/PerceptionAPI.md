# Perception Module API Reference

## Base URL
`http://localhost:8100`

## Endpoints

### POST /perceive
- **Description:** Processes an input file (or file path) to extract multi-modal features.
- **Request (JSON):**
  ```json
  {
    "image": "path/to/image.jpg",
    "audio": "path/to/audio.wav"
  }

Fields are optional; if not provided, corresponding features will be null.

Response (JSON):
json
Copy
{
  "vision_features": "vision_result",
  "audio_features": "audio_result"
}
Errors:
400 Bad Request for invalid input.
GET /health
Description: Returns basic health status.
Response: {"status": "ok"}
GET /ready
Description: Checks module readiness (e.g., Redis connectivity).
Response: {"status": "ready"} or error if dependencies fail.
yaml
Copy
