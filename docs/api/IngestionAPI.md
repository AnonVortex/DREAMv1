# Ingestion Module API Reference

## Base URL
`http://localhost:8000`

## Endpoints

### POST /ingest
- **Description:** Uploads a file (multipart/form-data) and stores it in chunks.
- **Request:**
  - Form-data parameter: `file` (e.g., a `.txt`, `.csv`, `.mp4`, etc.)
- **Response (JSON):**
  ```json
  {
    "message": "File successfully ingested",
    "filename": "example.txt",
    "file_path": "data/example.txt"
  }

Errors:
400 Bad Request if file type is unsupported.
429 Too Many Requests if rate limit is exceeded.
GET /health
Description: Returns basic health status.
Response: {"status": "ok"}
GET /ready
Description: Checks if Redis is reachable.
Response: {"status": "ready"} or a 500 error if Redis is unavailable.