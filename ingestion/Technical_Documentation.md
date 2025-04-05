# HMAS - Ingestion Module (Technical Documentation)

## Architecture
1. **FastAPI Lifespan** - Replaces `@app.on_event("startup"/"shutdown")` to avoid deprecation.
2. **Redis** - For rate-limiting data, readiness checks, potential caching.
3. **SlowAPI** - Limits requests, preventing DoS or abuse.
4. **Prometheus-FastAPI-Instrumentator** - Scrapes standard metrics for monitoring.
5. **AIOFiles** - Writes uploaded files asynchronously in chunks.

## Data Flow
- Client uploads file → `POST /ingest`
- `ingest_file` saves it to `/data/<filename>` chunk by chunk
- Rate limit (5 requests/min) via `slowapi`
- Health endpoints (`/health`, `/ready`) confirm service/Redis availability

## Key Files
- `ingestion_main.py` - Main FastAPI app
- `logging.conf` - Logging config
- `.env` - Environment variables (e.g., `REDIS_URL`)
- `tests/test_ingestion.py` - Basic test coverage

## Configuration & Environment Variables

For local development, we use a simple `.env` file to define basic settings:

```bash
HOST=0.0.0.0
PORT=8000
REDIS_URL=redis://localhost:6379

## Example Workflow
1. **User** calls `POST /ingest` with a `.mp4` video.
2. **App** chunk-writes the file to `/data/`.
3. **Prometheus** scrapes metrics at `/metrics`.
4. **Redis** ensures only 5 calls per minute per IP (adjustable).

## Next Steps
- Deploy via **Docker Compose** with a Redis container.
- Possibly add background tasks (Celery or Kafka).
- Integrate with **Perception** module for further AGI pipeline processing.

## Future Enhancements (Roadmap)

### 1. Security / Authentication
- Add OAuth2 or API key checks to ensure only authorized users can ingest data.

### 2. Real-Time / Streaming Ingestion
- Implement WebSocket endpoints (/ws/ingest) or integrate Kafka / Redis Streams for continuous feeds.

### 3. Additional File Preprocessing
- Consider partial validation, virus scanning, or indexing metadata during upload.

### 4. Cloud Storage Integration
- Offload file storage to S3 or GCS for high availability and scalability.

### 5. Health / Telemetry Enhancements
- Add advanced metrics for chunk speeds, queue lengths, file sizes, etc.

### 6. Thorough Integration with the Pipeline
- Possibly push metadata to a queue (e.g., Kafka, Celery) so the Perception module or others can automatically process newly ingested data.
