# HMAS - Ingestion Module

## Overview
This module ingests files (text, images, video, etc.) with:
- **Async** chunk-based writes
- **Redis-based** rate limiting (via `slowapi`)
- **Prometheus metrics** at `/metrics`
- **Health** (`/health`) & **Readiness** (`/ready`) checks
- **Chunk-based** ingestion for large files
- **Lifespan-based** startup & shutdown (no `on_event` warnings)

## Getting Started
1. **Install Dependencies**  
   ```bash
   pip install -r requirements.txt
