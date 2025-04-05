# HMAS - Integration Module

## Overview
This module fuses multiple feature sets (vision, audio, text, etc.) after Perception. It combines them into a unified "working memory" representation for the next pipeline step (Routing).

### Key Features
- **Async** chunk-based logic if needed (though not typical here).
- **Rate-limited** endpoint (`/integrate`) to prevent overload.
- **Lifespan-based** startup (no `@app.on_event` warnings).
- **Prometheus** metrics via `/metrics`.
- **Readiness** check ensures Redis connectivity (optional).

## Setup Instructions

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
