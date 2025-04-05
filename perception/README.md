# HMAS - Perception Module

## Overview
This **Perception** microservice (container) handles low-level processing of multi-modal inputs (images, audio, text, etc.)
to extract features for the rest of the pipeline.

## Key Features
- **Async** data processing stubs for vision/audio
- **Rate limiting** (if needed)
- **Prometheus** metrics via `/metrics`
- **Health/ready** endpoints

## Setup
1. **Install Dependencies**  
   ```bash
   pip install -r requirements.txt
