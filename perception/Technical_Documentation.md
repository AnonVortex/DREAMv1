# HMAS - Perception Module (Technical Documentation)

## 1. Architecture
- **Lifespan-based** startup/shutdown (no deprecation).
- **Optional** advanced libraries for image/audio processing.
- **Async** approach to avoid blocking the pipeline.

## 2. Data Flow
- Ingested data or references (like image paths) come in (e.g., from the Ingestion module).
- This module extracts features (e.g., vision, audio, text embeddings).
- Passes those features back to the pipeline (Integration or next stage).

## 3. Key Endpoints
- `POST /perceive`:
  - Receives multi-modal data references (e.g. `image_path`, `audio_path`).
  - Returns JSON with extracted features (`vision_features`, `audio_features`).

## 4. Future Enhancements
- **Real Model Integration**: YOLO, CLIP, or Whisper for advanced vision/audio.
- **GPU Acceleration**: Add device detection for PyTorch or TensorFlow.
- **Streaming**: Real-time video streams or microphone input.

## 5. Testing & Validation
- `tests/test_perception.py` for unit/integration tests.
- Mock real CV/audio calls for now if you have no model in place.
