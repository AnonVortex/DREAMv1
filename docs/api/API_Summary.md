# HMAS API Summary

This document provides a high-level overview of the REST API endpoints for each module in the HMAS pipeline. Each module is designed as an independent microservice with its own endpoints, and they are orchestrated via the Pipeline Aggregator.

---

## 1. Ingestion Module
- **Base URL:** `http://localhost:8000`
- **Key Endpoints:**
  - **POST /ingest:** Uploads a file (multipart/form-data) in chunks.  
    _Response Example:_  
    ```json
    {
      "message": "File successfully ingested",
      "filename": "example.txt",
      "file_path": "data/example.txt"
    }
    ```
  - **GET /health:** Returns `{"status": "ok"}`.
  - **GET /ready:** Checks readiness (e.g., Redis connectivity).

---

## 2. Perception Module
- **Base URL:** `http://localhost:8100`
- **Key Endpoints:**
  - **POST /perceive:** Processes the provided file or file path to extract multi-modal features.  
    _Response Example:_  
    ```json
    {
      "vision_features": "vision_result",
      "audio_features": "audio_result"
    }
    ```
  - **GET /health:** Basic health check.
  - **GET /ready:** Readiness check.

---

## 3. Integration Module
- **Base URL:** `http://localhost:8200`
- **Key Endpoints:**
  - **POST /integrate:** Fuses extracted features into a unified representation.  
    _Response Example:_  
    ```json
    {
      "fused_features": "Fused(vision_result,audio_result)"
    }
    ```
  - **GET /health:** Health check.
  - **GET /ready:** Readiness check.

---

## 4. Routing Module
- **Base URL:** `http://localhost:8300`
- **Key Endpoints:**
  - **POST /route:** Analyzes the fused features to determine which specialized agent should process the data.  
    _Response Example:_  
    ```json
    {
      "agent": "VisionOptimizationAgent"
    }
    ```
  - **GET /health:** Health check.
  - **GET /ready:** Readiness check.

---

## 5. Specialized Module
- **Base URL:** `http://localhost:8400`
- **Key Endpoints:**
  - **POST /specialize:** Processes routed data using domain-specific algorithms (e.g., Graph RL) and returns specialized output.  
    _Response Example:_  
    ```json
    {
      "graph_optimization_action": 1,
      "graph_optimization_value": 0.98
    }
    ```
  - **GET /health:** Health check.
  - **GET /ready:** Readiness check.

---

## 6. Meta Module
- **Base URL:** `http://localhost:8301`
- **Key Endpoints:**
  - **POST /meta:** Evaluates the specialized output and produces a meta-evaluation report.  
    _Response Example:_  
    ```json
    {
      "Verification": "Outputs consistent",
      "Consensus": "Majority agreement reached",
      "SelfMonitoring": "Performance within acceptable limits",
      "Iteration": "No further iteration required"
    }
    ```
  - **GET /health:** Health check.
  - **GET /ready:** Readiness check.

---

## 7. Memory Module
- **Base URL:** `http://localhost:8401`
- **Key Endpoints:**
  - **POST /memory:** Archives the meta output and retrieves the latest archived record.  
    _Response Example:_  
    ```json
    {
      "archive": [{"stage": "meta", "evaluation": { ... }}],
      "query_result": {"stage": "meta", "evaluation": { ... }}
    }
    ```
  - **GET /health:** Health check.
  - **GET /ready:** Readiness check.

---

## 8. Aggregation Module
- **Base URL:** `http://localhost:8500`
- **Key Endpoints:**
  - **POST /aggregate:** Aggregates the memory archive to produce a final decision.  
    _Response Example:_  
    ```json
    {
      "final_decision": { "stage": "meta", "evaluation": { ... } }
    }
    ```
  - **GET /health:** Health check.
  - **GET /ready:** Readiness check.

---

## 9. Feedback Module
- **Base URL:** `http://localhost:8600`
- **Key Endpoints:**
  - **POST /feedback:** Generates a feedback summary based on the final decision.  
    _Response Example:_  
    ```json
    {
      "feedback": { "accuracy": 0.92, "latency": 0.35, "error_rate": 0.03 },
      "updated_params": { "learning_rate": 0.001, "batch_size": 24, "update_frequency": 3 }
    }
    ```
  - **GET /health:** Health check.
  - **GET /ready:** Readiness check.

---

## 10. Monitoring Module
- **Base URL:** `http://localhost:8700`
- **Key Endpoints:**
  - **GET /monitor:** Provides system resource usage, diagnostics, and scaling decisions.  
    _Response Example:_  
    ```json
    {
      "resource_usage": { "cpu_usage": 35.0, "memory_usage": 50.0 },
      "diagnostics": "All systems operational.",
      "scaling_decision": "No scaling required."
    }
    ```
  - **GET /health:** Health check.
  - **GET /ready:** Readiness check.

---

## 11. Graph RL Module
- **Base URL:** `http://localhost:8800`
- **Key Endpoints:**
  - **POST /graph_rl:** Triggers Graph RL training/inference (dummy training loop).  
    _Response Example:_  
    ```json
    { "action": 2, "value_estimate": 0.995 }
    ```
  - **GET /health:** Health check.
  - **GET /ready:** Readiness check.

---

## 12. Communication Optimization Module
- **Base URL:** `http://localhost:8900`
- **Key Endpoints:**
  - **POST /optimize:** Optimizes inter-agent communication and returns the chosen strategy with performance metrics.  
    _Response Example:_  
    ```json
    {
      "metrics": { "message_latency": 0.25, "message_success_rate": 0.89 },
      "strategy": "unicast"
    }
    ```
  - **GET /health:** Health check.
  - **GET /ready:** Readiness check.

---

## Conclusion
Each module exposes standardized health and readiness endpoints, along with dedicated API endpoints to handle its specific functionality. For detailed API specifications for each module, refer to the individual documentation files in this folder.

This summary serves as an at-a-glance guide to the HMAS pipeline's REST API surface, ensuring consistency and ease-of-use for both developers and external integrators.
