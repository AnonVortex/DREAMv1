# HMAS Architecture

## Overview
This document provides a **high-level architecture** of the Hierarchical Multi-Agent System (HMAS).

## System Components
1. **Ingestion** - Receives and stores incoming data (files, streams).
2. **Perception** - Extracts features from images, audio, or other modalities.
3. **Integration** - Fuses extracted features into a unified representation.
4. **Routing** - Determines specialized agent allocation based on integrated data.
5. **Specialized** - Executes domain-specific tasks (e.g., Graph RL).
6. **Meta** - Evaluates specialized outputs for consistency and correctness.
7. **Memory** - Archives meta evaluations for historical reference.
8. **Aggregation** - Synthesizes archived data into a final decision.
9. **Feedback** - Generates metrics (accuracy, latency) and updated system parameters.
10. **Monitoring** - Tracks system resource usage and scaling decisions.
11. **Graph RL** - Provides advanced decision-making using reinforcement learning on graph data.
12. **Communication** - Optimizes inter-agent communication strategies.
13. **Pipeline Aggregator** - Orchestrates the entire pipeline by sequentially calling each module.

## Data Flow
A typical pipeline run might look like:
1. **Ingestion** → 2. **Perception** → 3. **Integration** → 4. **Routing** → 5. **Specialized** → 6. **Meta** → 7. **Memory** → 8. **Aggregation** → 9. **Feedback** → 10. **Monitoring** → 11. **Graph RL** → 12. **Communication**

See [`diagrams/system-architecture.png`](./diagrams/system-architecture.png) and [`diagrams/data-flow.svg`](./diagrams/data-flow.svg) for visual representations.

## Technology Stack
- **Docker** & **Docker Compose** for containerization.
- **FastAPI** microservices for each module.
- **Redis** for caching/rate limiting (via slowapi).
- **Prometheus** for metrics exposure.

## Future Enhancements
- Parallel execution of independent modules.
- Advanced error handling & retries in the aggregator.
- Deeper ML integration for each module’s logic.

