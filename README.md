# HMAS (AGI DREAM) - Hierarchical Multi-Agent System

## Overview
This repository contains a **Hierarchical Multi-Agent System (HMAS)** prototype aimed at progressing toward AGI. The system is structured as multiple containerized microservices (Ingestion, Perception, Integration, Routing, Specialized, Meta, Memory, Aggregation, Feedback, Monitoring, Graph RL, Communication) plus a Pipeline Aggregator that orchestrates them end-to-end.

## Key Features
- **Modular Architecture**: Each stage in the pipeline is its own containerized FastAPI service.
- **Multi-Modal Data Handling**: Processes images, audio, text, etc.
- **Rate Limiting & Monitoring**: Uses Redis + slowapi for rate limiting; Prometheus for metrics.
- **Scalable & Extensible**: Docker-based microservices with potential for cloud deployment.
- **Comprehensive Documentation**: Check the [`docs/`](./docs/) folder for architecture diagrams, deployment guides, developer setup, and API references.

## Directory Structure

H-MAS(AGI DREAM)/
├── ingestion/       # Data ingestion module
├── perception/      # Perception module for multi-modal feature extraction
├── integration/     # Integration module for fusing features
├── routing/         # Routing module to allocate tasks
├── specialized/     # Specialized processing (Graph RL, domain-specific tasks)
├── meta/            # Meta-evaluation of specialized outputs
├── memory/          # Archives meta outputs (long-term memory)
├── aggregation/     # Combines memory data into a final decision
├── feedback/        # Gathers system feedback metrics
├── monitoring/      # Monitors resource usage & diagnostics
├── graph_rl/        # Graph RL module for advanced decision-making
├── communication/   # Optimizes inter-agent communication strategies
├── pipelines/       # Pipeline aggregator that calls each module in sequence
├── docs/            # Documentation (architecture, deployment, API references)
└── docker-compose.yml

## Quick Start
 - Clone the repository:
	``bash
	Copy
	git clone https://github.com/YourUsername/hmas-prototype.git
	cd hmas-prototype
	
  - Run the entire pipeline:
	``bash
	Copy
	docker-compose up --build
	Trigger the pipeline aggregator:
	``bash
	Copy
	curl -X POST http://localhost:9000/run_pipeline

Documentation
High-Level Architecture: docs/Architecture.md
Deployment Guide: docs/DeploymentGuide.md
Developer Setup: docs/DeveloperSetup.md
API References: docs/api/
Roadmap: docs/Roadmap.md

Testing
Each module has its own tests/ folder with Pytest scripts. You can run them individually:

	``bash
	Copy
	python -m pytest ingestion/tests/
	python -m pytest perception/tests/
	# ...
	Or use the Makefile to run them all at once (make test).

Contributing
-Fork the repo & create feature branches from develop.
-Open PRs for review.
-See docs/DeveloperSetup.md for environment setup instructions.
-License
 See LICENSE for details

Contact
For inquiries, please open an issue or reach out to sotoyaneza@gmail.com