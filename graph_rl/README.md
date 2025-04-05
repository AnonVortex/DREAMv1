# HMAS Graph RL Module

## Overview
The Graph RL module implements a Graph Reinforcement Learning (Graph RL) agent designed to improve decision-making within the HMAS pipeline. It leverages PyTorch for model training and inference. This service exposes an endpoint to trigger training (or inference) and returns the agent's final output. The module is containerized, uses environment-based configuration, logging, rate limiting, and Prometheus metrics for monitoring.

## Key Features
- **Graph RL Agent** built with PyTorch.
- **FastAPI Microservice** with async lifespan context.
- **Rate Limiting** via slowapi.
- **Prometheus Metrics** exposed at `/metrics`.
- **Health & Readiness Endpoints**.
- Environment configuration via `.env` and `config.py`.

## Setup Instructions

### Local Development
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
