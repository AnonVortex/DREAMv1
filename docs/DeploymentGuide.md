# HMAS Deployment Guide

## Overview
This guide describes how to deploy the HMAS pipeline, either locally or in a production-like environment.

## Prerequisites
- **Docker** and **Docker Compose** installed.
- **Git** to clone the repository.
- **Sufficient CPU/RAM** if running all containers locally.

## Local Deployment
1. **Clone the Repo**:
   ```bash
   git clone https://github.com/YourUsername/hmas-prototype.git
   cd hmas-prototype

   docker-compose up --build

Check each container’s logs (docker logs <container_name>).
Test the aggregator at http://localhost:9000/run_pipeline. If running locally.

Production Deployment
Configure a remote host or cloud service (e.g., AWS EC2).
Set Environment Variables in .env or your orchestrator’s config.
Deploy using Docker Compose or Kubernetes. For Kubernetes, create a Deployment and Service for each module.

Post-Deployment
Monitor using Prometheus endpoints (/metrics) on each container.
Check health endpoints (/health, /ready) for each module.
Scale containers or pods if resource usage is high.

Troubleshooting
Module Failing: Inspect container logs. Ensure port conflicts are resolved.
High Resource Usage: Consider running fewer containers or scaling out with a larger instance.
Network Issues: Verify Docker network configuration. Check each service’s environment variables for correct URLs.