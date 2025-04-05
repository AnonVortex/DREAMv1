# HMAS Developer Setup

## Overview
This document helps new developers quickly set up a local development environment for the HMAS pipeline.

## Prerequisites
- **Python 3.9+**
- **Docker & Docker Compose** (for containerized runs)
- **Git** for version control
- **Optional**: Virtual environment tool (venv, conda, pyenv)

## Steps

1. **Clone the Repository**
   ```bash
   git clone https://github.com/YourUsername/hmas-prototype.git
   cd hmas-prototype

Install Python Dependencies (if you want to run modules outside Docker):

bash
Copy
pip install -r ingestion/requirements.txt
pip install -r perception/requirements.txt
# ... repeat for each module or set up a monorepo approach

Run Tests

bash
Copy
python -m pytest ingestion/tests/
python -m pytest perception/tests/
# ... etc.

Start Services with Docker Compose

bash
Copy
docker-compose up --build
This builds and launches all modules in containers.

Check Endpoints

Visit http://localhost:<port>/docs for each module’s FastAPI docs.
e.g., http://localhost:8000/docs for Ingestion.

Development Tips
Use .env Files: Each module can have a .env to override environment variables.
Logging: Check logs in each container (docker logs <container_name>) or watch them in real time.
Incremental Development: You can run individual modules locally (via uvicorn) while others run in Docker.

Contributing
Create feature branches from develop.
Submit PRs for review.
Follow any coding style or linting guidelines set in the repo.