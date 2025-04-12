# HMAS Developer Setup Guide

## Overview
This document provides comprehensive setup instructions for developers working on the HMAS (Hierarchical Multi-Agent System) AGI prototype.

## Prerequisites
- **Python 3.9+**
- **Docker & Docker Compose** (for containerized deployment)
- **Git** for version control
- **Redis** (for caching and message queues)
- **MongoDB** (for persistent storage)
- **NVIDIA GPU** (recommended for machine learning components)
- **CUDA Toolkit** (if using GPU)

## Core Modules
The system consists of the following core modules:
- Perception Service (Port 8100)
- Memory Service (Port 8200)
- Learning Service (Port 8300)
- Reasoning Service (Port 8400)
- Communication Service (Port 8500)
- Feedback Service (Port 8600)
- Integration Service (Port 8700)
- Meta Service (Port 8800)

## Setup Steps

1. **Clone the Repository**
   ```bash
   git clone https://github.com/AnonVortex/DREAMv1.git
   cd DREAMv1
   ```

2. **Set Up Virtual Environment**
   ```bash
   python -m venv venv
   # On Windows
   .\venv\Scripts\activate
   # On Unix/MacOS
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   # Install core dependencies
   pip install -r requirements.txt
   
   # Install module-specific dependencies
   for module in perception memory learning reasoning communication feedback integration meta; do
       pip install -r ${module}/requirements.txt
   done
   ```

4. **Environment Configuration**
   - Copy `.env.example` to `.env` in each module directory
   - Update configuration values as needed
   - Key environment variables:
     ```
     REDIS_HOST=localhost
     REDIS_PORT=6379
     MONGO_URI=mongodb://localhost:27017
     LOG_LEVEL=INFO
     ```

5. **Database Setup**
   ```bash
   # Start Redis
   docker run -d -p 6379:6379 redis
   
   # Start MongoDB
   docker run -d -p 27017:27017 mongo
   ```

6. **Run Tests**
   ```bash
   # Run all tests
   python -m pytest
   
   # Run specific module tests
   python -m pytest perception/tests/
   python -m pytest memory/tests/
   # ... etc.
   ```

7. **Start Services**

   **Using Docker Compose (Recommended for Production)**:
   ```bash
   docker-compose up --build
   ```

   **For Local Development**:
   ```bash
   # Start each service in a separate terminal
   uvicorn perception.perception_service:app --host 0.0.0.0 --port 8100 --reload
   uvicorn memory.memory_service:app --host 0.0.0.0 --port 8200 --reload
   # ... repeat for other services
   ```

## API Documentation
- Each service exposes a Swagger UI at `http://localhost:<port>/docs`
- OpenAPI specification available at `http://localhost:<port>/openapi.json`

## Development Guidelines

1. **Code Style**
   - Follow PEP 8 guidelines
   - Use type hints
   - Document functions and classes using docstrings
   - Maximum line length: 100 characters

2. **Git Workflow**
   - Create feature branches from `develop`
   - Branch naming: `feature/`, `bugfix/`, `enhancement/`
   - Submit PRs for review
   - Squash commits before merging

3. **Testing**
   - Write unit tests for new features
   - Maintain minimum 80% code coverage
   - Include integration tests for API endpoints
   - Test both success and failure cases

4. **Logging**
   - Use the configured logger
   - Include appropriate log levels
   - Add context to log messages

## Troubleshooting

1. **Common Issues**
   - Port conflicts: Check if ports are already in use
   - Database connection: Verify Redis and MongoDB are running
   - GPU issues: Check CUDA installation and GPU drivers

2. **Debug Mode**
   - Set `LOG_LEVEL=DEBUG` in `.env`
   - Use debugger in your IDE
   - Check Docker logs: `docker logs <container_name>`

## License
This project is under a dual license - free for non-commercial use, commercial use requires a license. See LICENSE.txt for details.

## Support
For technical issues or commercial licensing:
- Email: sotoyaneza@gmail.com
- GitHub Issues: [Project Issues Page]

## Contributing
1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request
