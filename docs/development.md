# HMAS Development Guide

## Overview

This guide provides comprehensive information for developers working on the Hierarchical Multi-Agent System (HMAS). It covers development setup, coding standards, testing procedures, and best practices.

## Development Environment Setup

### Prerequisites
- Python 3.8+
- Docker & Docker Compose
- Git
- Visual Studio Code (recommended)
- CUDA toolkit (for GPU support)

### Initial Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/hmas.git
   cd hmas
   ```

2. Create virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # or `venv\Scripts\activate` on Windows
   pip install -r requirements.txt
   ```

3. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

4. Copy environment template:
   ```bash
   cp .env.example .env
   # Edit .env with your settings
   ```

## Project Structure

### Core Modules
```
hmas/
├── perception/          # Input processing
├── memory/             # Knowledge storage
├── learning/           # Learning systems
├── reasoning/          # Logic and inference
├── communication/      # Agent messaging
└── integration/        # System coordination
```

### Supporting Modules
```
hmas/
├── monitoring/         # System metrics
├── security/          # Access control
├── deployment/        # Deployment configs
└── tests/             # Test suites
```

## Development Guidelines

### Code Style
- Follow PEP 8 guidelines
- Use type hints
- Document all functions and classes
- Keep functions focused and small
- Use meaningful variable names

### Example:
```python
from typing import List, Optional
from datetime import datetime

class DataProcessor:
    """Processes input data for the perception module."""
    
    def __init__(self, config: dict):
        """Initialize processor with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.initialized = False
    
    def process_batch(
        self,
        data: List[dict],
        batch_size: Optional[int] = None
    ) -> List[dict]:
        """Process a batch of data.
        
        Args:
            data: List of data items
            batch_size: Optional batch size
            
        Returns:
            List of processed items
        """
        # Implementation
```

### Testing
1. Unit Tests:
   ```python
   def test_data_processor():
       processor = DataProcessor({"mode": "test"})
       result = processor.process_batch([{"data": "test"}])
       assert len(result) == 1
       assert result[0]["processed"] == True
   ```

2. Integration Tests:
   ```python
   async def test_memory_integration():
       memory = MemoryService()
       await memory.initialize()
       await memory.store({"key": "value"})
       result = await memory.retrieve("key")
       assert result == {"key": "value"}
   ```

### Error Handling
```python
from fastapi import HTTPException

async def process_request(data: dict):
    try:
        result = await process_data(data)
        return result
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error"
        )
```

## Module Development

### Creating a New Module
1. Create module directory structure:
   ```
   new_module/
   ├── __init__.py
   ├── new_module_service.py
   ├── new_module_main.py
   ├── config.py
   ├── logging.conf
   ├── requirements.txt
   ├── Dockerfile
   ├── README.md
   └── tests/
   ```

2. Implement service class:
   ```python
   class NewModuleService:
       def __init__(self):
           self.config = load_config()
           self.logger = setup_logging()
           
       async def initialize(self):
           """Initialize service resources."""
           
       async def cleanup(self):
           """Cleanup resources."""
           
       async def process_request(self, data: dict):
           """Process incoming request."""
   ```

3. Create FastAPI application:
   ```python
   app = FastAPI(title="New Module Service")
   
   @app.on_event("startup")
   async def startup():
       service = NewModuleService()
       await service.initialize()
       app.state.service = service
   
   @app.post("/process")
   async def process(data: dict):
       return await app.state.service.process_request(data)
   ```

### Module Integration
1. Register with Integration Module:
   ```python
   await register_service({
       "name": "new_module",
       "endpoint": "http://new_module:8000",
       "health_check": "/health"
   })
   ```

2. Implement health checks:
   ```python
   @app.get("/health")
   async def health_check():
       return {
           "status": "healthy",
           "timestamp": datetime.now().isoformat()
       }
   ```

## Deployment

### Local Development
1. Start services:
   ```bash
   docker-compose up -d
   ```

2. View logs:
   ```bash
   docker-compose logs -f [service_name]
   ```

3. Rebuild service:
   ```bash
   docker-compose build [service_name]
   docker-compose up -d [service_name]
   ```

### Production Deployment
1. Build images:
   ```bash
   docker build -t hmas/[module] .
   ```

2. Push to registry:
   ```bash
   docker push hmas/[module]:version
   ```

3. Deploy to Kubernetes:
   ```bash
   kubectl apply -f k8s/
   ```

## Monitoring

### Prometheus Metrics
```python
from prometheus_fastapi_instrumentator import Instrumentator

# Initialize metrics
Instrumentator().instrument(app).expose(app)

# Custom metrics
from prometheus_client import Counter
request_counter = Counter(
    "requests_total",
    "Total requests processed"
)
```

### Logging
```python
import logging.config
import yaml

with open("logging.conf") as f:
    config = yaml.safe_load(f)
    logging.config.dictConfig(config)

logger = logging.getLogger(__name__)
```

## Security

### Authentication
```python
from fastapi.security import OAuth2PasswordBearer

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

@app.post("/protected")
async def protected_route(
    token: str = Depends(oauth2_scheme)
):
    # Verify token
    # Process request
```

### Rate Limiting
```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.post("/api")
@limiter.limit("100/minute")
async def rate_limited_route():
    # Process request
```

## Best Practices

### Configuration Management
- Use environment variables for sensitive data
- Store configs in version control
- Use separate configs for development/production

### Error Handling
- Log all errors with context
- Return appropriate HTTP status codes
- Include helpful error messages

### Performance
- Use async/await for I/O operations
- Implement caching where appropriate
- Monitor resource usage

### Documentation
- Keep README files updated
- Document API endpoints
- Include usage examples

## Troubleshooting

### Common Issues
1. Connection errors:
   - Check service availability
   - Verify network settings
   - Check firewall rules

2. Performance issues:
   - Monitor resource usage
   - Check for memory leaks
   - Review database queries

3. Authentication issues:
   - Verify token validity
   - Check permissions
   - Review security logs

### Debugging
1. Enable debug logging:
   ```python
   logging.getLogger().setLevel(logging.DEBUG)
   ```

2. Use debugger:
   ```python
   import pdb; pdb.set_trace()
   ```

3. Monitor metrics:
   ```bash
   curl localhost:8000/metrics
   ```

## Contributing

### Pull Request Process
1. Create feature branch
2. Write tests
3. Update documentation
4. Submit PR

### Code Review
- Review for style
- Check test coverage
- Verify documentation
- Test functionality

## Coding Standards

### Python Style Guide
- Follow PEP 8
- Use Black for formatting (line length: 88)
- Use isort for import sorting
- Use type hints
- Document using Google style docstrings

### Example Module
```python
"""Module docstring explaining purpose and functionality."""

from typing import List, Optional

import numpy as np
import torch
from pydantic import BaseModel

class ExampleConfig(BaseModel):
    """Configuration class for example module.
    
    Attributes:
        param1: Description of parameter 1
        param2: Description of parameter 2
    """
    param1: str
    param2: Optional[int] = None

class ExampleClass:
    """Example class demonstrating coding standards.
    
    Args:
        config: Configuration object
        name: Instance name
    """
    
    def __init__(self, config: ExampleConfig, name: str) -> None:
        self.config = config
        self.name = name
        self._private_var = None
    
    def process_data(self, data: np.ndarray) -> List[float]:
        """Process input data and return results.
        
        Args:
            data: Input data array
            
        Returns:
            List of processed values
            
        Raises:
            ValueError: If data is empty
        """
        if data.size == 0:
            raise ValueError("Empty data array")
        
        return [float(x) for x in data]

## Testing

### Test Structure
```
tests/
├── unit/              # Unit tests
├── integration/       # Integration tests
├── performance/       # Performance tests
└── conftest.py       # Test fixtures
```

### Writing Tests
```python
import pytest
from hmas.example import ExampleClass, ExampleConfig

@pytest.fixture
def example_instance():
    """Fixture providing an ExampleClass instance."""
    config = ExampleConfig(param1="test", param2=42)
    return ExampleClass(config, "test_instance")

def test_process_data(example_instance):
    """Test data processing functionality."""
    input_data = np.array([1.0, 2.0, 3.0])
    result = example_instance.process_data(input_data)
    assert len(result) == 3
    assert all(isinstance(x, float) for x in result)

@pytest.mark.parametrize("input_data", [
    np.array([]),
    np.array([1.0]),
    np.array([1.0, 2.0])
])
def test_process_data_validation(example_instance, input_data):
    """Test data validation with different inputs."""
    if input_data.size == 0:
        with pytest.raises(ValueError):
            example_instance.process_data(input_data)
    else:
        result = example_instance.process_data(input_data)
        assert len(result) == len(input_data)
```

### Running Tests
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/unit/test_example.py

# Run tests with coverage
pytest --cov=hmas tests/

# Generate coverage report
coverage html
```

## Documentation

### Code Documentation
- Use Google style docstrings
- Document all public classes and methods
- Include type hints
- Provide usage examples

### API Documentation
- Use OpenAPI/Swagger
- Document all endpoints
- Include request/response examples
- Document error responses

### Building Documentation
```bash
# Generate API documentation
python scripts/generate_api_docs.py

# Build documentation site
mkdocs build
```

## Git Workflow

### Branch Naming
- feature/: New features
- bugfix/: Bug fixes
- hotfix/: Critical fixes
- release/: Release preparation

### Commit Messages
```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

Types:
- feat: New feature
- fix: Bug fix
- docs: Documentation
- style: Formatting
- refactor: Code restructuring
- test: Adding tests
- chore: Maintenance

### Pull Requests
- Create feature branch
- Write descriptive PR title
- Include issue references
- Add tests and documentation
- Request review from team members

## Debugging

### Logging
```python
import logging

logger = logging.getLogger(__name__)

def example_function():
    logger.debug("Debug information")
    logger.info("Processing started")
    try:
        # Process
        logger.info("Processing completed")
    except Exception as e:
        logger.error("Processing failed: %s", str(e))
        raise
```

### Debugging Tools
- pdb/ipdb for interactive debugging
- logging for tracking execution
- pytest --pdb for test debugging
- VS Code/PyCharm debuggers

## Performance Optimization

### Profiling
```bash
# Profile execution
python -m cProfile -o output.prof script.py

# Analyze profile
python -m pstats output.prof

# Visual profiling
python -m snakeviz output.prof
```

### Memory Profiling
```bash
# Track memory usage
python -m memory_profiler script.py

# Generate memory usage graph
mprof run script.py
mprof plot
```

## Security

### Code Security
- Use security linters
- Regular dependency updates
- Input validation
- Proper error handling

### Running Security Checks
```bash
# Run security checks
bandit -r hmas/

# Check dependencies
safety check
```

## Continuous Integration

### GitHub Actions
```yaml
name: CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.9
      - name: Install dependencies
        run: |
          pip install -r requirements-dev.txt
      - name: Run tests
        run: |
          pytest --cov=hmas tests/
      - name: Upload coverage
        uses: codecov/codecov-action@v2
```

## Release Process

1. Version Update
```bash
# Update version
bump2version patch  # or minor, major
```

2. Update Changelog
```markdown
## [1.0.1] - 2024-03-21
### Added
- New feature X
### Fixed
- Bug in component Y
```

3. Create Release
```bash
git tag -a v1.0.1 -m "Release version 1.0.1"
git push origin v1.0.1
```

## Troubleshooting

### Common Issues
1. Environment Setup
   - Check Python version
   - Verify virtual environment
   - Update dependencies

2. Testing Issues
   - Check test dependencies
   - Verify test data
   - Review test logs

3. Build Issues
   - Check build logs
   - Verify dependencies
   - Review configuration 