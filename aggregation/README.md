# HMAS Aggregation Module

## Overview
The Aggregation Module is a critical component of the HMAS system that consolidates evaluation results from various stages into coherent, final decisions. It provides sophisticated decision-making capabilities with confidence scoring, trend analysis, and supporting evidence.

## Features
- **Intelligent Aggregation**: Combines current and historical data for informed decisions
- **Confidence Scoring**: Calculates confidence levels based on multiple factors
- **Trend Detection**: Identifies patterns and trends in evaluation metrics
- **Evidence Collection**: Gathers supporting evidence for decisions
- **Rate Limiting**: Prevents system overload
- **Prometheus Metrics**: Real-time monitoring capabilities
- **Health Checks**: Built-in service health monitoring

## Quick Start

### Prerequisites
- Python 3.9+
- Redis
- Docker (optional)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/hmas.git
cd hmas/aggregation
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

4. Start Redis:
```bash
docker run -d -p 6379:6379 redis
```

5. Run the service:
```bash
python -m aggregation_main
```

### Docker Deployment

1. Build the image:
```bash
docker build -t hmas/aggregation:latest .
```

2. Run the container:
```bash
docker run -d \
    -p 8500:8500 \
    -e REDIS_URL=redis://redis:6379 \
    --name hmas-aggregation \
    hmas/aggregation:latest
```

## Usage Examples

### Basic Aggregation Request
```python
import requests
import json

url = "http://localhost:8500/aggregate"
payload = {
    "archive": [
        {
            "stage": "meta",
            "evaluation": {
                "accuracy": 0.85,
                "confidence": 0.75,
                "latency": 120
            },
            "timestamp": "2024-03-15T10:30:00Z"
        }
    ],
    "query_result": {
        "stage": "meta",
        "evaluation": {
            "accuracy": 0.87,
            "confidence": 0.78,
            "latency": 115
        }
    }
}

response = requests.post(url, json=payload)
result = response.json()
print(json.dumps(result, indent=2))
```

### Python Client Integration
```python
from aggregation.aggregation_main import AggregationEngine
from aggregation.models import AggregationInput

# Initialize engine
engine = AggregationEngine()

# Prepare input
input_data = AggregationInput(
    archive=[...],
    query_result={
        "stage": "meta",
        "evaluation": {
            "metric1": 0.85,
            "metric2": 120
        }
    }
)

# Get result
result = engine.aggregate(input_data)
print(f"Decision confidence: {result.confidence_score}")
print(f"Detected trends: {result.trends}")
```

## Configuration

### Environment Variables
| Variable | Description | Default |
|----------|-------------|---------|
| AGGREGATION_HOST | Service host | 0.0.0.0 |
| AGGREGATION_PORT | Service port | 8500 |
| REDIS_URL | Redis connection URL | redis://localhost:6379 |
| LOG_LEVEL | Logging level | INFO |
| RATE_LIMIT | Rate limit per minute | 10/minute |

### Logging Configuration
The service uses a standard logging configuration file (`logging.conf`). Customize it for your needs:

```ini
[loggers]
keys=root,aggregation

[handlers]
keys=console,file

[formatters]
keys=standard

[logger_root]
level=INFO
handlers=console

[logger_aggregation]
level=INFO
handlers=console,file
qualname=aggregation
propagate=0
```

## Monitoring

### Health Checks
- **Basic Health**: `GET /health`
- **Readiness**: `GET /ready`
- **Metrics**: `GET /metrics`

### Prometheus Metrics
The service exposes metrics at `/metrics`, including:
- Request counts and latencies
- Aggregation processing times
- Error rates
- Resource usage

## Development

### Running Tests
```bash
# Install test dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Run with coverage
pytest --cov=aggregation tests/
```

### Code Style
The project follows PEP 8 guidelines. Format your code using:
```bash
black aggregation/
flake8 aggregation/
```

## Documentation
- [Technical Documentation](Technical_Documentation.md)
- [API Reference](../docs/api/AggregationAPI.md)
- [Development Guide](../docs/development.md)

## Contributing
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

## License
This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.
