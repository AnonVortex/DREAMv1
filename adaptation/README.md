# HMAS Adaptation Service

The HMAS Adaptation Service is a critical component of the Hierarchical Multi-Agent System that provides dynamic adaptation, resource management, and evolutionary capabilities to ensure optimal system performance and resilience.

## Features

### Resource Management
- Real-time monitoring of system resources (CPU, Memory, Disk, Network)
- Resource usage trend analysis
- Automated resource allocation recommendations

### Dynamic Scaling
- Automatic agent scaling based on load and resource metrics
- Container-based scaling using Docker
- Resource limit and reservation management

### Load Balancing
- Intelligent task distribution across agents
- Task-agent compatibility scoring
- Load optimization algorithms

### Self-Modification
- Dynamic architecture modifications
- Component addition/removal
- Connection management
- Resource reallocation

### Evolutionary Architecture
- Genetic algorithm-based architecture optimization
- Constraint-based evolution
- Fitness evaluation and selection

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Copy `.env.example` to `.env` and configure environment variables
4. Configure logging in `logging.conf`

## Configuration

### Environment Variables
- `ENVIRONMENT`: Development/Production environment
- `LOG_LEVEL`: Logging verbosity
- `DOCKER_HOST`: Docker daemon socket
- `API_PORT`: Service port number
- `METRICS_INTERVAL`: Resource metrics collection interval
- `MAX_HISTORY_POINTS`: Maximum metrics history points

### Logging Configuration
The `logging.conf` file contains logging configuration for different components:
- Console logging
- File logging
- Log rotation
- Format configuration

## API Endpoints

### Resource Management
- `POST /resources/metrics`: Collect current resource metrics
- `GET /resources/trends`: Get resource usage trends

### Dynamic Scaling
- `POST /scale/agent`: Scale specific agent
- `POST /scale/create`: Create new agent

### Load Balancing
- `POST /load/balance`: Get agent for task execution
- `GET /load/status`: Get current load distribution

### Self-Modification
- `POST /modify/architecture`: Apply architecture modification
- `GET /modify/status`: Get modification status

### Evolutionary Architecture
- `POST /evolve/architecture`: Start architecture evolution
- `GET /evolve/status`: Get evolution progress

## Testing

Run tests using pytest:
```bash
pytest tests/
```

### Test Coverage
- Unit tests for all components
- Integration tests for API endpoints
- Mock tests for Docker interactions
- Resource management tests
- Load balancing algorithm tests

## Monitoring

The service includes a Streamlit-based monitoring GUI:
```bash
streamlit run gui.py
```

### GUI Features
1. System Overview
   - Real-time resource metrics
   - Historical trends
   - Auto-refresh capability

2. Agents Management
   - Active agents list
   - Load distribution
   - Performance metrics

3. Architecture Visualization
   - Component network graph
   - Resource allocation view
   - Relationship mapping

4. Evolution Tracking
   - Evolution parameters
   - Progress monitoring
   - Results visualization

## Development

### Adding New Features
1. Create feature branch
2. Implement changes
3. Add tests
4. Update documentation
5. Submit pull request

### Code Style
- Follow PEP 8 guidelines
- Use type hints
- Document all public methods
- Include docstring examples

## Contributing
1. Fork the repository
2. Create feature branch
3. Commit changes
4. Push to branch
5. Create pull request

## License
MIT License - see LICENSE file for details 