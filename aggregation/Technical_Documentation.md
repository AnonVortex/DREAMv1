# HMAS Aggregation Module - Technical Documentation

## 1. Overview
The Aggregation module is a critical component in the HMAS pipeline, responsible for synthesizing evaluation results from various system stages into coherent, final decisions. It implements sophisticated algorithms for confidence scoring, trend analysis, and evidence gathering to ensure reliable decision-making.

## 2. Architecture

### 2.1 Core Components
- **AggregationEngine**: Central component handling aggregation logic
- **EvaluationResult**: Data model for individual evaluations
- **AggregationInput**: Input model with archive and current evaluation
- **AggregationResult**: Output model with decision, confidence, and trends

### 2.2 Dependencies
- **FastAPI**: Web framework for API endpoints
- **Redis**: Used for readiness checks and caching
- **Prometheus**: Metrics collection and exposure
- **NumPy**: Numerical computations and trend analysis
- **Pydantic**: Data validation and serialization

## 3. Implementation Details

### 3.1 Confidence Calculation
The confidence scoring algorithm combines multiple factors:

1. **Base Confidence**:
   - Derived from the current evaluation's confidence score
   - Default value: 0.5 if not provided

2. **Historical Consistency**:
   - Analyzes last 10 evaluations of the same stage
   - Calculates similarity scores between current and historical evaluations
   - Weighted average: 70% current confidence, 30% historical consistency

3. **Similarity Calculation**:
   ```python
   similarity = 1.0 - average_normalized_difference
   normalized_diff = diff / max(abs(value1), abs(value2), epsilon)
   ```

### 3.2 Trend Detection
Implements linear regression for trend analysis:

1. **Data Preparation**:
   - Groups evaluations by type
   - Extracts numerical metrics
   - Uses last 5 entries for trend calculation

2. **Trend Classification**:
   - Calculates slope using numpy.polyfit
   - Classifies as:
     - "increasing": slope > 0.01
     - "decreasing": slope < -0.01
     - "stable": |slope| ≤ 0.01

### 3.3 Supporting Evidence
Gathers evidence to support decisions:

1. **Selection**:
   - Uses last 5 entries from archive
   - Calculates relevance using similarity scoring
   - Includes timestamps for temporal context

2. **History Management**:
   - Maintains rolling history of decisions
   - Configurable history size (default: 100)
   - Automatic pruning of old entries

## 4. Performance Considerations

### 4.1 Optimization Techniques
- Efficient metric extraction using type checking
- Vectorized operations for trend calculations
- Configurable history limits to prevent memory bloat

### 4.2 Rate Limiting
- Default: 10 requests/minute per IP
- Configurable through environment variables
- Uses sliding window algorithm

### 4.3 Caching Strategy
- Redis-based caching for frequent calculations
- Configurable cache TTL
- Automatic cache invalidation on updates

## 5. Integration Guidelines

### 5.1 Module Integration
```python
from aggregation.aggregation_main import AggregationEngine

# Initialize engine
engine = AggregationEngine()

# Prepare input
input_data = AggregationInput(
    archive=[...],
    query_result={...},
    context={...}
)

# Get aggregated result
result = engine.aggregate(input_data)
```

### 5.2 Metrics Integration
1. **Prometheus Configuration**:
   ```python
   from prometheus_fastapi_instrumentator import Instrumentator
   
   Instrumentator().instrument(app).expose(app)
   ```

2. **Custom Metrics**:
   ```python
   from prometheus_client import Counter, Histogram
   
   AGGREGATION_COUNTER = Counter(
       "aggregation_total",
       "Total aggregations performed"
   )
   ```

### 5.3 Environment Configuration
Required environment variables:
```bash
AGGREGATION_HOST=0.0.0.0
AGGREGATION_PORT=8500
REDIS_URL=redis://localhost:6379
LOG_LEVEL=INFO
RATE_LIMIT="10/minute"
```

## 6. Error Handling

### 6.1 Common Errors
1. **Input Validation**:
   - Empty archive
   - Invalid evaluation format
   - Missing required fields

2. **Processing Errors**:
   - Calculation failures
   - Trend detection errors
   - History management issues

### 6.2 Error Response Format
```json
{
    "detail": "Error description",
    "error_code": "ERROR_TYPE",
    "timestamp": "ISO datetime"
}
```

## 7. Monitoring and Maintenance

### 7.1 Health Checks
- Basic health: `/health`
- Dependency check: `/ready`
- Metrics endpoint: `/metrics`

### 7.2 Logging
- Structured logging using logging.conf
- Log levels: DEBUG, INFO, WARNING, ERROR
- Request/response logging
- Error tracking

### 7.3 Metrics Collection
- Request duration
- Success/failure rates
- Resource utilization
- Trend detection accuracy

## 8. Security Considerations

### 8.1 Input Validation
- Strict type checking
- Size limits on input data
- Sanitization of string inputs

### 8.2 Rate Limiting
- Per-IP rate limiting
- Configurable limits
- Automatic blocking of excessive requests

### 8.3 Dependencies
- Regular security updates
- Vulnerability scanning
- Dependency version pinning

## 9. Future Improvements

### 9.1 Planned Features
- Advanced anomaly detection
- Machine learning-based confidence scoring
- Real-time visualization
- Enhanced trend analysis

### 9.2 Performance Optimizations
- Caching improvements
- Parallel processing
- Database integration
- Batch processing support

## 10. Data Flow
- **Input**: Expects a JSON payload structured as:
  ```json
  {
      "archive": [
          {"stage": "meta", "evaluation": {...}},
          ...
      ],
      "query_result": {"stage": "meta", "evaluation": {...}}
  }
