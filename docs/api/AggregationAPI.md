# Aggregation Module API Reference

## Overview
The Aggregation Module is responsible for consolidating evaluation results from various stages of the system into coherent, final decisions. It provides confidence scoring, trend analysis, and supporting evidence for each aggregation.

## Base URL
`http://localhost:8500`

## Authentication
Currently, the API does not require authentication. However, it is rate-limited to 10 requests per minute per client IP.

## Endpoints

### POST /aggregate
Aggregates evaluation results into a final decision with confidence scoring and trend analysis.

#### Request
- **Content-Type**: `application/json`
- **Rate Limit**: 10 requests per minute

##### Request Body Schema
```json
{
    "archive": [
        {
            "stage": "string",
            "evaluation": {
                "metric_name": "number|string|boolean",
                ...
            },
            "timestamp": "string (ISO format, optional)"
        }
    ],
    "query_result": {
        "stage": "string",
        "evaluation": {
            "metric_name": "number|string|boolean",
            ...
        }
    },
    "context": {
        "additional_info": "any",
        ... (optional)
    }
}
```

##### Example Request
```json
{
    "archive": [
        {
            "stage": "meta",
            "evaluation": {
                "accuracy": 0.85,
                "confidence": 0.75,
                "latency": 120,
                "resource_usage": 0.65
            },
            "timestamp": "2024-03-15T10:30:00Z"
        }
    ],
    "query_result": {
        "stage": "meta",
        "evaluation": {
            "accuracy": 0.87,
            "confidence": 0.78,
            "latency": 115,
            "resource_usage": 0.62
        }
    }
}
```

#### Response
- **Content-Type**: `application/json`

##### Response Schema
```json
{
    "final_decision": {
        "metric_name": "number|string|boolean",
        ...
    },
    "confidence_score": "number (0-1)",
    "supporting_evidence": [
        {
            "stage": "string",
            "timestamp": "string (ISO format)",
            "relevance": "number (0-1)"
        }
    ],
    "trends": {
        "evaluation_type": {
            "metric_name": {
                "slope": "number",
                "direction": "string (increasing|decreasing|stable)"
            }
        }
    },
    "timestamp": "string (ISO format)"
}
```

##### Example Response
```json
{
    "final_decision": {
        "accuracy": 0.87,
        "confidence": 0.78,
        "latency": 115,
        "resource_usage": 0.62
    },
    "confidence_score": 0.82,
    "supporting_evidence": [
        {
            "stage": "meta",
            "timestamp": "2024-03-15T10:30:00Z",
            "relevance": 0.95
        }
    ],
    "trends": {
        "meta": {
            "accuracy": {
                "slope": 0.02,
                "direction": "increasing"
            },
            "latency": {
                "slope": -5.0,
                "direction": "decreasing"
            }
        }
    },
    "timestamp": "2024-03-15T10:35:00Z"
}
```

##### Error Responses
- **400 Bad Request**: Empty archive provided
- **422 Unprocessable Entity**: Invalid request payload
- **429 Too Many Requests**: Rate limit exceeded
- **500 Internal Server Error**: Server-side error
- **503 Service Unavailable**: Service not ready

### GET /health
Health check endpoint to verify service status.

#### Response
```json
{
    "status": "ok"
}
```

### GET /ready
Readiness check endpoint that verifies all dependencies (including Redis) are available.

#### Response
```json
{
    "status": "ready"
}
```

#### Error Response
```json
{
    "detail": "Service not ready: {error_message}"
}
```

## Metrics
The module exposes Prometheus metrics at the `/metrics` endpoint, including:
- Request counts and latencies
- Aggregation processing times
- Error rates
- Resource usage

## Rate Limiting
- Default rate limit: 10 requests per minute per IP
- Rate limit headers are included in responses:
  - `X-RateLimit-Limit`: Request limit per window
  - `X-RateLimit-Remaining`: Remaining requests in current window
  - `X-RateLimit-Reset`: Time until the rate limit resets