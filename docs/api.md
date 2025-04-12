# API Documentation

## License

For commercial licensing inquiries, please contact: [info@hmas.ai]

## Overview

This document describes the API endpoints for the HMAS (Hierarchical Multi-Agent System) with DREAMv1 AGI engine. All services follow RESTful principles and use JSON for request/response payloads unless otherwise specified.

## Base URLs

- Development: `http://localhost:{PORT}`
- Staging: `https://staging.api.hmas.ai`
- Production: `https://api.hmas.ai`

## Authentication

All API endpoints require authentication using JWT tokens. Include the token in the Authorization header:

```
Authorization: Bearer <your_token>
```

## Common Headers

```
Content-Type: application/json
Accept: application/json
X-Request-ID: <unique_request_id>
X-API-Version: v1
```

## Service Endpoints

### 1. Perception Service (Port: 8100)

#### Input Processing
```
POST /v1/perception/process
GET /v1/perception/status/{task_id}
GET /v1/perception/features/{input_id}
```

#### Pattern Recognition
```
POST /v1/perception/recognize
GET /v1/perception/patterns/{pattern_id}
```

### 2. Memory Service (Port: 8200)

#### Knowledge Management
```
POST /v1/memory/store
GET /v1/memory/retrieve/{memory_id}
PUT /v1/memory/update/{memory_id}
DELETE /v1/memory/delete/{memory_id}
```

#### Pattern Storage
```
POST /v1/memory/patterns/store
GET /v1/memory/patterns/search
```

### 3. Learning Service (Port: 8300)

#### Model Management
```
POST /v1/learning/train
GET /v1/learning/status/{model_id}
PUT /v1/learning/update/{model_id}
```

#### Experience Processing
```
POST /v1/learning/experience/add
GET /v1/learning/experience/batch
```

### 4. Reasoning Service (Port: 8400)

#### Inference
```
POST /v1/reasoning/infer
GET /v1/reasoning/status/{inference_id}
```

#### Decision Making
```
POST /v1/reasoning/decide
GET /v1/reasoning/explanation/{decision_id}
```

### 5. Communication Service (Port: 8500)

#### Message Management
```
POST /v1/communication/send
GET /v1/communication/receive
GET /v1/communication/status/{message_id}
```

#### Protocol Management
```
GET /v1/communication/protocols
PUT /v1/communication/protocols/{protocol_id}
```

### 6. Feedback Service (Port: 8600)

#### Performance Monitoring
```
POST /v1/feedback/submit
GET /v1/feedback/metrics
GET /v1/feedback/analysis/{metric_id}
```

#### Adaptation
```
POST /v1/feedback/adapt
GET /v1/feedback/status/{adaptation_id}
```

### 7. Integration Service (Port: 8700)

#### External Systems
```
POST /v1/integration/connect
GET /v1/integration/status/{connection_id}
PUT /v1/integration/update/{connection_id}
```

#### Data Transformation
```
POST /v1/integration/transform
GET /v1/integration/schema/{transform_id}
```

### 8. Specialized Service (Port: 8800)

#### Custom Processing
```
POST /v1/specialized/process
GET /v1/specialized/status/{process_id}
```

#### Extension Management
```
POST /v1/specialized/extensions/register
GET /v1/specialized/extensions/list
```

## Response Formats

### Success Response
```json
{
    "status": "success",
    "data": {
        // Response data
    },
    "metadata": {
        "timestamp": "2024-03-21T12:00:00Z",
        "request_id": "req-123",
        "version": "1.0.0"
    }
}
```

### Error Response
```json
{
    "status": "error",
    "error": {
        "code": "ERROR_CODE",
        "message": "Error description",
        "details": {
            // Additional error details
        }
    },
    "metadata": {
        "timestamp": "2024-03-21T12:00:00Z",
        "request_id": "req-123",
        "version": "1.0.0"
    }
}
```

## Rate Limiting

- Default rate limit: 1000 requests per minute
- Burst limit: 50 requests per second
- Headers:
  - X-RateLimit-Limit
  - X-RateLimit-Remaining
  - X-RateLimit-Reset

## Versioning

API versioning is handled through:
1. URL path (/v1/, /v2/, etc.)
2. Accept header (application/vnd.hmas.v1+json)

## WebSocket Endpoints

### Real-time Updates
```
ws://api.hmas.ai/v1/updates
```

### Event Streaming
```
ws://api.hmas.ai/v1/events
```

## GraphQL API

### Endpoint
```
https://api.hmas.ai/v1/graphql
```

### Introspection
```
https://api.hmas.ai/v1/graphql/schema
```

## SDK Support

- Python SDK: `pip install hmas-client`
- JavaScript SDK: `npm install @hmas/client`
- Go SDK: `go get github.com/hmas/client-go`

## Additional Resources

- [API Changelog](./changelog.md)
- [SDK Documentation](./sdk.md)
- [Example Code](../examples/)
- [Postman Collection](./postman/) 
