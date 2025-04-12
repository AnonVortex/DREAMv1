# HMAS SDK Documentation

## License

For commercial licensing inquiries, please contact: [info@hmas.ai]

## Overview

The HMAS SDK provides client libraries for interacting with the DREAMv1 AGI engine. SDKs are available in Python, JavaScript, and Go, offering a consistent interface across languages.

## Installation

### Python
```bash
pip install hmas-client
```

### JavaScript
```bash
npm install @hmas/client
# or
yarn add @hmas/client
```

### Go
```bash
go get github.com/hmas/client-go
```

## Quick Start

### Python
```python
from hmas.client import HMASClient

# Initialize client
client = HMASClient(
    api_key="your_api_key",
    environment="production"  # or "development", "staging"
)

# Example: Process perception input
response = client.perception.process(
    input_data={
        "type": "text",
        "content": "Hello, world!",
        "options": {
            "features": ["entities", "sentiment"]
        }
    }
)

print(response.task_id)

# Check status
status = client.perception.get_status(task_id=response.task_id)
```

### JavaScript
```javascript
import { HMASClient } from '@hmas/client';

// Initialize client
const client = new HMASClient({
    apiKey: 'your_api_key',
    environment: 'production'
});

// Example: Store memory
async function storeMemory() {
    try {
        const response = await client.memory.store({
            type: 'episodic',
            content: {
                event: 'User interaction',
                timestamp: new Date().toISOString()
            },
            metadata: {
                priority: 1,
                tags: ['interaction', 'user']
            }
        });
        console.log(response.memory_id);
    } catch (error) {
        console.error('Error:', error);
    }
}
```

### Go
```go
package main

import (
    "context"
    "log"
    
    hmas "github.com/hmas/client-go"
)

func main() {
    // Initialize client
    client := hmas.NewClient(
        hmas.WithAPIKey("your_api_key"),
        hmas.WithEnvironment("production"),
    )
    
    // Example: Train learning model
    resp, err := client.Learning.Train(context.Background(), &hmas.TrainingRequest{
        ModelType: "reinforcement",
        Parameters: map[string]interface{}{
            "learning_rate": 0.001,
            "batch_size": 32,
        },
    })
    if err != nil {
        log.Fatal(err)
    }
    
    log.Printf("Model ID: %s", resp.ModelID)
}
```

## Service Clients

### 1. Perception Client

```python
# Python example
from hmas.client import HMASClient

client = HMASClient(api_key="your_api_key")

# Process different types of input
text_response = client.perception.process_text("Hello world")
image_response = client.perception.process_image("path/to/image.jpg")
audio_response = client.perception.process_audio("path/to/audio.wav")

# Get features
features = client.perception.get_features(input_id="input_123")

# Pattern recognition
patterns = client.perception.recognize(input_data="sample_data")
```

### 2. Memory Client

```python
# Store different types of memories
memory_id = client.memory.store(
    type="episodic",
    content={"event": "User interaction"},
    metadata={"priority": 1}
)

# Retrieve and search
memory = client.memory.retrieve(memory_id)
results = client.memory.search(query="user interaction")

# Update and delete
client.memory.update(memory_id, content={"updated": True})
client.memory.delete(memory_id)
```

### 3. Learning Client

```python
# Train models
model_id = client.learning.train(
    model_type="reinforcement",
    parameters={"learning_rate": 0.001}
)

# Add experiences
client.learning.add_experience({
    "state": "current_state",
    "action": "taken_action",
    "reward": 1.0
})

# Get model status
status = client.learning.get_status(model_id)
```

### 4. Reasoning Client

```python
# Perform inference
inference_id = client.reasoning.infer(
    context={"data": "sample"},
    options={"depth": 3}
)

# Make decisions
decision = client.reasoning.decide(
    situation="scenario",
    options=["option1", "option2"]
)

# Get explanations
explanation = client.reasoning.get_explanation(decision_id)
```

### 5. Communication Client

```python
# Send messages
message_id = client.communication.send(
    receiver="agent_123",
    content="Hello!",
    priority="high"
)

# Receive messages
messages = client.communication.receive()

# Check message status
status = client.communication.get_status(message_id)
```

### 6. Feedback Client

```python
# Submit feedback
feedback_id = client.feedback.submit({
    "type": "performance",
    "rating": 4.5,
    "comments": "Good response"
})

# Get metrics
metrics = client.feedback.get_metrics()

# Get analysis
analysis = client.feedback.get_analysis(metric_id)
```

### 7. Integration Client

```python
# Connect to external system
connection_id = client.integration.connect(
    system_type="external_api",
    config={"url": "https://api.example.com"}
)

# Transform data
transformed = client.integration.transform(
    data={"raw": "data"},
    schema_id="schema_123"
)
```

## Error Handling

```python
from hmas.exceptions import HMASError, APIError, AuthError

try:
    result = client.perception.process(input_data)
except AuthError as e:
    print("Authentication failed:", e)
except APIError as e:
    print("API error:", e.status_code, e.message)
except HMASError as e:
    print("General error:", e)
```

## Async Support

### Python
```python
import asyncio
from hmas.client.async_client import AsyncHMASClient

async def main():
    client = AsyncHMASClient(api_key="your_api_key")
    response = await client.perception.process(input_data)
    print(response)

asyncio.run(main())
```

### JavaScript
```javascript
// Already async by default
const response = await client.memory.store(data);
```

## WebSocket Support

```python
from hmas.client import HMASClient

client = HMASClient(api_key="your_api_key")

# Subscribe to updates
def on_update(update):
    print("Received update:", update)

client.subscribe_updates(callback=on_update)

# Subscribe to events
def on_event(event):
    print("Received event:", event)

client.subscribe_events(callback=on_event)
```

## GraphQL Support

```python
# Execute GraphQL query
response = client.graphql.execute("""
    query {
        agent(id: "agent_123") {
            id
            status
            currentTask {
                id
                type
            }
        }
    }
""")
```

## Configuration

```python
client = HMASClient(
    api_key="your_api_key",
    environment="production",
    timeout=30,
    max_retries=3,
    base_url="https://api.hmas.ai",
    version="v1"
)
```

## Rate Limiting

The SDK automatically handles rate limiting by implementing exponential backoff and retry logic.

```python
client = HMASClient(
    api_key="your_api_key",
    rate_limit=1000,  # requests per minute
    burst_limit=50    # requests per second
)
```

## Logging

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)
client = HMASClient(api_key="your_api_key")
```

## Testing

```python
from hmas.client.testing import MockHMASClient

# Create mock client for testing
client = MockHMASClient()

# Set up mock responses
client.perception.mock_response(
    method="process",
    response={"task_id": "mock_123"}
)

# Use in tests
response = client.perception.process(input_data)
assert response.task_id == "mock_123"
```

## Additional Resources

- [API Documentation](./api.md)
- [Example Applications](../examples/)
- [Changelog](./changelog.md)
- [Contributing Guide](./contributing.md) 