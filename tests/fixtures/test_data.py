from datetime import datetime, timedelta
import json
import uuid

# Input Data Fixtures
TEXT_INPUTS = [
    {
        "type": "text",
        "content": "Hello, how can I help you today?",
        "metadata": {"source": "user", "language": "en"}
    },
    {
        "type": "text",
        "content": "What's the weather like?",
        "metadata": {"source": "user", "language": "en"}
    }
]

IMAGE_INPUTS = [
    {
        "type": "image",
        "content": "base64_encoded_image_data",
        "metadata": {
            "format": "jpeg",
            "dimensions": {"width": 800, "height": 600}
        }
    }
]

AUDIO_INPUTS = [
    {
        "type": "audio",
        "content": "base64_encoded_audio_data",
        "metadata": {
            "format": "wav",
            "duration": 10.5,
            "sample_rate": 44100
        }
    }
]

# Memory Fixtures
MEMORY_SAMPLES = [
    {
        "type": "episodic",
        "content": {
            "event": "user_interaction",
            "timestamp": datetime.now().isoformat(),
            "details": {"action": "greeting", "response": "hello"}
        },
        "metadata": {
            "priority": 1,
            "tags": ["interaction", "greeting"],
            "source": "test"
        }
    },
    {
        "type": "semantic",
        "content": {
            "concept": "weather",
            "attributes": ["temperature", "conditions", "forecast"],
            "relationships": ["affects_mood", "affects_activities"]
        },
        "metadata": {
            "priority": 2,
            "tags": ["knowledge", "weather"],
            "source": "learning"
        }
    }
]

# Learning Model Fixtures
MODEL_CONFIGS = [
    {
        "name": "conversation",
        "type": "transformer",
        "parameters": {
            "num_layers": 6,
            "hidden_size": 768,
            "num_heads": 12,
            "dropout": 0.1
        }
    },
    {
        "name": "image_recognition",
        "type": "cnn",
        "parameters": {
            "num_layers": 50,
            "channels": [64, 128, 256, 512],
            "dropout": 0.2
        }
    }
]

TRAINING_DATA = [
    {
        "input": "Hello",
        "output": "Hi there!",
        "metadata": {"context": "greeting"}
    },
    {
        "input": "How's the weather?",
        "output": "I'll check the weather for you.",
        "metadata": {"context": "weather_query"}
    }
]

# Reasoning Fixtures
INFERENCE_CONTEXTS = [
    {
        "input": TEXT_INPUTS[0],
        "perception": {
            "intent": "greeting",
            "sentiment": "positive",
            "entities": []
        },
        "memory": {
            "relevant_interactions": [],
            "known_patterns": ["standard_greeting"]
        }
    },
    {
        "input": TEXT_INPUTS[1],
        "perception": {
            "intent": "weather_query",
            "sentiment": "neutral",
            "entities": []
        },
        "memory": {
            "relevant_interactions": [],
            "known_patterns": ["weather_request"]
        }
    }
]

# Communication Fixtures
MESSAGE_TEMPLATES = [
    {
        "type": "response",
        "content": "Hello! How can I assist you today?",
        "metadata": {
            "intent": "greeting",
            "confidence": 0.95
        }
    },
    {
        "type": "clarification",
        "content": "Could you please provide more details?",
        "metadata": {
            "intent": "clarification_request",
            "confidence": 0.85
        }
    }
]

# Feedback Fixtures
FEEDBACK_SAMPLES = [
    {
        "type": "user_satisfaction",
        "content": {
            "score": 0.9,
            "comment": "Very helpful response",
            "timestamp": datetime.now().isoformat()
        },
        "metadata": {
            "session_id": str(uuid.uuid4()),
            "user_id": "test_user_1"
        }
    },
    {
        "type": "accuracy",
        "content": {
            "score": 0.85,
            "aspects": ["relevance", "clarity"],
            "timestamp": datetime.now().isoformat()
        },
        "metadata": {
            "session_id": str(uuid.uuid4()),
            "user_id": "test_user_2"
        }
    }
]

# Integration Fixtures
INTEGRATION_CONFIGS = [
    {
        "type": "rest_api",
        "config": {
            "url": "https://api.example.com",
            "auth_type": "bearer",
            "token": "test_token"
        },
        "metadata": {
            "name": "Example API",
            "description": "Test integration"
        }
    },
    {
        "type": "webhook",
        "config": {
            "url": "https://webhook.example.com",
            "method": "POST",
            "headers": {"Content-Type": "application/json"}
        },
        "metadata": {
            "name": "Example Webhook",
            "description": "Test webhook"
        }
    }
]

# Performance Test Data
PERFORMANCE_SCENARIOS = [
    {
        "name": "high_load",
        "num_requests": 1000,
        "concurrent_users": 100,
        "duration_seconds": 60
    },
    {
        "name": "normal_load",
        "num_requests": 100,
        "concurrent_users": 10,
        "duration_seconds": 30
    }
]

# Error Test Data
ERROR_SCENARIOS = [
    {
        "name": "invalid_input",
        "input": None,
        "expected_error": ValueError
    },
    {
        "name": "missing_required_field",
        "input": {"partial": "data"},
        "expected_error": ValueError
    },
    {
        "name": "invalid_model_type",
        "input": {"type": "nonexistent_model"},
        "expected_error": ValueError
    }
]

# Time-based Test Data
TIME_SCENARIOS = [
    {
        "name": "recent_data",
        "start_time": datetime.now() - timedelta(hours=1),
        "end_time": datetime.now()
    },
    {
        "name": "old_data",
        "start_time": datetime.now() - timedelta(days=30),
        "end_time": datetime.now() - timedelta(days=29)
    }
]

def get_test_data(data_type, scenario=None):
    """Helper function to get test data"""
    data_mapping = {
        "text_input": TEXT_INPUTS,
        "image_input": IMAGE_INPUTS,
        "audio_input": AUDIO_INPUTS,
        "memory": MEMORY_SAMPLES,
        "model": MODEL_CONFIGS,
        "training": TRAINING_DATA,
        "inference": INFERENCE_CONTEXTS,
        "message": MESSAGE_TEMPLATES,
        "feedback": FEEDBACK_SAMPLES,
        "integration": INTEGRATION_CONFIGS,
        "performance": PERFORMANCE_SCENARIOS,
        "error": ERROR_SCENARIOS,
        "time": TIME_SCENARIOS
    }
    
    data = data_mapping.get(data_type, [])
    if scenario is not None and isinstance(data, list) and len(data) > 0:
        return data[scenario % len(data)]
    return data

def generate_sequential_data(base_data, num_samples):
    """Generate sequential test data based on a template"""
    results = []
    for i in range(num_samples):
        if isinstance(base_data, dict):
            new_data = base_data.copy()
            new_data["id"] = f"test_{i}"
            if "timestamp" in new_data:
                new_data["timestamp"] = (
                    datetime.now() - timedelta(minutes=i)
                ).isoformat()
            results.append(new_data)
    return results

def generate_random_data(base_data, num_samples):
    """Generate randomized test data based on a template"""
    import random
    
    results = []
    for i in range(num_samples):
        if isinstance(base_data, dict):
            new_data = base_data.copy()
            new_data["id"] = str(uuid.uuid4())
            if "score" in new_data:
                new_data["score"] = random.random()
            if "priority" in new_data:
                new_data["priority"] = random.randint(1, 5)
            results.append(new_data)
    return results 