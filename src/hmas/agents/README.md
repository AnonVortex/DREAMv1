# H-MAS Agents Module

This module contains various agent implementations for the H-MAS (Hierarchical Multi-Agent System) framework.

## LearningAgent

The `LearningAgent` is a sophisticated agent implementation capable of multiple types of learning and adaptation. It extends the base `Agent` class and provides comprehensive learning capabilities.

### Features

- **Multiple Learning Types**:
  - Reinforcement Learning
  - Supervised Learning
  - Unsupervised Learning
  - Meta-Learning

- **Adaptive Learning**:
  - Dynamic learning rate adjustment
  - Exploration rate adaptation
  - Performance-based strategy selection

- **State Management**:
  - Model state tracking
  - Performance history
  - Learning progress monitoring
  - Meta-learning strategy management

### Usage

#### Basic Initialization

```python
from hmas.agents.learning_agent import LearningAgent

# Create a learning agent with default settings
agent = LearningAgent(name="my_learning_agent")

# Create an agent with specific learning types
agent = LearningAgent(
    name="custom_learner",
    learning_types=["reinforcement", "supervised"],
    memory_size=2000,
    learning_rate=0.005,
    exploration_rate=0.2
)

# Initialize the agent
await agent.initialize()
```

#### Reinforcement Learning

```python
# Process reinforcement learning input
result = await agent.process({
    "type": "reinforcement_learning",
    "data": {
        "state": np.array([1.0, 0.0, 0.0]),
        "action": 1,
        "reward": 1.0,
        "next_state": np.array([0.0, 1.0, 0.0])
    },
    "context": {}
})

# Access results
next_action = result["next_action"]
model_update = result["model_update"]
current_exploration = result["exploration_rate"]
```

#### Supervised Learning

```python
# Process supervised learning input
result = await agent.process({
    "type": "supervised_learning",
    "data": {
        "inputs": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        "targets": np.array([0, 1])
    },
    "context": {}
})

# Access results
loss = result["loss"]
performance = result["performance"]
model_update = result["model_update"]
```

#### Unsupervised Learning

```python
# Process unsupervised learning input
result = await agent.process({
    "type": "unsupervised_learning",
    "data": {
        "inputs": np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
    },
    "context": {}
})

# Access results
patterns = result["patterns"]
quality = result["quality_metrics"]
clusters = result["model_update"]["num_clusters"]
```

#### Meta-Learning

```python
# Process meta-learning input
result = await agent.process({
    "type": "meta_learning",
    "data": {
        "learning_type": "supervised",
        "performance_data": [
            {"performance": 0.8, "timestamp": datetime.now()},
            {"performance": 0.85, "timestamp": datetime.now()},
            {"performance": 0.9, "timestamp": datetime.now()}
        ]
    },
    "context": {}
})

# Access results
analysis = result["analysis"]
new_strategy = result["new_strategy"]
adaptation = result["adaptation"]
```

#### Learning from Experience

```python
# Learn from experience
success = await agent.learn({
    "feedback": {
        "type": "supervised",
        "performance": 0.85
    }
})
```

#### Self-Reflection

```python
# Get agent's self-reflection
reflection = await agent.reflect()

# Access reflection data
status = reflection["status"]
performance = reflection["performance_summary"]
strategies = reflection["meta_learning"]["strategies"]
```

### Implementation Details

#### State Structure

The agent maintains a comprehensive state dictionary:

```python
state = {
    "models": {
        "reinforcement": {"q_table": {}, "policy": "epsilon_greedy"},
        "supervised": {"weights": array, "bias": float},
        "unsupervised": {"clusters": [], "embeddings": []},
        "meta": {"strategies": {}, "adaptations": []}
    },
    "performance_history": {
        learning_type: [{"performance": float, "timestamp": datetime}, ...]
    },
    "learning_progress": {
        "episodes": int,
        "improvements": {learning_type: float},
        "adaptation_rate": {learning_type: float}
    },
    "meta_learning": {
        "strategies": {learning_type: str},
        "effectiveness": {learning_type: float}
    }
}
```

#### Learning Strategies

The agent supports three learning strategies:
- `default`: Balanced exploration and exploitation
- `adaptive`: Reduced exploration for stable improvement
- `exploratory`: Increased exploration for declining performance

#### Performance Analysis

The agent continuously analyzes learning performance using metrics:
- Improvement rate
- Learning stability
- Adaptation effectiveness

### Testing

Comprehensive tests are available in `tests/hmas/agents/test_learning_agent.py`. Run tests using pytest:

```bash
pytest tests/hmas/agents/test_learning_agent.py -v
``` 