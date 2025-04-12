# HMAS Learning Module

## Overview
The Learning Module provides advanced learning capabilities for the Hierarchical Multi-Agent System, including curriculum learning, transfer learning, and meta-learning mechanisms.

## Features
- Curriculum Learning Framework
- Transfer Learning Capabilities
- Meta-Learning Implementation
- Self-Improvement Mechanisms
- Performance Monitoring
- Resource Management

## Configuration
The module can be configured through environment variables and the `config.py` file:

```bash
LEARNING_RATE=0.001
BATCH_SIZE=32
EPOCHS=10
```

## API Endpoints
- `POST /curriculum/add_task`: Add a new task to the curriculum
- `POST /transfer/store_knowledge`: Store knowledge for transfer learning
- `POST /transfer/transfer_knowledge`: Transfer knowledge between domains
- `POST /meta/update_parameters`: Update meta-learning parameters
- `POST /meta/adapt_task`: Adapt to a new task

## Dependencies
- PyTorch
- NumPy
- FastAPI
- Scikit-learn
- Pandas

## Usage
```python
from learning import LearningConfig, CurriculumManager

# Configure learning
config = LearningConfig(
    learning_rate=0.001,
    curriculum_enabled=True,
    transfer_learning_enabled=True
)

# Add curriculum task
curriculum_manager = CurriculumManager()
curriculum_manager.add_task(level=1, task={"type": "classification", "difficulty": "easy"})
```

## Learning Strategies
1. Curriculum Learning: Progressive task difficulty
2. Transfer Learning: Knowledge sharing between domains
3. Meta-Learning: Learning to learn
4. Self-Improvement: Continuous optimization

## Contributing
See the main project's contribution guidelines.

## License
Same as the main project. 