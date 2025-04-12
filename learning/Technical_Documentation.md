# Learning Agent Module Technical Documentation

## Overview
The Learning Agent module implements a deep reinforcement learning system that integrates with the Memory module for experience replay and knowledge transfer. It uses a neural network architecture for learning behaviors and supports both immediate learning from recent experiences and long-term learning from stored memories.

## Core Components

### LearningAgent
The main class that manages the learning process and integrates with the memory system.

#### Key Features
- Deep Q-Network (DQN) implementation with target network
- Experience replay from both local buffer and memory system
- Similar experience retrieval for knowledge transfer
- TensorBoard integration for training metrics
- Model state saving and loading

### LearningNetwork
Neural network architecture for behavior learning.

#### Architecture
- Input layer: Task-specific state dimensions
- Hidden layers: Configurable size (default 256 units)
- Output layer: Task-specific action dimensions
- Activation: ReLU for hidden layers

### ExperienceBuffer
Local buffer for immediate experience storage and replay.

#### Features
- Circular buffer implementation
- Configurable capacity
- Random sampling for experience replay
- Efficient memory usage

### MemoryClient
Client for interacting with the Memory module.

#### Capabilities
- Asynchronous communication
- Health checking
- Memory storage and retrieval
- Experience replay batch fetching
- Similar experience search

## Integration Points

### Memory System Integration
```python
# Store experience
await agent.store_experience(
    state=current_state,
    action=selected_action,
    reward=reward,
    next_state=next_state,
    done=episode_done,
    memory_client=memory_client
)

# Learn from experiences
await agent.learn_from_experiences(
    memory_client=memory_client,
    gamma=0.99,
    local_batch_size=32
)

# Get similar experiences
similar_experiences = await agent.get_similar_experiences(
    state=current_state,
    memory_client=memory_client,
    k=5
)
```

### Environment Integration
- Receives state observations
- Outputs action selections
- Processes rewards and transitions
- Handles episode boundaries

### Meta Module Integration
- Receives performance metrics
- Adjusts learning parameters
- Optimizes network architecture
- Monitors training progress

## Learning Process

### Experience Collection
1. Observe current state from environment
2. Select action using neural network
3. Execute action and observe results
4. Store experience in local buffer and memory system

### Learning Step
1. Sample experiences from local buffer
2. Retrieve experiences from memory system
3. Compute loss using both experience sets
4. Update neural network weights
5. Periodically update target network

### Knowledge Transfer
1. Convert current state to embedding
2. Retrieve similar experiences from memory
3. Use retrieved experiences to guide learning
4. Adapt behavior based on past successes

## Training Process

### TrainingManager
The `TrainingManager` class orchestrates the training process and handles interaction between the Learning Agent and the environment.

#### Key Features
- Asynchronous training loop
- Epsilon-greedy exploration
- Checkpoint management
- Performance evaluation
- Comprehensive metrics logging

#### Configuration
```python
training_config = {
    "max_episodes": 1000,
    "max_steps_per_episode": 500,
    "epsilon_start": 1.0,
    "epsilon_end": 0.01,
    "epsilon_decay": 0.995,
    "checkpoint_frequency": 100
}
```

### Training Loop
1. Initialize environment and agent
2. For each episode:
   - Reset environment
   - For each step:
     - Select action (epsilon-greedy)
     - Execute action in environment
     - Store experience
     - Learn from experiences
     - Update metrics
   - Update exploration rate
   - Save checkpoint (if needed)

### Evaluation Process
1. Load trained agent
2. Run multiple evaluation episodes
3. Collect performance metrics
4. Generate evaluation report

### Metrics and Monitoring
- Episode rewards and lengths
- Step-wise rewards
- Exploration rate (epsilon)
- Environment-specific metrics
- Best performance tracking
- Training progress visualization

## Task-Specific Reward System

### TaskRewardManager
The `TaskRewardManager` class provides task-specific reward calculations and metrics tracking.

#### Reward Components
- Base Reward: Standard reward for each step
- Time Penalty: Small penalty to encourage efficiency
- Collision Penalty: Penalty for environmental collisions
- Completion Bonus: Large reward for task completion
- Exploration Bonus: Reward for discovering new areas
- Cooperation Bonus: Reward for successful multi-agent coordination
- Efficiency Bonus: Reward for smooth, efficient actions

#### Configuration
```python
reward_config = RewardConfig(
    base_reward=1.0,
    time_penalty=-0.01,
    collision_penalty=-0.5,
    completion_bonus=10.0,
    exploration_bonus=0.5,
    cooperation_bonus=2.0,
    efficiency_bonus=1.0
)
```

### Task-Specific Rewards

#### Exploration Tasks
- Area Discovery: Reward for exploring new areas
- Resource Discovery: Bonus for finding resources
- Coverage Progress: Scaled reward based on total area explored
- Metrics Tracked:
  - Explored area size
  - Exploration coverage percentage
  - Resource discovery rate

#### Multi-Agent Tasks
- Formation Maintenance: Reward for keeping team formation
- Joint Action Success: Bonus for coordinated actions
- Team Cooperation: Tracked per-agent cooperation scores
- Metrics Tracked:
  - Individual agent cooperation scores
  - Team average cooperation
  - Formation maintenance rate

### Reward Calculation Process
1. Calculate base reward and time penalty
2. Apply collision penalties if applicable
3. Add task-specific rewards (exploration/cooperation)
4. Add completion bonus if task is finished
5. Apply efficiency bonus based on action smoothness
6. Track and store reward components

### Integration with Training

#### Initialization
```python
trainer = TrainingManager(
    learning_agent=agent,
    memory_client=memory_client,
    task_type=TaskType.EXPLORATION,
    reward_config=reward_config
)
```

#### Training Loop Integration
```python
# Calculate task-specific reward
reward, components = reward_manager.calculate_reward(
    state=state,
    action=action,
    next_state=next_state,
    info=info
)

# Store experience with calculated reward
await agent.store_experience(
    state=state,
    action=action,
    reward=reward,
    next_state=next_state,
    done=done,
    memory_client=memory_client
)
```

### Metrics and Monitoring

#### Reward Components
- Individual component contributions
- Component value distributions
- Temporal reward patterns
- Task-specific success rates

#### Task Performance
- Overall task completion rate
- Average completion time
- Resource efficiency
- Team coordination metrics

#### Visualization
```python
# Access reward metrics
metrics = reward_manager.get_metrics()

# Plot reward components
for component, value in metrics["reward_components"].items():
    print(f"{component}: {value:.2f}")

# View task-specific metrics
if task_type == TaskType.EXPLORATION:
    print(f"Exploration coverage: {metrics['exploration_coverage']:.2%}")
elif task_type == TaskType.MULTI_AGENT:
    print(f"Team cooperation: {metrics['average_cooperation']:.2f}")
```

### Customization

#### Adding New Reward Components
1. Add component to RewardComponent enum
2. Add configuration parameter to RewardConfig
3. Implement calculation in TaskRewardManager
4. Update metrics collection and visualization

#### Task-Specific Adaptations
1. Identify key performance indicators
2. Design appropriate reward functions
3. Implement metric tracking
4. Add visualization support

## Configuration

### Neural Network
```python
network_config = {
    "input_size": state_dim,
    "hidden_size": 256,
    "output_size": action_dim,
    "learning_rate": 0.001
}
```

### Experience Replay
```python
replay_config = {
    "memory_batch_size": 32,
    "local_buffer_size": 1000,
    "gamma": 0.99
}
```

### Memory Client
```python
memory_config = {
    "base_url": "http://localhost:8401",
    "timeout": 30
}
```

## Performance Considerations

### Memory Usage
- Local buffer size is configurable
- Efficient tensor operations
- Automatic garbage collection
- Memory system offloading

### Computational Efficiency
- Batched learning updates
- Asynchronous memory operations
- GPU acceleration support
- Optimized tensor operations

### Scalability
- Distributed training support
- Parallel environment interaction
- Efficient memory system communication
- Resource usage optimization

## Monitoring

### TensorBoard Metrics
- Total loss
- Local experience loss
- Memory experience loss
- Training steps
- Network gradients
- Learning rate

### System Metrics
- Memory usage
- GPU utilization
- Training throughput
- Network latency

## Error Handling
- Graceful degradation with memory system failures
- Automatic session recovery
- Exception logging and monitoring
- Data validation and sanitization

## Future Enhancements

### Short-term
- Advanced embedding generation
- Prioritized experience replay
- Dynamic network architecture
- Multi-task learning support

### Medium-term
- Meta-learning capabilities
- Curriculum learning
- Hierarchical learning
- Advanced exploration strategies

### Long-term
- Multi-agent learning
- Transfer learning
- Continual learning
- Active learning strategies

## Curriculum Learning

### Overview
The curriculum learning implementation provides a structured approach to progressively increase task difficulty as the agent improves. This helps in efficient learning by starting with simpler tasks and gradually introducing more complex challenges.

### Core Components

#### DifficultyLevel Enum
- `BEGINNER`: Basic tasks with minimal complexity
- `INTERMEDIATE`: Moderate difficulty with some advanced features
- `ADVANCED`: Complex tasks with multiple challenges
- `EXPERT`: Maximum difficulty with all features enabled

#### TaskParameters
Dataclass defining task configuration:
- Workspace dimensions
- Number of obstacles
- Time limits
- Feature requirements (vision, tools, cooperation)
- Success rate thresholds
- Minimum episode requirements

#### CurriculumStage
Manages individual difficulty stages:
- Stage-specific parameters
- Completion criteria
- Progress tracking
- Metrics history

### Curriculum Manager

#### Initialization
```python
curriculum_manager = CurriculumManager()
trainer = TrainingManager(
    learning_agent=agent,
    memory_client=memory_client,
    task_type=TaskType.NAVIGATION,
    use_curriculum=True
)
```

#### Stage Progression
1. **Beginner Stage**
   - Small workspace (5x5x3)
   - Few obstacles (3)
   - Long time limit (300s)
   - Basic navigation only
   - 70% success rate required

2. **Intermediate Stage**
   - Medium workspace (8x8x4)
   - More obstacles (5)
   - Reduced time limit (250s)
   - Vision requirements
   - 60% success rate required

3. **Advanced Stage**
   - Large workspace (10x10x5)
   - Many obstacles (8)
   - Strict time limit (200s)
   - Tool use enabled
   - 50% success rate required

4. **Expert Stage**
   - Maximum workspace (15x15x6)
   - Complex obstacles (12)
   - Minimal time limit (180s)
   - All features enabled
   - 40% success rate required

### Integration with Training

#### Task Creation
```python
# Get current parameters from curriculum
task_params = curriculum_manager.get_current_parameters()

# Create task with appropriate difficulty
task = create_task(task_type, task_params=task_params)
```

#### Progress Tracking
```python
# Update curriculum with episode results
metrics = {
    "task_success": success,
    "average_reward": reward,
    "exploration_coverage": coverage,
    "collision_rate": collisions,
    "tool_use_efficiency": tool_efficiency,
    "cooperation_score": cooperation
}
difficulty_increased = curriculum_manager.update_progress(metrics)
```

### Metrics and Monitoring

#### Stage Metrics
- Episodes completed
- Success rate
- Task-specific metrics
- Completion criteria progress

#### Progress Visualization
- Current difficulty level
- Success rate over time
- Criteria achievement
- Stage transition points

### Checkpoint Management

#### Saving State
```python
# Save curriculum state with checkpoint
checkpoint = {
    "model_state": agent_state,
    "curriculum_state": curriculum_manager.save_state()
}
torch.save(checkpoint, "checkpoint.pt")
```

#### Loading State
```python
# Load curriculum state from checkpoint
checkpoint = torch.load("checkpoint.pt")
curriculum_manager.load_state(checkpoint["curriculum_state"])
```

### Configuration

Curriculum stages can be configured through parameters:

```python
stage_config = {
    "difficulty": DifficultyLevel.INTERMEDIATE,
    "task_params": {
        "workspace_size": (8.0, 8.0, 4.0),
        "num_obstacles": 5,
        "time_limit": 250,
        "vision_required": True
    },
    "completion_criteria": {
        "average_reward": 75.0,
        "exploration_coverage": 0.7,
        "collision_rate": 0.15
    }
}
```

### Performance Considerations

1. **Learning Efficiency**
   - Progressive difficulty reduces learning time
   - Targeted skill development
   - Reduced exploration space

2. **Stability**
   - Smooth difficulty transitions
   - Robust progress tracking
   - Checkpoint management

3. **Resource Usage**
   - Efficient metric tracking
   - Minimal memory overhead
   - Optimized state management

### Future Enhancements

1. **Dynamic Difficulty Adjustment**
   - Real-time parameter tuning
   - Performance-based adaptation
   - Custom difficulty paths

2. **Advanced Metrics**
   - Skill acquisition tracking
   - Learning curve analysis
   - Transfer learning metrics

3. **Multi-Agent Curriculum**
   - Team skill progression
   - Role-specific curricula
   - Coordination complexity

### Error Handling

The curriculum implementation includes robust error handling:
- Invalid parameter detection
- Progress tracking validation
- State consistency checks
- Graceful difficulty transitions

## Usage Examples

### Training
```python
# Initialize components
agent = LearningAgent(
    input_size=state_dim,
    output_size=action_dim
)
memory_client = MemoryClient()
trainer = TrainingManager(
    learning_agent=agent,
    memory_client=memory_client,
    task_type=TaskType.EXPLORATION
)

# Start training
await trainer.train()
```

### Evaluation
```python
# Load checkpoint
trainer.load_checkpoint("checkpoints/best_model.pt")

# Evaluate performance
mean_reward, std_reward = await trainer.evaluate(num_episodes=10)
```

### Monitoring
```python
# Access training metrics
current_reward = trainer.episode_rewards[-1]
best_reward = trainer.best_reward
training_steps = agent.training_steps

# View TensorBoard metrics
# tensorboard --logdir=logs
```

## Training Tips

### Hyperparameter Tuning
- Adjust epsilon decay for exploration balance
- Modify learning rate based on convergence
- Tune network architecture for task complexity
- Adjust batch sizes for stability

### Performance Optimization
- Enable GPU acceleration when available
- Use appropriate batch sizes
- Monitor memory usage
- Implement early stopping

### Common Issues
- Memory system connectivity
- GPU memory management
- Training instability
- Checkpoint corruption

## Deployment

### Model Export
- Save final model state
- Export configuration
- Document hyperparameters
- Version control integration

### Production Setup
- Environment configuration
- Resource allocation
- Monitoring setup
- Backup procedures

## Action Spaces

### Overview
The action spaces implementation provides a flexible and type-safe way to handle different types of actions across various tasks. It includes validation, normalization, and safety constraints for different action types.

### Core Components

#### ActionType Enum
- `CONTINUOUS`: For continuous action spaces (e.g., velocities, positions)
- `DISCRETE`: For discrete action spaces (e.g., grid movement)
- `HYBRID`: For mixed continuous and discrete actions (e.g., movement + communication)

#### ActionBounds
A dataclass that defines the valid ranges for continuous actions:
- `low`: Lower bounds for each action dimension
- `high`: Upper bounds for each action dimension
- `validate()`: Method to check if an action is within bounds

#### Base ActionSpace Class
Common functionality for all action spaces:
- Action type validation
- Normalization to [-1, 1] range
- Denormalization to original space
- Random action sampling
- Action validation

### Task-Specific Action Spaces

#### NavigationActionSpace
- Dimensions: 2 (velocity_x, velocity_y)
- Continuous action space
- Safety constraints near obstacles
- Velocity bounds: [-1.0, 1.0]

#### ManipulationActionSpace
- Dimensions: 7 (joint angles/end-effector pose)
- Continuous action space
- Joint limit validation
- Velocity constraints

#### CooperativeActionSpace
- Hybrid action space
- Per-agent movement and communication signals
- Action decomposition for multi-agent scenarios
- Bounded communication signals

### Integration with Training

#### Action Space Factory
- Creates appropriate action space based on task type
- Handles task-specific parameters
- Ensures consistent action space creation

#### Training Manager Integration
- Action validation during selection
- Invalid action tracking and logging
- Action normalization for network output
- Checkpoint saving/loading of action spaces

### Usage Example

```python
# Create task-specific action space
action_space = ActionSpaceFactory.create_action_space(
    task_type="navigation",
    num_agents=1
)

# Sample random action
action = action_space.sample()

# Validate action
is_valid = action_space.validate_action(action)

# Normalize action for neural network
normalized_action = action_space.normalize_action(action)

# Denormalize network output
original_action = action_space.denormalize_action(normalized_action)
```

### Metrics and Monitoring

The training process tracks several action-related metrics:
- Invalid action count
- Invalid action rate
- Action distribution statistics
- Safety constraint violations

### Safety Considerations

1. **Bounds Checking**
   - All actions are validated against defined bounds
   - Invalid actions are clipped to valid ranges
   - Safety constraints are applied based on environment state

2. **Velocity Constraints**
   - Maximum velocity limits near obstacles
   - Joint velocity limits for manipulation
   - Gradual velocity changes

3. **Multi-Agent Safety**
   - Communication signal bounds
   - Coordinated movement constraints
   - Collision avoidance considerations

### Future Enhancements

1. **Advanced Action Spaces**
   - Hierarchical action spaces
   - Learned action constraints
   - Dynamic action masking

2. **Safety Improvements**
   - Predictive safety constraints
   - Learning-based safety bounds
   - Multi-agent coordination rules

3. **Performance Optimization**
   - Vectorized action processing
   - GPU-accelerated validation
   - Cached bound checking

### Configuration

Action spaces can be configured through the training configuration:

```python
config = {
    "task_type": "navigation",
    "num_agents": 2,
    "action_bounds": {
        "velocity": [-1.0, 1.0],
        "acceleration": [-0.5, 0.5]
    },
    "safety_constraints": {
        "max_velocity_near_obstacle": 0.5,
        "min_agent_distance": 1.0
    }
}
```

### Error Handling

The action space implementation includes robust error handling:
- Invalid initialization detection
- Action validation errors
- Bound violation warnings
- Safety constraint notifications

## Meta-Learning System

### Overview
The meta-learning system provides automated optimization of hyperparameters, neural network architectures, and policy adaptation. It uses Optuna for efficient hyperparameter optimization and implements policy transfer mechanisms for adapting to new tasks.

### Core Components

#### OptimizationType
Supported optimization types:
- `HYPERPARAMETERS`: Optimize learning parameters
- `ARCHITECTURE`: Search for optimal network structures
- `POLICY`: Adapt policies between tasks
- `TRANSFER`: Transfer learning optimization

#### OptimizationConfig
Configuration for optimization processes:
- Number of trials
- Timeout duration
- Optimization metric
- Direction (maximize/minimize)
- Pruning strategy

#### NetworkArchitecture
Manages neural network configurations:
- Input/output dimensions
- Hidden layer structure
- Activation functions
- Dropout rates

### Meta-Learning Manager

#### Initialization
```python
meta_manager = MetaLearningManager(
    base_config={
        "optimization_metric": "mean_reward",
        "storage_url": "sqlite:///meta_learning.db"
    },
    study_name="navigation_optimization"
)

trainer = TrainingManager(
    learning_agent=agent,
    memory_client=memory_client,
    task_type=TaskType.NAVIGATION,
    use_meta_learning=True,
    meta_config=meta_config
)
```

#### Hyperparameter Optimization
Optimizes key learning parameters:
- Learning rate
- Batch size
- Discount factor (gamma)
- Update frequency
- Buffer size
- Network parameters

Example usage:
```python
best_params = await trainer.optimize_hyperparameters()
print(f"Optimal parameters: {best_params}")
```

#### Architecture Search
Searches for optimal network structures:
- Number of layers
- Units per layer
- Dropout rates
- Activation functions

Example usage:
```python
best_architecture = await trainer.optimize_architecture(
    input_dim=state_dim,
    output_dim=action_dim
)
print(f"Optimal architecture: {best_architecture}")
```

#### Policy Adaptation
Adapts policies between related tasks:
- Source policy loading
- Weight transfer
- Fine-tuning
- Performance validation

Example usage:
```python
await trainer.adapt_to_new_task(
    source_task="navigation_simple",
    target_task="navigation_complex"
)
```

### Integration with Training

#### Periodic Optimization
```python
# Training with periodic optimization
await trainer.train(
    n_episodes=10000,
    optimize_interval=100  # Optimize every 100 episodes
)
```

#### Checkpointing
```python
# Save training state including meta-learning
trainer.save_checkpoint("checkpoint.pt")

# Load complete training state
trainer.load_checkpoint("checkpoint.pt")
```

### Metrics and Monitoring

#### Optimization Metrics
- Trial success rate
- Parameter convergence
- Architecture efficiency
- Adaptation performance

#### Progress Tracking
- Best parameters found
- Architecture evolution
- Transfer learning success
- Resource utilization

### Performance Considerations

1. **Computational Efficiency**
   - Parallel trial evaluation
   - Early stopping for poor trials
   - Resource-aware scheduling
   - Cached evaluations

2. **Memory Management**
   - Efficient state storage
   - Trial history pruning
   - Optimized checkpointing
   - Resource monitoring

3. **Scalability**
   - Distributed optimization
   - Multi-GPU support
   - Database storage
   - Load balancing

### Future Enhancements

1. **Advanced Optimization**
   - Multi-objective optimization
   - Population-based training
   - Neural architecture search
   - Meta-gradients

2. **Enhanced Transfer**
   - Progressive policy distillation
   - Multi-task transfer
   - Continual learning
   - Knowledge consolidation

3. **Improved Monitoring**
   - Interactive visualization
   - Performance analytics
   - Resource prediction
   - Anomaly detection

### Error Handling

The meta-learning system includes comprehensive error handling:
- Trial failure recovery
- State validation
- Resource monitoring
- Graceful degradation 