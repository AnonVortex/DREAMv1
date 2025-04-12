"""Training orchestrator module for H-MAS."""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Tuple, Callable, Union
from dataclasses import dataclass
import numpy as np
from pathlib import Path
import json
import logging
from datetime import datetime
import ray
from concurrent.futures import ThreadPoolExecutor
from .environments import EnvironmentType
from .meta_learning import MetaLearningManager, MAML
from .replay import ExperienceReplay
from .optimization import HyperparameterOptimizer

@dataclass
class OrchestratorConfig:
    """Configuration for training orchestrator."""
    max_parallel_tasks: int = 4
    task_queue_size: int = 100
    checkpoint_freq: int = 1000
    eval_freq: int = 100
    save_dir: str = "orchestrator_results"
    min_task_episodes: int = 10
    max_task_episodes: int = 100
    performance_threshold: float = 0.8
    curriculum_length: int = 5
    knowledge_transfer_threshold: float = 0.7
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class TrainingTask:
    """Represents a single training task."""
    
    def __init__(
        self,
        task_id: str,
        env_type: EnvironmentType,
        config: Dict[str, Any],
        difficulty: float = 0.0
    ):
        """Initialize training task."""
        self.task_id = task_id
        self.env_type = env_type
        self.config = config
        self.difficulty = difficulty
        self.episodes_completed = 0
        self.performance_history: List[float] = []
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.status = "pending"
        self.model: Optional[nn.Module] = None
        
    def update_performance(self, performance: float) -> None:
        """Update task performance history."""
        self.performance_history.append(performance)
        
    def get_average_performance(self, window: int = 5) -> float:
        """Get average performance over last n episodes."""
        if not self.performance_history:
            return 0.0
        return np.mean(self.performance_history[-window:])
        
    def is_completed(self, threshold: float) -> bool:
        """Check if task is completed based on performance."""
        return self.get_average_performance() >= threshold

class TrainingOrchestrator:
    """Manages multi-task training across environments."""
    
    def __init__(
        self,
        config: OrchestratorConfig,
        meta_learning_manager: MetaLearningManager,
        experience_replay: ExperienceReplay,
        hyperparameter_optimizer: HyperparameterOptimizer
    ):
        """Initialize training orchestrator."""
        self.config = config
        self.meta_learning_manager = meta_learning_manager
        self.experience_replay = experience_replay
        self.hyperparameter_optimizer = hyperparameter_optimizer
        self.logger = logging.getLogger("training_orchestrator")
        
        self.active_tasks: Dict[str, TrainingTask] = {}
        self.completed_tasks: Dict[str, TrainingTask] = {}
        self.task_queue: List[TrainingTask] = []
        self.global_step = 0
        
        # Create save directory
        Path(config.save_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize Ray for parallel training if not already initialized
        if not ray.is_initialized():
            ray.init(num_cpus=config.max_parallel_tasks)
            
    def add_task(self, task: TrainingTask) -> None:
        """Add new task to training queue."""
        if len(self.task_queue) >= self.config.task_queue_size:
            self.logger.warning("Task queue is full, dropping oldest task")
            self.task_queue.pop(0)
            
        self.task_queue.append(task)
        self._update_task_priorities()
        
    def train(self) -> None:
        """Main training loop."""
        while self.task_queue or self.active_tasks:
            # Start new tasks if capacity available
            while (
                len(self.active_tasks) < self.config.max_parallel_tasks
                and self.task_queue
            ):
                task = self._get_next_task()
                self._start_task(task)
                
            # Update active tasks
            completed_tasks = []
            for task_id, task in self.active_tasks.items():
                if self._update_task(task):
                    completed_tasks.append(task_id)
                    
            # Move completed tasks
            for task_id in completed_tasks:
                task = self.active_tasks.pop(task_id)
                self.completed_tasks[task_id] = task
                
            # Periodic operations
            self._periodic_operations()
            
            self.global_step += 1
            
    def _get_next_task(self) -> Optional[TrainingTask]:
        """Get next task from queue based on priority."""
        if not self.task_queue:
            return None
            
        # Get task with highest priority
        task = self.task_queue.pop(0)
        
        # Generate curriculum if needed
        if task.difficulty > 0.5:  # Only for more difficult tasks
            curriculum = self.meta_learning_manager.generate_curriculum(
                task.task_id,
                self.config.curriculum_length
            )
            
            # Add curriculum tasks back to queue
            for curr_task_id in curriculum:
                if (
                    curr_task_id not in self.active_tasks
                    and curr_task_id not in self.completed_tasks
                ):
                    curr_task = self._create_curriculum_task(curr_task_id, task)
                    self.task_queue.append(curr_task)
                    
        return task
        
    def _start_task(self, task: TrainingTask) -> None:
        """Start training a task."""
        task.start_time = datetime.now()
        task.status = "running"
        
        # Initialize model for task
        if task.model is None:
            # Try to transfer knowledge from similar tasks
            similar_tasks = self.meta_learning_manager.task_buffer.get_similar_tasks(
                task.task_id
            )
            
            if similar_tasks:
                best_task_id = similar_tasks[0][0]
                if best_task_id in self.completed_tasks:
                    source_task = self.completed_tasks[best_task_id]
                    if source_task.get_average_performance() >= self.config.knowledge_transfer_threshold:
                        # Adapt model from similar task
                        task.model = self._adapt_model(source_task.model, task)
                        self.logger.info(f"Transferred knowledge from task {best_task_id}")
                        
            if task.model is None:
                # Create new model if no transfer possible
                task.model = self._create_model(task)
                
        self.active_tasks[task.task_id] = task
        
    def _update_task(self, task: TrainingTask) -> bool:
        """Update task training and check completion."""
        # Perform training step
        performance = self._train_step(task)
        task.update_performance(performance)
        
        # Check completion criteria
        if task.episodes_completed >= self.config.min_task_episodes:
            if task.is_completed(self.config.performance_threshold):
                task.status = "completed"
                task.end_time = datetime.now()
                return True
            elif task.episodes_completed >= self.config.max_task_episodes:
                task.status = "failed"
                task.end_time = datetime.now()
                return True
                
        return False
        
    def _periodic_operations(self) -> None:
        """Perform periodic maintenance operations."""
        # Save checkpoint
        if self.global_step % self.config.checkpoint_freq == 0:
            self._save_checkpoint()
            
        # Evaluate active tasks
        if self.global_step % self.config.eval_freq == 0:
            self._evaluate_tasks()
            
        # Update task priorities
        self._update_task_priorities()
        
        # Optimize hyperparameters if needed
        if self.global_step % 5000 == 0:  # Adjust frequency as needed
            self._optimize_hyperparameters()
            
    def _update_task_priorities(self) -> None:
        """Update priorities of tasks in queue."""
        if not self.task_queue:
            return
            
        # Sort tasks by priority score
        self.task_queue.sort(key=lambda t: self._compute_task_priority(t), reverse=True)
        
    def _compute_task_priority(self, task: TrainingTask) -> float:
        """Compute priority score for a task."""
        # Base priority based on difficulty
        priority = task.difficulty
        
        # Adjust based on dependencies
        curriculum = self.meta_learning_manager.generate_curriculum(task.task_id, 1)
        if curriculum:
            prereq_task_id = curriculum[0]
            if prereq_task_id in self.completed_tasks:
                priority *= 1.5  # Boost priority if prerequisites are completed
                
        # Adjust based on similar completed tasks
        similar_tasks = self.meta_learning_manager.task_buffer.get_similar_tasks(
            task.task_id
        )
        completed_similar = sum(
            1 for task_id, _ in similar_tasks
            if task_id in self.completed_tasks
        )
        priority *= (1.0 + 0.1 * completed_similar)
        
        return priority
        
    def _train_step(self, task: TrainingTask) -> float:
        """Perform single training step for a task."""
        if task.model is None:
            raise ValueError(f"Model not initialized for task {task.task_id}")
            
        # Set model to training mode
        task.model.train()
        
        # Sample experience batch from replay buffer
        experiences = self.experience_replay.sample_batch(
            env_type=task.env_type,
            batch_size=32,  # Could be configurable
            prioritize=True
        )
        
        if not experiences:
            # If no experiences available, collect some through interaction
            experiences = self._collect_experiences(task)
            
        # Convert experiences to tensors
        states = torch.tensor([exp.state for exp in experiences], device=self.config.device)
        actions = torch.tensor([exp.action for exp in experiences], device=self.config.device)
        rewards = torch.tensor([exp.reward for exp in experiences], device=self.config.device)
        next_states = torch.tensor([exp.next_state for exp in experiences], device=self.config.device)
        dones = torch.tensor([exp.done for exp in experiences], device=self.config.device)
        
        # Compute value estimates
        with torch.no_grad():
            next_values = task.model(next_states)
            target_values = rewards + (1.0 - dones) * 0.99 * next_values.max(1)[0]
            
        # Compute current values
        values = task.model(states)
        action_values = values.gather(1, actions.unsqueeze(1)).squeeze()
        
        # Compute loss
        loss = nn.MSELoss()(action_values, target_values)
        
        # Update model
        task.optimizer.zero_grad()
        loss.backward()
        task.optimizer.step()
        
        # Update experience priorities based on TD error
        td_errors = (target_values - action_values).abs().detach().cpu().numpy()
        self.experience_replay.update_priorities(experiences, td_errors)
        
        # Evaluate performance
        with torch.no_grad():
            performance = self._evaluate_episode(task)
            
        task.episodes_completed += 1
        return performance
        
    def _collect_experiences(self, task: TrainingTask) -> List[Experience]:
        """Collect experiences through environment interaction."""
        experiences = []
        state = task.env.reset()
        
        for _ in range(100):  # Maximum steps per episode
            # Get action from model
            with torch.no_grad():
                state_tensor = torch.tensor(state, device=self.config.device).unsqueeze(0)
                action = task.model(state_tensor).argmax().item()
                
            # Take action in environment
            next_state, reward, done, _ = task.env.step(action)
            
            # Store experience
            experience = Experience(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done
            )
            experiences.append(experience)
            self.experience_replay.add(experience)
            
            if done:
                break
                
            state = next_state
            
        return experiences
        
    def _evaluate_episode(self, task: TrainingTask) -> float:
        """Evaluate model performance on a single episode."""
        task.model.eval()
        total_reward = 0.0
        state = task.env.reset()
        done = False
        
        while not done:
            with torch.no_grad():
                state_tensor = torch.tensor(state, device=self.config.device).unsqueeze(0)
                action = task.model(state_tensor).argmax().item()
                
            state, reward, done, _ = task.env.step(action)
            total_reward += reward
            
        return total_reward
        
    def _evaluate_tasks(self) -> None:
        """Evaluate all active tasks."""
        for task in self.active_tasks.values():
            # TODO: Implement actual evaluation
            self.logger.info(
                f"Task {task.task_id}: "
                f"Episodes = {task.episodes_completed}, "
                f"Performance = {task.get_average_performance():.3f}"
            )
            
    def _optimize_hyperparameters(self) -> None:
        """Trigger hyperparameter optimization."""
        if not self.completed_tasks:
            return
            
        # Select best performing completed tasks for optimization
        best_tasks = sorted(
            self.completed_tasks.values(),
            key=lambda t: t.get_average_performance(),
            reverse=True
        )[:5]
        
        # Optimize hyperparameters using these tasks
        for task in best_tasks:
            self.hyperparameter_optimizer.optimize_environment(
                task.env_type,
                lambda params: self._evaluate_hyperparameters(task, params)
            )
            
    def _evaluate_hyperparameters(
        self,
        task: TrainingTask,
        params: Dict[str, Any]
    ) -> float:
        """Evaluate hyperparameter configuration."""
        # TODO: Implement actual hyperparameter evaluation
        return np.random.uniform(0.0, 1.0)  # Placeholder
        
    def _save_checkpoint(self) -> None:
        """Save orchestrator state."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = Path(self.config.save_dir) / f"checkpoint_{timestamp}.json"
        
        state = {
            "global_step": self.global_step,
            "active_tasks": {
                task_id: self._serialize_task(task)
                for task_id, task in self.active_tasks.items()
            },
            "completed_tasks": {
                task_id: self._serialize_task(task)
                for task_id, task in self.completed_tasks.items()
            },
            "task_queue": [
                self._serialize_task(task) for task in self.task_queue
            ]
        }
        
        with open(path, "w") as f:
            json.dump(state, f, indent=2)
            
    def _serialize_task(self, task: TrainingTask) -> Dict[str, Any]:
        """Convert task to serializable format."""
        return {
            "task_id": task.task_id,
            "env_type": task.env_type.value,
            "config": task.config,
            "difficulty": task.difficulty,
            "episodes_completed": task.episodes_completed,
            "performance_history": task.performance_history,
            "start_time": task.start_time.isoformat() if task.start_time else None,
            "end_time": task.end_time.isoformat() if task.end_time else None,
            "status": task.status
        }
        
    def _create_curriculum_task(
        self,
        task_id: str,
        target_task: TrainingTask
    ) -> TrainingTask:
        """Create a curriculum task based on target task."""
        return TrainingTask(
            task_id=task_id,
            env_type=target_task.env_type,
            config=self._adjust_task_config(target_task.config, 0.7),  # Reduce difficulty
            difficulty=target_task.difficulty * 0.7
        )
        
    def _adjust_task_config(
        self,
        config: Dict[str, Any],
        scale_factor: float
    ) -> Dict[str, Any]:
        """Adjust task configuration for curriculum."""
        new_config = config.copy()
        
        # Scale numerical parameters
        for key, value in new_config.items():
            if isinstance(value, (int, float)):
                new_config[key] = value * scale_factor
                
        return new_config
        
    def _create_model(self, task: TrainingTask) -> nn.Module:
        """Create new model for task."""
        # Get environment observation and action spaces
        obs_dim = task.env.observation_space.shape[0]
        action_dim = task.env.action_space.n
        
        # Create model architecture based on environment type
        if task.env_type == EnvironmentType.PERCEPTION:
            model = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, stride=2),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=2),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(64 * 6 * 6, 512),
                nn.ReLU(),
                nn.Linear(512, action_dim)
            )
        elif task.env_type == EnvironmentType.REASONING:
            model = nn.Sequential(
                nn.Linear(obs_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, action_dim)
            )
        else:  # Default architecture for other environment types
            model = nn.Sequential(
                nn.Linear(obs_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, action_dim)
            )
            
        model = model.to(self.config.device)
        
        # Initialize optimizer
        task.optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        return model
        
    def _adapt_model(
        self,
        source_model: nn.Module,
        target_task: TrainingTask
    ) -> nn.Module:
        """Adapt model from source task to target task using meta-learning."""
        # Create MAML adapter for the source model
        maml = MAML(
            model=source_model,
            inner_lr=0.01,
            first_order=True,
            allow_nograd=True
        )
        
        # Collect adaptation data
        adaptation_data = self._collect_adaptation_data(target_task)
        
        if not adaptation_data:
            return self._create_model(target_task)
            
        # Prepare adaptation data
        support_x = torch.tensor([d[0] for d in adaptation_data], device=self.config.device)
        support_y = torch.tensor([d[1] for d in adaptation_data], device=self.config.device)
        
        # Adapt model using MAML
        adapted_model = maml.adapt(
            support_x,
            support_y,
            num_steps=5,  # Number of adaptation steps
            allow_nograd=True
        )
        
        # Initialize optimizer for the adapted model
        target_task.optimizer = torch.optim.Adam(adapted_model.parameters(), lr=0.001)
        
        return adapted_model
        
    def _collect_adaptation_data(self, task: TrainingTask) -> List[Tuple[np.ndarray, int]]:
        """Collect data for model adaptation."""
        adaptation_data = []
        state = task.env.reset()
        
        for _ in range(50):  # Collect limited adaptation data
            # Random action for exploration
            action = task.env.action_space.sample()
            next_state, reward, done, _ = task.env.step(action)
            
            # Store state-action pair
            adaptation_data.append((state, action))
            
            if done:
                break
                
            state = next_state
            
        return adaptation_data
        
    def get_training_summary(self) -> Dict[str, Any]:
        """Get summary of training progress."""
        return {
            "global_step": self.global_step,
            "active_tasks": len(self.active_tasks),
            "completed_tasks": len(self.completed_tasks),
            "queued_tasks": len(self.task_queue),
            "performance_stats": self._get_performance_stats(),
            "training_time": self._get_training_time()
        }
        
    def _get_performance_stats(self) -> Dict[str, Any]:
        """Calculate performance statistics."""
        all_tasks = {
            **self.active_tasks,
            **self.completed_tasks
        }
        
        if not all_tasks:
            return {
                "average": 0.0,
                "best": 0.0,
                "worst": 0.0,
                "completion_rate": 0.0
            }
            
        performances = [
            task.get_average_performance()
            for task in all_tasks.values()
        ]
        
        completed = sum(
            1 for task in all_tasks.values()
            if task.status == "completed"
        )
        
        return {
            "average": float(np.mean(performances)),
            "best": float(np.max(performances)),
            "worst": float(np.min(performances)),
            "completion_rate": completed / len(all_tasks)
        }
        
    def _get_training_time(self) -> Dict[str, float]:
        """Calculate training time statistics."""
        completed_times = [
            (task.end_time - task.start_time).total_seconds()
            for task in self.completed_tasks.values()
            if task.start_time and task.end_time
        ]
        
        if not completed_times:
            return {
                "average_task_time": 0.0,
                "total_training_time": 0.0
            }
            
        return {
            "average_task_time": float(np.mean(completed_times)),
            "total_training_time": float(np.sum(completed_times))
        } 