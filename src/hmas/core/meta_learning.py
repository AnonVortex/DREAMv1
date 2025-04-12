"""Meta-learning module for H-MAS."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Any, Tuple, Callable, Union
from dataclasses import dataclass
import numpy as np
from pathlib import Path
import json
import logging
from datetime import datetime
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from .environments import EnvironmentType

@dataclass
class MetaLearningConfig:
    """Configuration for meta-learning."""
    inner_lr: float = 0.01  # Inner loop learning rate
    outer_lr: float = 0.001  # Outer loop learning rate
    num_inner_steps: int = 5  # Number of gradient steps for adaptation
    meta_batch_size: int = 16  # Number of tasks per meta-batch
    num_epochs: int = 1000
    task_embedding_dim: int = 64
    save_dir: str = "meta_learning_results"
    checkpoint_freq: int = 10
    eval_freq: int = 5
    early_stopping_patience: int = 20
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class TaskEmbedding(nn.Module):
    """Neural network for learning task embeddings."""
    
    def __init__(
        self,
        input_dim: int,
        embedding_dim: int,
        hidden_dims: List[int] = [256, 128]
    ):
        """Initialize task embedding network."""
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
            
        layers.append(nn.Linear(prev_dim, embedding_dim))
        self.network = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.network(x)

class MAML(nn.Module):
    """Model-Agnostic Meta-Learning implementation."""
    
    def __init__(
        self,
        model: nn.Module,
        config: MetaLearningConfig
    ):
        """Initialize MAML."""
        super().__init__()
        self.model = model
        self.config = config
        self.task_embedding = None  # Will be set when task embedding is learned
        
    def adapt(
        self,
        support_data: Tuple[torch.Tensor, torch.Tensor],
        num_steps: Optional[int] = None
    ) -> nn.Module:
        """Adapt the model to a new task using support data."""
        num_steps = num_steps or self.config.num_inner_steps
        x_support, y_support = support_data
        
        # Create a clone of the model for adaptation
        adapted_model = copy.deepcopy(self.model)
        adapted_model.train()
        
        # Create optimizer for adaptation
        optimizer = torch.optim.Adam(
            adapted_model.parameters(),
            lr=self.config.inner_lr
        )
        
        # Perform adaptation steps
        for _ in range(num_steps):
            optimizer.zero_grad()
            predictions = adapted_model(x_support)
            loss = F.mse_loss(predictions, y_support)
            loss.backward()
            optimizer.step()
            
        return adapted_model
        
    def meta_learn(
        self,
        task_batch: List[Tuple[torch.Tensor, torch.Tensor]],
        outer_optimizer: torch.optim.Optimizer
    ) -> float:
        """Perform meta-learning step on a batch of tasks."""
        outer_loss = 0.0
        
        for support, query in task_batch:
            # Adapt model to task
            adapted_model = self.adapt((support[0], support[1]))
            
            # Compute loss on query set
            predictions = adapted_model(query[0])
            task_loss = F.mse_loss(predictions, query[1])
            outer_loss += task_loss
            
        # Update meta-parameters
        outer_loss = outer_loss / len(task_batch)
        outer_optimizer.zero_grad()
        outer_loss.backward()
        outer_optimizer.step()
        
        return outer_loss.item()

class TaskBuffer:
    """Buffer for storing task data and computing embeddings."""
    
    def __init__(
        self,
        embedding_dim: int,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """Initialize task buffer."""
        self.embedding_dim = embedding_dim
        self.device = device
        self.tasks: Dict[str, Dict[str, Any]] = {}
        self.embeddings: Dict[str, torch.Tensor] = {}
        self.embedding_network = TaskEmbedding(
            input_dim=self._get_task_feature_dim(),
            embedding_dim=embedding_dim
        ).to(device)
        
    def add_task(
        self,
        task_id: str,
        env_type: EnvironmentType,
        task_data: Dict[str, Any]
    ) -> None:
        """Add new task to buffer."""
        self.tasks[task_id] = {
            "env_type": env_type,
            "data": task_data,
            "performance_history": [],
            "timestamp": datetime.now()
        }
        
        # Compute and store task embedding
        features = self._extract_task_features(env_type, task_data)
        embedding = self.embedding_network(features.unsqueeze(0))
        self.embeddings[task_id] = embedding.squeeze(0)
        
    def update_task_performance(
        self,
        task_id: str,
        performance: float
    ) -> None:
        """Update performance history for a task."""
        if task_id in self.tasks:
            self.tasks[task_id]["performance_history"].append(performance)
            
    def get_similar_tasks(
        self,
        task_id: str,
        k: int = 5
    ) -> List[Tuple[str, float]]:
        """Find k most similar tasks based on embeddings."""
        if task_id not in self.embeddings:
            return []
            
        query_embedding = self.embeddings[task_id]
        similarities = []
        
        for other_id, other_embedding in self.embeddings.items():
            if other_id != task_id:
                similarity = F.cosine_similarity(
                    query_embedding.unsqueeze(0),
                    other_embedding.unsqueeze(0)
                ).item()
                similarities.append((other_id, similarity))
                
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]
        
    def get_task_curriculum(
        self,
        target_task_id: str,
        num_tasks: int = 5
    ) -> List[str]:
        """Generate curriculum of tasks leading to target task."""
        if target_task_id not in self.embeddings:
            return []
            
        target_embedding = self.embeddings[target_task_id]
        curriculum = []
        available_tasks = set(self.tasks.keys()) - {target_task_id}
        
        while len(curriculum) < num_tasks and available_tasks:
            # Find task with embedding closest to target but further than previous tasks
            best_task = None
            best_distance = float('inf')
            
            for task_id in available_tasks:
                embedding = self.embeddings[task_id]
                distance = torch.norm(embedding - target_embedding).item()
                
                if distance < best_distance:
                    # Check if task is more difficult than previous tasks
                    if not curriculum or self._is_task_progression_valid(
                        curriculum[-1], task_id
                    ):
                        best_task = task_id
                        best_distance = distance
                        
            if best_task is None:
                break
                
            curriculum.append(best_task)
            available_tasks.remove(best_task)
            
        return curriculum
        
    def _extract_task_features(
        self,
        env_type: EnvironmentType,
        task_data: Dict[str, Any]
    ) -> torch.Tensor:
        """Extract numerical features from task data."""
        features = []
        
        # Add environment type as one-hot encoding
        env_encoding = torch.zeros(len(EnvironmentType))
        env_encoding[env_type.value] = 1
        features.extend(env_encoding.tolist())
        
        # Add task-specific features
        if env_type == EnvironmentType.PERCEPTION:
            features.extend([
                task_data.get("input_dimension", 0),
                task_data.get("num_classes", 0),
                task_data.get("noise_level", 0.0)
            ])
        elif env_type == EnvironmentType.COMMUNICATION:
            features.extend([
                task_data.get("vocab_size", 0),
                task_data.get("max_sequence_length", 0),
                task_data.get("num_agents", 0)
            ])
        elif env_type == EnvironmentType.PLANNING:
            features.extend([
                task_data.get("horizon_length", 0),
                task_data.get("branching_factor", 0),
                task_data.get("num_objectives", 0)
            ])
        elif env_type == EnvironmentType.REASONING:
            features.extend([
                task_data.get("num_premises", 0),
                task_data.get("rule_complexity", 0.0),
                task_data.get("inference_depth", 0)
            ])
            
        return torch.tensor(features, dtype=torch.float32, device=self.device)
        
    def _get_task_feature_dim(self) -> int:
        """Get dimension of task features."""
        return len(EnvironmentType) + 3  # env_type one-hot + 3 task-specific features
        
    def _is_task_progression_valid(
        self,
        prev_task_id: str,
        next_task_id: str
    ) -> bool:
        """Check if task progression is valid based on performance history."""
        prev_task = self.tasks[prev_task_id]
        next_task = self.tasks[next_task_id]
        
        if not prev_task["performance_history"]:
            return True
            
        # Check if agent has achieved good performance on previous task
        prev_performance = np.mean(prev_task["performance_history"][-5:])
        return prev_performance >= 0.7  # Threshold for task progression

class MetaLearningManager:
    """Manages meta-learning process across tasks."""
    
    def __init__(self, config: MetaLearningConfig):
        """Initialize meta-learning manager."""
        self.config = config
        self.logger = logging.getLogger("meta_learning_manager")
        self.task_buffer = TaskBuffer(
            embedding_dim=config.task_embedding_dim,
            device=config.device
        )
        
        # Create save directory
        Path(config.save_dir).mkdir(parents=True, exist_ok=True)
        
    def train_meta_learner(
        self,
        model: nn.Module,
        task_batch_generator: Callable[[], List[Tuple[torch.Tensor, torch.Tensor]]],
        num_epochs: Optional[int] = None
    ) -> MAML:
        """Train meta-learner on a set of tasks."""
        num_epochs = num_epochs or self.config.num_epochs
        device = self.config.device
        
        # Initialize MAML
        maml = MAML(model, self.config).to(device)
        outer_optimizer = torch.optim.Adam(
            maml.parameters(),
            lr=self.config.outer_lr
        )
        
        best_loss = float('inf')
        patience_counter = 0
        training_history = []
        
        for epoch in range(num_epochs):
            epoch_losses = []
            
            # Train on multiple meta-batches
            for _ in range(self.config.meta_batch_size):
                task_batch = task_batch_generator()
                loss = maml.meta_learn(task_batch, outer_optimizer)
                epoch_losses.append(loss)
                
            avg_loss = np.mean(epoch_losses)
            training_history.append(avg_loss)
            
            # Log progress
            self.logger.info(f"Epoch {epoch}: Loss = {avg_loss:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % self.config.checkpoint_freq == 0:
                self._save_checkpoint(maml, training_history, epoch)
                
            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= self.config.early_stopping_patience:
                self.logger.info("Early stopping triggered")
                break
                
        return maml
        
    def adapt_to_new_task(
        self,
        maml: MAML,
        support_data: Tuple[torch.Tensor, torch.Tensor],
        task_id: str,
        num_steps: Optional[int] = None
    ) -> nn.Module:
        """Adapt meta-learned model to new task."""
        # Find similar tasks for potential knowledge transfer
        similar_tasks = self.task_buffer.get_similar_tasks(task_id)
        
        if similar_tasks:
            # Use information from similar tasks to initialize adaptation
            self.logger.info(f"Found {len(similar_tasks)} similar tasks for {task_id}")
            # TODO: Implement knowledge transfer from similar tasks
            
        # Adapt model to new task
        adapted_model = maml.adapt(support_data, num_steps)
        return adapted_model
        
    def generate_curriculum(
        self,
        target_task_id: str,
        num_tasks: int = 5
    ) -> List[str]:
        """Generate curriculum for target task."""
        return self.task_buffer.get_task_curriculum(target_task_id, num_tasks)
        
    def _save_checkpoint(
        self,
        maml: MAML,
        training_history: List[float],
        epoch: int
    ) -> None:
        """Save training checkpoint."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_path = Path(self.config.save_dir) / f"checkpoint_{timestamp}_epoch_{epoch}.pt"
        
        torch.save({
            "model_state_dict": maml.state_dict(),
            "training_history": training_history,
            "config": self.config.__dict__,
            "epoch": epoch
        }, checkpoint_path)
        
    def load_checkpoint(self, path: str) -> Tuple[MAML, List[float]]:
        """Load training checkpoint."""
        checkpoint = torch.load(path)
        
        # Recreate MAML instance
        config = MetaLearningConfig(**checkpoint["config"])
        model = self._create_base_model()  # You need to implement this
        maml = MAML(model, config)
        maml.load_state_dict(checkpoint["model_state_dict"])
        
        return maml, checkpoint["training_history"] 