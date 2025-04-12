"""Learning module for H-MAS AGI system."""

from typing import Dict, List, Optional, Any, Tuple, Union, Set
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from enum import Enum
from pathlib import Path
import logging
from datetime import datetime
import json
from uuid import UUID, uuid4

from .memory import MemorySystem
from .reasoning import ReasoningEngine
from .consciousness import ConsciousnessCore

class LearningMode(Enum):
    """Types of learning supported by the system."""
    SUPERVISED = "supervised"
    UNSUPERVISED = "unsupervised"
    REINFORCEMENT = "reinforcement"
    META = "meta"
    ACTIVE = "active"
    CURRICULUM = "curriculum"
    TRANSFER = "transfer"
    CONTINUAL = "continual"

@dataclass
class LearningConfig:
    """Configuration for learning system."""
    base_learning_rate: float = 0.001
    meta_learning_rate: float = 0.0001
    batch_size: int = 32
    max_epochs: int = 100
    patience: int = 10
    validation_split: float = 0.2
    curriculum_difficulty_step: float = 0.1
    experience_buffer_size: int = 10000
    num_meta_steps: int = 5
    transfer_threshold: float = 0.7
    save_dir: str = "learning_data"

class LearningMetrics:
    """Metrics tracking for learning processes."""
    
    def __init__(self):
        """Initialize metrics tracking."""
        self.metrics: Dict[str, List[float]] = {
            "loss": [],
            "accuracy": [],
            "validation_loss": [],
            "validation_accuracy": [],
            "transfer_performance": [],
            "adaptation_speed": []
        }
        
    def update(self, metric_name: str, value: float) -> None:
        """Update a specific metric."""
        if metric_name in self.metrics:
            self.metrics[metric_name].append(value)
            
    def get_average(self, metric_name: str, window: int = 100) -> float:
        """Get moving average of a metric."""
        if metric_name not in self.metrics:
            return 0.0
        values = self.metrics[metric_name][-window:]
        return sum(values) / len(values) if values else 0.0
        
    def reset(self) -> None:
        """Reset all metrics."""
        for key in self.metrics:
            self.metrics[key] = []

class MetaLearner(nn.Module):
    """Neural network for meta-learning."""
    
    def __init__(self, config: LearningConfig):
        """Initialize meta-learner."""
        super().__init__()
        self.config = config
        
        # Meta-learning networks
        self.task_encoder = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        
        self.adaptation_network = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512)
        )
        
        self.optimizer = optim.Adam(
            self.parameters(),
            lr=config.meta_learning_rate
        )
        
    def forward(
        self,
        task_data: torch.Tensor,
        support_set: torch.Tensor,
        query_set: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of meta-learner."""
        # Encode task
        task_encoding = self.task_encoder(task_data)
        
        # Generate task-specific adaptation
        adaptation_params = self.adaptation_network(task_encoding)
        
        # Apply adaptation to support and query sets
        adapted_support = self._adapt_data(support_set, adaptation_params)
        adapted_query = self._adapt_data(query_set, adaptation_params)
        
        return adapted_support, adapted_query
        
    def _adapt_data(
        self,
        data: torch.Tensor,
        adaptation_params: torch.Tensor
    ) -> torch.Tensor:
        """Apply adaptation to data."""
        return data + adaptation_params

class CurriculumManager:
    """Manager for curriculum learning."""
    
    def __init__(self, config: LearningConfig):
        """Initialize curriculum manager."""
        self.config = config
        self.current_difficulty = 0.0
        self.task_performance: Dict[str, float] = {}
        self.task_history: List[Dict[str, Any]] = []
        
    def update_difficulty(self, performance: float) -> None:
        """Update curriculum difficulty based on performance."""
        if performance > 0.8:  # Threshold for increasing difficulty
            self.current_difficulty = min(
                1.0,
                self.current_difficulty + self.config.curriculum_difficulty_step
            )
        elif performance < 0.4:  # Threshold for decreasing difficulty
            self.current_difficulty = max(
                0.0,
                self.current_difficulty - self.config.curriculum_difficulty_step
            )
            
    def select_next_task(
        self,
        available_tasks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Select next task based on current difficulty."""
        # Filter tasks by difficulty
        suitable_tasks = [
            task for task in available_tasks
            if abs(task["difficulty"] - self.current_difficulty) < 0.2
        ]
        
        if not suitable_tasks:
            return available_tasks[0]  # Fallback to first task
            
        # Select task with best learning potential
        return max(
            suitable_tasks,
            key=lambda t: self._calculate_task_potential(t)
        )
        
    def _calculate_task_potential(self, task: Dict[str, Any]) -> float:
        """Calculate learning potential of a task."""
        current_performance = self.task_performance.get(task["id"], 0.0)
        times_attempted = sum(
            1 for h in self.task_history if h["task_id"] == task["id"]
        )
        
        # Balance between exploration and exploitation
        exploration_factor = 1.0 / (1.0 + times_attempted)
        exploitation_factor = 1.0 - current_performance
        
        return 0.7 * exploration_factor + 0.3 * exploitation_factor

class LearningSystem:
    """Core learning system implementation."""
    
    def __init__(
        self,
        config: LearningConfig,
        memory: MemorySystem,
        reasoning: ReasoningEngine,
        consciousness: ConsciousnessCore
    ):
        """Initialize learning system."""
        self.config = config
        self.memory = memory
        self.reasoning = reasoning
        self.consciousness = consciousness
        self.logger = logging.getLogger("learning_system")
        
        # Create save directory
        Path(config.save_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.meta_learner = MetaLearner(config)
        self.curriculum_manager = CurriculumManager(config)
        self.metrics = LearningMetrics()
        
        # Initialize learning state
        self.current_task: Optional[Dict[str, Any]] = None
        self.experience_buffer: List[Dict[str, Any]] = []
        self.learned_skills: Dict[str, Dict[str, Any]] = {}
        
    async def learn(
        self,
        task_data: Dict[str, Any],
        mode: LearningMode = LearningMode.SUPERVISED
    ) -> Dict[str, Any]:
        """Main learning function."""
        try:
            # Prepare learning task
            prepared_task = await self._prepare_task(task_data, mode)
            
            # Execute learning based on mode
            if mode == LearningMode.SUPERVISED:
                results = await self._supervised_learning(prepared_task)
            elif mode == LearningMode.UNSUPERVISED:
                results = await self._unsupervised_learning(prepared_task)
            elif mode == LearningMode.REINFORCEMENT:
                results = await self._reinforcement_learning(prepared_task)
            elif mode == LearningMode.META:
                results = await self._meta_learning(prepared_task)
            elif mode == LearningMode.ACTIVE:
                results = await self._active_learning(prepared_task)
            elif mode == LearningMode.CURRICULUM:
                results = await self._curriculum_learning(prepared_task)
            elif mode == LearningMode.TRANSFER:
                results = await self._transfer_learning(prepared_task)
            elif mode == LearningMode.CONTINUAL:
                results = await self._continual_learning(prepared_task)
            else:
                raise ValueError(f"Unknown learning mode: {mode}")
                
            # Update metrics
            self._update_metrics(results)
            
            # Store learning experience
            await self._store_experience(prepared_task, results)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in learning process: {str(e)}")
            return {"success": False, "error": str(e)}
            
    async def adapt_to_task(
        self,
        task_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Adapt learned knowledge to new task."""
        # Find similar tasks
        similar_tasks = await self._find_similar_tasks(task_data)
        
        # Generate adaptation strategy
        adaptation = await self._generate_adaptation(task_data, similar_tasks)
        
        # Apply adaptation
        adapted_knowledge = await self._apply_adaptation(adaptation)
        
        # Validate adaptation
        validation_results = await self._validate_adaptation(
            adapted_knowledge,
            task_data
        )
        
        if validation_results["success"]:
            # Store successful adaptation
            await self._store_adaptation(adaptation, validation_results)
            
        return {
            "success": validation_results["success"],
            "adaptation": adaptation,
            "validation": validation_results
        }
        
    async def evaluate_performance(
        self,
        task_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Evaluate learning performance on a task."""
        # Prepare evaluation data
        eval_data = await self._prepare_evaluation(task_data)
        
        # Perform evaluation
        results = await self._run_evaluation(eval_data)
        
        # Analyze results
        analysis = await self._analyze_evaluation(results)
        
        # Update metrics
        self._update_metrics({
            "evaluation": results,
            "analysis": analysis
        })
        
        return {
            "results": results,
            "analysis": analysis,
            "metrics": self.metrics.metrics
        }
        
    async def _supervised_learning(
        self,
        task_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Implement supervised learning."""
        # Extract training data
        inputs = torch.tensor(task_data["inputs"])
        targets = torch.tensor(task_data["targets"])
        
        # Create data loaders
        train_loader, val_loader = self._create_data_loaders(
            inputs,
            targets
        )
        
        # Training loop
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config.max_epochs):
            # Training phase
            train_loss = await self._train_epoch(train_loader)
            
            # Validation phase
            val_loss = await self._validate_epoch(val_loader)
            
            # Update metrics
            self.metrics.update("loss", train_loss)
            self.metrics.update("validation_loss", val_loss)
            
            # Early stopping check
            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= self.config.patience:
                break
                
        return {
            "success": True,
            "final_loss": train_loss,
            "best_validation_loss": best_loss,
            "epochs_trained": epoch + 1
        }
        
    async def _unsupervised_learning(
        self,
        task_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Implement unsupervised learning."""
        # Extract data
        data = torch.tensor(task_data["data"])
        
        # Create data loader
        data_loader = self._create_unsupervised_loader(data)
        
        # Training loop
        for epoch in range(self.config.max_epochs):
            # Training phase
            loss = await self._train_unsupervised_epoch(data_loader)
            
            # Update metrics
            self.metrics.update("loss", loss)
            
            # Extract and analyze patterns
            patterns = await self._extract_patterns(data_loader)
            
            # Update knowledge base with discovered patterns
            await self._update_knowledge_base(patterns)
            
        return {
            "success": True,
            "final_loss": loss,
            "discovered_patterns": patterns
        }
        
    async def _reinforcement_learning(
        self,
        task_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Implement reinforcement learning."""
        # Initialize environment
        env = task_data["environment"]
        
        # Training loop
        total_rewards = []
        
        for episode in range(self.config.max_epochs):
            # Run episode
            episode_rewards = await self._run_episode(env)
            
            # Update policy
            policy_loss = await self._update_policy(episode_rewards)
            
            # Update metrics
            self.metrics.update("episode_reward", sum(episode_rewards))
            self.metrics.update("policy_loss", policy_loss)
            
            total_rewards.append(sum(episode_rewards))
            
        return {
            "success": True,
            "average_reward": np.mean(total_rewards),
            "best_reward": max(total_rewards),
            "episodes_trained": episode + 1
        }
        
    async def _meta_learning(
        self,
        task_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Implement meta-learning."""
        # Prepare meta-learning data
        support_set = task_data["support_set"]
        query_set = task_data["query_set"]
        
        # Meta-training loop
        for step in range(self.config.num_meta_steps):
            # Inner loop - task adaptation
            adapted_support, adapted_query = self.meta_learner(
                task_data["task_encoding"],
                support_set,
                query_set
            )
            
            # Compute adaptation loss
            adaptation_loss = self._compute_adaptation_loss(
                adapted_support,
                adapted_query
            )
            
            # Update meta-learner
            self.meta_learner.optimizer.zero_grad()
            adaptation_loss.backward()
            self.meta_learner.optimizer.step()
            
            # Update metrics
            self.metrics.update("adaptation_loss", adaptation_loss.item())
            
        return {
            "success": True,
            "final_adaptation_loss": adaptation_loss.item(),
            "meta_steps": step + 1
        }
        
    async def _active_learning(
        self,
        task_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Implement active learning."""
        # Initialize query strategy
        query_strategy = self._initialize_query_strategy()
        
        # Active learning loop
        labeled_data = []
        unlabeled_data = task_data["unlabeled_data"]
        
        while len(unlabeled_data) > 0:
            # Select samples to label
            selected_indices = query_strategy.select_samples(unlabeled_data)
            
            # Get labels for selected samples
            new_labels = await self._get_labels(
                [unlabeled_data[i] for i in selected_indices]
            )
            
            # Update labeled dataset
            labeled_data.extend(zip(
                [unlabeled_data[i] for i in selected_indices],
                new_labels
            ))
            
            # Remove labeled samples from unlabeled pool
            unlabeled_data = [
                x for i, x in enumerate(unlabeled_data)
                if i not in selected_indices
            ]
            
            # Train on current labeled dataset
            train_results = await self._train_on_labeled_data(labeled_data)
            
            # Update metrics
            self.metrics.update("active_learning_accuracy", train_results["accuracy"])
            
        return {
            "success": True,
            "final_accuracy": train_results["accuracy"],
            "num_queries": len(labeled_data)
        }
        
    async def _curriculum_learning(
        self,
        task_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Implement curriculum learning."""
        available_tasks = task_data["tasks"]
        results_history = []
        
        while True:
            # Select next task
            current_task = self.curriculum_manager.select_next_task(
                available_tasks
            )
            
            # Train on task
            task_results = await self._train_on_task(current_task)
            
            # Update curriculum
            self.curriculum_manager.update_difficulty(
                task_results["performance"]
            )
            
            # Store results
            results_history.append({
                "task_id": current_task["id"],
                "difficulty": current_task["difficulty"],
                "performance": task_results["performance"]
            })
            
            # Check completion criteria
            if self._curriculum_completed(results_history):
                break
                
        return {
            "success": True,
            "results_history": results_history,
            "final_difficulty": self.curriculum_manager.current_difficulty
        }
        
    async def _transfer_learning(
        self,
        task_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Implement transfer learning."""
        # Analyze source and target tasks
        source_task = task_data["source_task"]
        target_task = task_data["target_task"]
        
        # Extract transferable knowledge
        transferable_knowledge = await self._extract_transferable_knowledge(
            source_task
        )
        
        # Adapt knowledge to target task
        adapted_knowledge = await self._adapt_knowledge(
            transferable_knowledge,
            target_task
        )
        
        # Fine-tune on target task
        fine_tune_results = await self._fine_tune(
            adapted_knowledge,
            target_task
        )
        
        # Evaluate transfer effectiveness
        transfer_metrics = await self._evaluate_transfer(
            source_task,
            target_task,
            fine_tune_results
        )
        
        return {
            "success": True,
            "transfer_metrics": transfer_metrics,
            "fine_tune_results": fine_tune_results
        }
        
    async def _continual_learning(
        self,
        task_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Implement continual learning."""
        # Initialize replay buffer
        replay_buffer = self._initialize_replay_buffer()
        
        # Process tasks sequentially
        results_history = []
        
        for task in task_data["task_sequence"]:
            # Train on current task
            task_results = await self._train_with_replay(
                task,
                replay_buffer
            )
            
            # Update replay buffer
            await self._update_replay_buffer(
                replay_buffer,
                task,
                task_results
            )
            
            # Evaluate on all previous tasks
            evaluation_results = await self._evaluate_all_tasks(
                task_data["task_sequence"][:task_data["task_sequence"].index(task) + 1]
            )
            
            # Store results
            results_history.append({
                "task_id": task["id"],
                "task_results": task_results,
                "evaluation_results": evaluation_results
            })
            
        return {
            "success": True,
            "results_history": results_history,
            "final_evaluation": evaluation_results
        }
        
    async def _store_experience(
        self,
        task_data: Dict[str, Any],
        results: Dict[str, Any]
    ) -> None:
        """Store learning experience in memory."""
        # Create experience record
        experience = {
            "id": uuid4(),
            "timestamp": datetime.now().isoformat(),
            "task_data": task_data,
            "results": results,
            "metrics": {
                k: self.metrics.get_average(k)
                for k in self.metrics.metrics
            }
        }
        
        # Add to experience buffer
        self.experience_buffer.append(experience)
        
        # Trim buffer if needed
        if len(self.experience_buffer) > self.config.experience_buffer_size:
            self.experience_buffer = self.experience_buffer[
                -self.config.experience_buffer_size:
            ]
            
        # Store in long-term memory
        await self.memory.store_episodic(
            episode_data=experience,
            context={"type": "learning_experience"}
        )
        
    def save_state(self, save_path: str) -> None:
        """Save learning system state."""
        state = {
            "config": self.config.__dict__,
            "metrics": self.metrics.metrics,
            "curriculum_state": {
                "current_difficulty": self.curriculum_manager.current_difficulty,
                "task_performance": self.curriculum_manager.task_performance
            },
            "learned_skills": self.learned_skills,
            "experience_buffer": [
                {
                    "id": str(exp["id"]),
                    "timestamp": exp["timestamp"],
                    "task_data": exp["task_data"],
                    "results": exp["results"],
                    "metrics": exp["metrics"]
                }
                for exp in self.experience_buffer
            ]
        }
        
        # Save neural network states
        torch.save(
            self.meta_learner.state_dict(),
            str(Path(save_path).with_suffix(".pth"))
        )
        
        # Save learning state
        with open(save_path, "w") as f:
            json.dump(state, f, indent=2, default=str)
            
    def load_state(self, load_path: str) -> None:
        """Load learning system state."""
        # Load neural network states
        self.meta_learner.load_state_dict(
            torch.load(str(Path(load_path).with_suffix(".pth")))
        )
        
        # Load learning state
        with open(load_path, "r") as f:
            state = json.load(f)
            
        self.config = LearningConfig(**state["config"])
        self.metrics.metrics = state["metrics"]
        self.curriculum_manager.current_difficulty = state["curriculum_state"]["current_difficulty"]
        self.curriculum_manager.task_performance = state["curriculum_state"]["task_performance"]
        self.learned_skills = state["learned_skills"]
        
        self.experience_buffer = [
            {
                "id": UUID(exp["id"]),
                "timestamp": exp["timestamp"],
                "task_data": exp["task_data"],
                "results": exp["results"],
                "metrics": exp["metrics"]
            }
            for exp in state["experience_buffer"]
        ] 