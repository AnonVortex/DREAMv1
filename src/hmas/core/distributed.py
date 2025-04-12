"""Distributed training system for H-MAS agents."""

import asyncio
import ray
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from datetime import datetime
import json
import logging
from .environments import EnvironmentType
from .training import TrainingConfig, TrainingOrchestrator
from .visualization import VisualizationConfig, PerformanceTracker

@dataclass
class DistributedConfig:
    """Configuration for distributed training."""
    num_workers: int = 4
    sync_interval: int = 100
    checkpoint_interval: int = 1000
    experience_batch_size: int = 64
    parameter_sync_method: str = "average"  # "average" or "distillation"
    redis_address: Optional[str] = None
    checkpoint_dir: str = "checkpoints"
    log_level: str = "INFO"

@ray.remote
class TrainingWorker:
    """Worker node for distributed training."""
    
    def __init__(
        self,
        worker_id: int,
        config: DistributedConfig,
        training_config: TrainingConfig,
        visualization_config: VisualizationConfig
    ):
        """Initialize training worker."""
        self.worker_id = worker_id
        self.config = config
        
        # Initialize training components
        self.orchestrator = TrainingOrchestrator(training_config)
        self.performance_tracker = PerformanceTracker(visualization_config)
        
        # Setup logging
        self.logger = logging.getLogger(f"worker_{worker_id}")
        self.logger.setLevel(config.log_level)
        
        # Initialize experience buffer
        self.experience_buffer = []
        
    async def train_step(self) -> Dict[str, Any]:
        """Execute one training step."""
        # Run training step
        step_result = await self.orchestrator.run_episode()
        
        # Update metrics
        self.performance_tracker.update_metrics(
            step_result["environment_type"],
            step_result["metrics"]
        )
        
        # Add to experience buffer
        self.experience_buffer.append(step_result["experience"])
        
        # Return step metrics
        return {
            "worker_id": self.worker_id,
            "metrics": step_result["metrics"],
            "environment_type": step_result["environment_type"]
        }
        
    def get_experience_batch(self) -> List[Dict[str, Any]]:
        """Get and clear current experience batch."""
        batch = self.experience_buffer[:self.config.experience_batch_size]
        self.experience_buffer = self.experience_buffer[self.config.experience_batch_size:]
        return batch
        
    def update_parameters(self, new_parameters: Dict[str, torch.Tensor]) -> None:
        """Update model parameters."""
        self.orchestrator.update_agent_parameters(new_parameters)
        
    def get_parameters(self) -> Dict[str, torch.Tensor]:
        """Get current model parameters."""
        return self.orchestrator.get_agent_parameters()
        
    def save_checkpoint(self) -> str:
        """Save worker checkpoint."""
        checkpoint_path = Path(self.config.checkpoint_dir) / f"worker_{self.worker_id}"
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        # Save training state
        self.orchestrator.save_checkpoint(str(checkpoint_path / "training.pt"))
        
        # Save metrics
        self.performance_tracker.save_metrics_history()
        
        return str(checkpoint_path)
        
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load worker checkpoint."""
        checkpoint_path = Path(checkpoint_path)
        
        # Load training state
        self.orchestrator.load_checkpoint(str(checkpoint_path / "training.pt"))
        
        # Load metrics
        metrics_path = checkpoint_path / "metrics_history.json"
        if metrics_path.exists():
            self.performance_tracker.load_metrics_history(str(metrics_path))

class DistributedTrainer:
    """Coordinates distributed training across workers."""
    
    def __init__(
        self,
        config: DistributedConfig,
        training_config: TrainingConfig,
        visualization_config: VisualizationConfig
    ):
        """Initialize distributed trainer."""
        self.config = config
        self.training_config = training_config
        self.visualization_config = visualization_config
        
        # Initialize Ray
        if not ray.is_initialized():
            ray.init(
                address=config.redis_address,
                logging_level=config.log_level
            )
            
        # Create workers
        self.workers = [
            TrainingWorker.remote(
                i,
                config,
                training_config,
                visualization_config
            )
            for i in range(config.num_workers)
        ]
        
        # Setup logging
        self.logger = logging.getLogger("distributed_trainer")
        self.logger.setLevel(config.log_level)
        
        # Initialize performance tracking
        self.performance_tracker = PerformanceTracker(visualization_config)
        
    async def synchronize_parameters(self) -> None:
        """Synchronize parameters across workers."""
        # Gather parameters from all workers
        parameter_futures = [
            worker.get_parameters.remote()
            for worker in self.workers
        ]
        parameters_list = await asyncio.gather(*[
            asyncio.to_thread(ray.get, future)
            for future in parameter_futures
        ])
        
        if self.config.parameter_sync_method == "average":
            # Compute average parameters
            averaged_parameters = {}
            for key in parameters_list[0].keys():
                averaged_parameters[key] = torch.mean(
                    torch.stack([p[key] for p in parameters_list]),
                    dim=0
                )
            new_parameters = averaged_parameters
        else:  # distillation
            # TODO: Implement knowledge distillation
            new_parameters = parameters_list[0]
            
        # Update all workers with new parameters
        update_futures = [
            worker.update_parameters.remote(new_parameters)
            for worker in self.workers
        ]
        await asyncio.gather(*[
            asyncio.to_thread(ray.get, future)
            for future in update_futures
        ])
        
    async def share_experiences(self) -> None:
        """Share experiences between workers."""
        # Gather experiences from all workers
        experience_futures = [
            worker.get_experience_batch.remote()
            for worker in self.workers
        ]
        experiences_list = await asyncio.gather(*[
            asyncio.to_thread(ray.get, future)
            for future in experience_futures
        ])
        
        # Flatten and shuffle experiences
        all_experiences = []
        for experiences in experiences_list:
            all_experiences.extend(experiences)
        np.random.shuffle(all_experiences)
        
        # Split experiences evenly among workers
        batch_size = len(all_experiences) // len(self.workers)
        for i, worker in enumerate(self.workers):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size if i < len(self.workers) - 1 else len(all_experiences)
            worker_experiences = all_experiences[start_idx:end_idx]
            # TODO: Implement experience replay buffer update
            
    async def save_distributed_checkpoint(self) -> None:
        """Save checkpoint for all workers."""
        checkpoint_futures = [
            worker.save_checkpoint.remote()
            for worker in self.workers
        ]
        checkpoint_paths = await asyncio.gather(*[
            asyncio.to_thread(ray.get, future)
            for future in checkpoint_futures
        ])
        
        # Save trainer state
        trainer_state = {
            "config": self.config.__dict__,
            "checkpoint_paths": checkpoint_paths,
            "timestamp": datetime.now().isoformat()
        }
        
        checkpoint_path = Path(self.config.checkpoint_dir) / "trainer_state.json"
        with open(checkpoint_path, "w") as f:
            json.dump(trainer_state, f, indent=2)
            
    async def load_distributed_checkpoint(self, checkpoint_dir: str) -> None:
        """Load checkpoint for all workers."""
        checkpoint_dir = Path(checkpoint_dir)
        
        # Load trainer state
        with open(checkpoint_dir / "trainer_state.json", "r") as f:
            trainer_state = json.load(f)
            
        # Load worker checkpoints
        load_futures = [
            worker.load_checkpoint.remote(path)
            for worker, path in zip(self.workers, trainer_state["checkpoint_paths"])
        ]
        await asyncio.gather(*[
            asyncio.to_thread(ray.get, future)
            for future in load_futures
        ])
        
    async def train(
        self,
        num_steps: int,
        checkpoint_dir: Optional[str] = None
    ) -> None:
        """Run distributed training."""
        # Load checkpoint if provided
        if checkpoint_dir:
            await self.load_distributed_checkpoint(checkpoint_dir)
            
        for step in range(num_steps):
            # Run training steps on all workers
            step_futures = [
                worker.train_step.remote()
                for worker in self.workers
            ]
            step_results = await asyncio.gather(*[
                asyncio.to_thread(ray.get, future)
                for future in step_futures
            ])
            
            # Update global metrics
            for result in step_results:
                self.performance_tracker.update_metrics(
                    result["environment_type"],
                    result["metrics"]
                )
                
            # Periodic synchronization
            if step > 0 and step % self.config.sync_interval == 0:
                await self.synchronize_parameters()
                await self.share_experiences()
                
            # Periodic checkpointing
            if step > 0 and step % self.config.checkpoint_interval == 0:
                await self.save_distributed_checkpoint()
                
            # Log progress
            if step % 100 == 0:
                self.logger.info(f"Step {step}/{num_steps} completed")
                
        # Final checkpoint
        await self.save_distributed_checkpoint()
        
    def shutdown(self) -> None:
        """Shutdown distributed training."""
        ray.shutdown() 