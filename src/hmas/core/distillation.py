"""Knowledge distillation module for H-MAS distributed training."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
import numpy as np
from pathlib import Path
import logging
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

@dataclass
class DistillationConfig:
    """Configuration for knowledge distillation."""
    temperature: float = 2.0
    alpha: float = 0.5  # Balance between hard and soft targets
    batch_size: int = 64
    num_epochs: int = 5
    learning_rate: float = 1e-4
    validation_split: float = 0.2
    patience: int = 3  # Early stopping patience
    use_cuda: bool = torch.cuda.is_available()

class KnowledgeDistiller:
    """Handles knowledge distillation between teacher and student models."""
    
    def __init__(self, config: DistillationConfig):
        """Initialize knowledge distiller."""
        self.config = config
        self.device = torch.device("cuda" if config.use_cuda else "cpu")
        self.logger = logging.getLogger("knowledge_distiller")
        
    def prepare_data(
        self,
        teacher_outputs: torch.Tensor,
        hard_targets: torch.Tensor
    ) -> Tuple[DataLoader, DataLoader]:
        """Prepare training and validation data loaders."""
        # Split data into training and validation
        num_samples = len(teacher_outputs)
        indices = np.random.permutation(num_samples)
        split_idx = int(num_samples * (1 - self.config.validation_split))
        
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]
        
        # Create datasets
        train_dataset = TensorDataset(
            teacher_outputs[train_indices],
            hard_targets[train_indices]
        )
        val_dataset = TensorDataset(
            teacher_outputs[val_indices],
            hard_targets[val_indices]
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False
        )
        
        return train_loader, val_loader
        
    def distillation_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        hard_targets: torch.Tensor,
        temperature: float
    ) -> torch.Tensor:
        """Calculate the distillation loss combining soft and hard targets."""
        # Soft targets with temperature scaling
        soft_targets = F.softmax(teacher_logits / temperature, dim=1)
        soft_prob = F.log_softmax(student_logits / temperature, dim=1)
        soft_loss = F.kl_div(soft_prob, soft_targets, reduction='batchmean')
        
        # Hard targets
        hard_loss = F.cross_entropy(student_logits, hard_targets)
        
        # Combined loss
        loss = (
            self.config.alpha * temperature * temperature * soft_loss +
            (1 - self.config.alpha) * hard_loss
        )
        
        return loss
        
    def train_epoch(
        self,
        student_model: nn.Module,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer
    ) -> float:
        """Train for one epoch."""
        student_model.train()
        total_loss = 0.0
        num_batches = 0
        
        for teacher_batch, target_batch in train_loader:
            teacher_batch = teacher_batch.to(self.device)
            target_batch = target_batch.to(self.device)
            
            optimizer.zero_grad()
            student_output = student_model(teacher_batch)
            
            loss = self.distillation_loss(
                student_output,
                teacher_batch,
                target_batch,
                self.config.temperature
            )
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
        return total_loss / num_batches
        
    def validate(
        self,
        student_model: nn.Module,
        val_loader: DataLoader
    ) -> float:
        """Validate model performance."""
        student_model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for teacher_batch, target_batch in val_loader:
                teacher_batch = teacher_batch.to(self.device)
                target_batch = target_batch.to(self.device)
                
                student_output = student_model(teacher_batch)
                loss = self.distillation_loss(
                    student_output,
                    teacher_batch,
                    target_batch,
                    self.config.temperature
                )
                
                total_loss += loss.item()
                num_batches += 1
                
        return total_loss / num_batches
        
    def distill_knowledge(
        self,
        student_model: nn.Module,
        teacher_outputs: torch.Tensor,
        hard_targets: torch.Tensor
    ) -> Tuple[nn.Module, Dict[str, List[float]]]:
        """Perform knowledge distillation from teacher to student."""
        student_model = student_model.to(self.device)
        teacher_outputs = teacher_outputs.to(self.device)
        hard_targets = hard_targets.to(self.device)
        
        # Prepare data loaders
        train_loader, val_loader = self.prepare_data(teacher_outputs, hard_targets)
        
        # Setup optimizer
        optimizer = Adam(student_model.parameters(), lr=self.config.learning_rate)
        
        # Training history
        history = {
            "train_loss": [],
            "val_loss": []
        }
        
        # Early stopping variables
        best_val_loss = float('inf')
        best_model_state = student_model.state_dict()
        patience_counter = 0
        
        # Training loop
        for epoch in range(self.config.num_epochs):
            train_loss = self.train_epoch(student_model, train_loader, optimizer)
            val_loss = self.validate(student_model, val_loader)
            
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            
            self.logger.info(
                f"Epoch {epoch + 1}/{self.config.num_epochs} - "
                f"Train Loss: {train_loss:.4f} - "
                f"Val Loss: {val_loss:.4f}"
            )
            
            # Check for improvement
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = student_model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1
                
            # Early stopping
            if patience_counter >= self.config.patience:
                self.logger.info("Early stopping triggered")
                break
                
        # Restore best model
        student_model.load_state_dict(best_model_state)
        
        return student_model, history
        
    def ensemble_distillation(
        self,
        student_model: nn.Module,
        teacher_models: List[nn.Module],
        validation_data: Tuple[torch.Tensor, torch.Tensor]
    ) -> nn.Module:
        """Perform ensemble knowledge distillation from multiple teachers."""
        student_model = student_model.to(self.device)
        for teacher in teacher_models:
            teacher.to(self.device)
            teacher.eval()
            
        val_inputs, val_targets = validation_data
        val_inputs = val_inputs.to(self.device)
        val_targets = val_targets.to(self.device)
        
        # Generate ensemble predictions
        ensemble_outputs = []
        with torch.no_grad():
            for teacher in teacher_models:
                teacher_output = teacher(val_inputs)
                ensemble_outputs.append(teacher_output)
                
        # Average ensemble predictions
        ensemble_output = torch.mean(torch.stack(ensemble_outputs), dim=0)
        
        # Perform distillation
        student_model, _ = self.distill_knowledge(
            student_model,
            ensemble_output,
            val_targets
        )
        
        return student_model
        
    def incremental_distillation(
        self,
        base_model: nn.Module,
        new_data: Tuple[torch.Tensor, torch.Tensor],
        old_data: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        memory_size: int = 1000
    ) -> nn.Module:
        """Perform incremental knowledge distillation for continual learning."""
        base_model = base_model.to(self.device)
        new_inputs, new_targets = new_data
        new_inputs = new_inputs.to(self.device)
        new_targets = new_targets.to(self.device)
        
        # If we have old data, combine it with new data
        if old_data is not None:
            old_inputs, old_targets = old_data
            old_inputs = old_inputs.to(self.device)
            old_targets = old_targets.to(self.device)
            
            # Generate old task predictions
            base_model.eval()
            with torch.no_grad():
                old_predictions = base_model(old_inputs)
                
            # Combine data
            combined_inputs = torch.cat([old_inputs[:memory_size], new_inputs])
            combined_targets = torch.cat([old_targets[:memory_size], new_targets])
            teacher_outputs = torch.cat([old_predictions[:memory_size], new_targets])
        else:
            combined_inputs = new_inputs
            combined_targets = new_targets
            teacher_outputs = new_targets
            
        # Create new model instance
        new_model = type(base_model)().to(self.device)
        new_model.load_state_dict(base_model.state_dict())
        
        # Perform distillation
        new_model, _ = self.distill_knowledge(
            new_model,
            teacher_outputs,
            combined_targets
        )
        
        return new_model
        
    def save_distilled_model(
        self,
        model: nn.Module,
        save_path: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Save distilled model with metadata."""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare save data
        save_data = {
            "model_state": model.state_dict(),
            "config": self.config.__dict__,
            "metadata": metadata or {}
        }
        
        torch.save(save_data, save_path)
        
    def load_distilled_model(
        self,
        model: nn.Module,
        load_path: str
    ) -> Tuple[nn.Module, Dict[str, Any]]:
        """Load distilled model and return metadata."""
        load_path = Path(load_path)
        if not load_path.exists():
            raise FileNotFoundError(f"Model file not found: {load_path}")
            
        save_data = torch.load(load_path, map_location=self.device)
        model.load_state_dict(save_data["model_state"])
        
        return model, save_data["metadata"] 