import os
import logging.config
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Any, Union, Tuple
from enum import Enum
from datetime import datetime
import json
import asyncio
import numpy as np
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.util import get_remote_address
from prometheus_fastapi_instrumentator import Instrumentator
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import gymnasium as gym

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Setup logging
LOGGING_CONFIG_PATH = "logging.conf"
if os.path.exists(LOGGING_CONFIG_PATH):
    logging.config.fileConfig(LOGGING_CONFIG_PATH, disable_existing_loggers=False)
logger = logging.getLogger(__name__)

class LearningType(str, Enum):
    REINFORCEMENT = "reinforcement"
    META = "meta"
    CURRICULUM = "curriculum"
    COALITION = "coalition"
    TRANSFER = "transfer"

class ActionSpace(str, Enum):
    DISCRETE = "discrete"
    CONTINUOUS = "continuous"
    HYBRID = "hybrid"

class RewardType(str, Enum):
    SPARSE = "sparse"
    DENSE = "dense"
    HIERARCHICAL = "hierarchical"

class LearningConfig(BaseModel):
    learning_type: LearningType
    action_space: ActionSpace
    reward_type: RewardType
    state_dim: int
    action_dim: int
    hidden_dim: int = 256
    learning_rate: float = 0.001
    gamma: float = 0.99
    tau: float = 0.005
    batch_size: int = 64
    buffer_size: int = 1000000
    min_steps_before_learning: int = 1000
    curriculum_stages: Optional[List[Dict[str, Any]]] = None
    meta_learning_params: Optional[Dict[str, Any]] = None
    coalition_params: Optional[Dict[str, Any]] = None

class Experience(BaseModel):
    state: List[float]
    action: List[float]
    reward: float
    next_state: List[float]
    done: bool
    info: Optional[Dict[str, Any]] = None

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer: List[Experience] = []
        self.position = 0
        
    def push(self, experience: Experience):
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size: int) -> List[Experience]:
        return np.random.choice(self.buffer, batch_size).tolist()
        
    def __len__(self) -> int:
        return len(self.buffer)

class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)

class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=1)
        return self.net(x)

class MetaLearner:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.meta_model = nn.ModuleDict({
            "encoder": nn.Sequential(
                nn.Linear(config["input_dim"], config["hidden_dim"]),
                nn.ReLU(),
                nn.Linear(config["hidden_dim"], config["latent_dim"])
            ),
            "policy": nn.Sequential(
                nn.Linear(config["latent_dim"], config["hidden_dim"]),
                nn.ReLU(),
                nn.Linear(config["hidden_dim"], config["output_dim"])
            )
        })
        self.optimizer = optim.Adam(
            self.meta_model.parameters(),
            lr=config["learning_rate"]
        )
        
    def adapt(self, task_data: torch.Tensor) -> nn.Module:
        """Adapt meta-model to new task."""
        latent = self.meta_model["encoder"](task_data)
        adapted_policy = self.meta_model["policy"]
        return adapted_policy
        
    def meta_update(self, task_batch: List[torch.Tensor]):
        """Update meta-model using batch of tasks."""
        total_loss = 0
        self.optimizer.zero_grad()
        
        for task_data in task_batch:
            adapted_policy = self.adapt(task_data)
            loss = self.compute_loss(adapted_policy, task_data)
            total_loss += loss
            
        total_loss.backward()
        self.optimizer.step()
        return total_loss.item()
        
    def compute_loss(self, policy: nn.Module, data: torch.Tensor) -> torch.Tensor:
        """Compute loss for meta-learning."""
        # Implement appropriate loss computation
        return torch.tensor(0.0, requires_grad=True)

class CurriculumManager:
    def __init__(self, stages: List[Dict[str, Any]]):
        self.stages = stages
        self.current_stage = 0
        self.stage_metrics: Dict[int, List[float]] = {}
        
    def get_current_task(self) -> Dict[str, Any]:
        """Get current curriculum task."""
        return self.stages[self.current_stage]
        
    def update_progress(self, metrics: List[float]):
        """Update progress and potentially advance curriculum."""
        stage = self.current_stage
        if stage not in self.stage_metrics:
            self.stage_metrics[stage] = []
        self.stage_metrics[stage].extend(metrics)
        
        if self.should_advance():
            self.advance_stage()
            
    def should_advance(self) -> bool:
        """Check if should advance to next stage."""
        if not self.stage_metrics.get(self.current_stage):
            return False
            
        recent_metrics = self.stage_metrics[self.current_stage][-10:]
        avg_performance = np.mean(recent_metrics)
        return avg_performance >= self.stages[self.current_stage]["threshold"]
        
    def advance_stage(self):
        """Advance to next curriculum stage."""
        if self.current_stage < len(self.stages) - 1:
            self.current_stage += 1
            logger.info(f"Advanced to curriculum stage {self.current_stage}")

class CoalitionLearner:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.agents: Dict[str, Any] = {}
        self.coalition_values: Dict[str, float] = {}
        
    def form_coalition(self, available_agents: List[str]) -> List[str]:
        """Form optimal coalition of agents."""
        best_coalition = []
        best_value = float("-inf")
        
        # Simple greedy coalition formation
        for agent_id in available_agents:
            coalition = best_coalition + [agent_id]
            value = self.evaluate_coalition(coalition)
            
            if value > best_value:
                best_coalition = coalition
                best_value = value
                
        return best_coalition
        
    def evaluate_coalition(self, coalition: List[str]) -> float:
        """Evaluate value of a coalition."""
        # Implement coalition value calculation
        return sum(self.coalition_values.get(agent_id, 0) for agent_id in coalition)
        
    def update_coalition_values(self, performance_metrics: Dict[str, float]):
        """Update coalition values based on performance."""
        for agent_id, metric in performance_metrics.items():
            if agent_id not in self.coalition_values:
                self.coalition_values[agent_id] = metric
            else:
                self.coalition_values[agent_id] = 0.9 * self.coalition_values[agent_id] + 0.1 * metric

class LearningManager:
    def __init__(self):
        self.configs: Dict[str, LearningConfig] = {}
        self.replay_buffers: Dict[str, ReplayBuffer] = {}
        self.actors: Dict[str, Actor] = {}
        self.critics: Dict[str, Critic] = {}
        self.meta_learners: Dict[str, MetaLearner] = {}
        self.curriculum_managers: Dict[str, CurriculumManager] = {}
        self.coalition_learners: Dict[str, CoalitionLearner] = {}
        self.optimizers: Dict[str, Dict[str, optim.Optimizer]] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        
    def register_config(self, config_id: str, config: LearningConfig):
        """Register learning configuration."""
        self.configs[config_id] = config
        
        # Initialize components based on config
        self.replay_buffers[config_id] = ReplayBuffer(config.buffer_size)
        
        self.actors[config_id] = Actor(
            config.state_dim,
            config.action_dim,
            config.hidden_dim
        )
        
        self.critics[config_id] = Critic(
            config.state_dim,
            config.action_dim,
            config.hidden_dim
        )
        
        self.optimizers[config_id] = {
            "actor": optim.Adam(
                self.actors[config_id].parameters(),
                lr=config.learning_rate
            ),
            "critic": optim.Adam(
                self.critics[config_id].parameters(),
                lr=config.learning_rate
            )
        }
        
        self.scalers[config_id] = StandardScaler()
        
        if config.meta_learning_params:
            self.meta_learners[config_id] = MetaLearner(config.meta_learning_params)
            
        if config.curriculum_stages:
            self.curriculum_managers[config_id] = CurriculumManager(
                config.curriculum_stages
            )
            
        if config.coalition_params:
            self.coalition_learners[config_id] = CoalitionLearner(
                config.coalition_params
            )
            
    async def process_experience(
        self,
        config_id: str,
        experience: Experience,
        background_tasks: BackgroundTasks
    ):
        """Process and learn from experience."""
        if config_id not in self.configs:
            raise HTTPException(
                status_code=404,
                detail=f"Configuration {config_id} not found"
            )
            
        try:
            # Store experience
            self.replay_buffers[config_id].push(experience)
            
            # Check if should perform learning update
            if len(self.replay_buffers[config_id]) >= self.configs[config_id].min_steps_before_learning:
                background_tasks.add_task(
                    self.update,
                    config_id
                )
                
        except Exception as e:
            logger.error(f"Error processing experience: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
            
    async def update(self, config_id: str):
        """Perform learning update."""
        config = self.configs[config_id]
        
        # Sample batch
        batch = self.replay_buffers[config_id].sample(config.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor([exp.state for exp in batch])
        actions = torch.FloatTensor([exp.action for exp in batch])
        rewards = torch.FloatTensor([exp.reward for exp in batch])
        next_states = torch.FloatTensor([exp.next_state for exp in batch])
        dones = torch.FloatTensor([exp.done for exp in batch])
        
        # Normalize states
        states = torch.FloatTensor(
            self.scalers[config_id].fit_transform(states.numpy())
        )
        next_states = torch.FloatTensor(
            self.scalers[config_id].transform(next_states.numpy())
        )
        
        # Update critic
        next_actions = self.actors[config_id](next_states)
        target_q = rewards + (1 - dones) * config.gamma * self.critics[config_id](
            next_states, next_actions
        ).detach()
        
        current_q = self.critics[config_id](states, actions)
        critic_loss = nn.MSELoss()(current_q, target_q)
        
        self.optimizers[config_id]["critic"].zero_grad()
        critic_loss.backward()
        self.optimizers[config_id]["critic"].step()
        
        # Update actor
        actor_loss = -self.critics[config_id](
            states,
            self.actors[config_id](states)
        ).mean()
        
        self.optimizers[config_id]["actor"].zero_grad()
        actor_loss.backward()
        self.optimizers[config_id]["actor"].step()
        
        # Soft update target networks if using DDPG
        if config.learning_type == LearningType.REINFORCEMENT:
            self.soft_update(config_id)
            
    def soft_update(self, config_id: str):
        """Soft update target networks."""
        config = self.configs[config_id]
        
        for target_param, param in zip(
            self.critics[config_id].parameters(),
            self.critics[config_id].parameters()
        ):
            target_param.data.copy_(
                config.tau * param.data + (1 - config.tau) * target_param.data
            )
            
        for target_param, param in zip(
            self.actors[config_id].parameters(),
            self.actors[config_id].parameters()
        ):
            target_param.data.copy_(
                config.tau * param.data + (1 - config.tau) * target_param.data
            )
            
    async def get_action(
        self,
        config_id: str,
        state: List[float],
        deterministic: bool = False
    ) -> List[float]:
        """Get action for given state."""
        if config_id not in self.configs:
            raise HTTPException(
                status_code=404,
                detail=f"Configuration {config_id} not found"
            )
            
        try:
            # Normalize state
            state = torch.FloatTensor(
                self.scalers[config_id].transform([state])
            )
            
            # Get action from actor
            with torch.no_grad():
                action = self.actors[config_id](state)
                
                if not deterministic:
                    action += torch.randn_like(action) * 0.1
                    action = torch.clamp(action, -1, 1)
                    
            return action.numpy().tolist()[0]
            
        except Exception as e:
            logger.error(f"Error getting action: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

# Initialize FastAPI app
@asynccontextmanager
async def lifespan(app: FastAPI):
    # STARTUP
    logger.info("Initializing learning service...")
    try:
        learning_manager = LearningManager()
        app.state.learning_manager = learning_manager
        logger.info("Learning service initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize learning service: {str(e)}")
        raise
        
    yield
    
    # SHUTDOWN
    logger.info("Shutting down learning service...")

app = FastAPI(title="HMAS Learning Service", lifespan=lifespan)

# Rate limiting
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

# Prometheus metrics
Instrumentator().instrument(app).expose(app)

@app.post("/config/{config_id}")
@limiter.limit("20/minute")
async def register_config(
    request: Request,
    config_id: str,
    config: LearningConfig
):
    """Register a learning configuration."""
    try:
        request.app.state.learning_manager.register_config(config_id, config)
        return {"status": "success", "config_id": config_id}
    except Exception as e:
        logger.error(f"Error registering configuration: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/experience/{config_id}")
@limiter.limit("1000/minute")
async def process_experience(
    request: Request,
    config_id: str,
    experience: Experience,
    background_tasks: BackgroundTasks
):
    """Process and learn from experience."""
    try:
        await request.app.state.learning_manager.process_experience(
            config_id,
            experience,
            background_tasks
        )
        return {"status": "success"}
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error processing experience: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/action/{config_id}")
@limiter.limit("1000/minute")
async def get_action(
    request: Request,
    config_id: str,
    state: List[float],
    deterministic: bool = False
):
    """Get action for given state."""
    try:
        action = await request.app.state.learning_manager.get_action(
            config_id,
            state,
            deterministic
        )
        return {"action": action}
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error getting action: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8300) 