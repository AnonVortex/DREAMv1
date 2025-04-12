import os
import logging
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from collections import defaultdict

from .learning_agent import LearningAgent
from .memory_client import MemoryClient
from .task_reward_manager import TaskRewardManager, RewardConfig
from .action_spaces import ActionSpaceFactory, ActionSpace
from .curriculum_manager import CurriculumManager
from ..environments.tasks_3d import TaskType, create_task
from .message_types import MessageType
from ..multi_agent.role_manager import RoleManager, RoleType
from ..multi_agent.task_allocator import TaskAllocator, TaskPriority, TaskRequirements, TaskStatus
from ..multi_agent.coalition_manager import CoalitionManager, CoalitionStatus
from ..utils.metrics import MetricsTracker
from ..multi_agent.task_generator import TaskGenerator, TaskDifficulty
from ..multi_agent.coalition_communication import CoalitionCommunicationManager, MessageType, MessagePriority
from .coalition_learning_manager import CoalitionLearningManager, LearningStrategy, LearningConfig
from ..monitoring.system_monitor import SystemMonitor, MonitoringConfig, MonitoringLevel

logger = logging.getLogger(__name__)

class TrainingManager:
    """Manages the training process and environment interaction."""
    def __init__(
        self,
        learning_agent: LearningAgent,
        memory_client: MemoryClient,
        task_type: TaskType,
        reward_config: Optional[RewardConfig] = None,
        max_episodes: int = 1000,
        max_steps_per_episode: int = 500,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        checkpoint_frequency: int = 100,
        checkpoint_dir: str = "checkpoints",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        use_curriculum: bool = True,
        use_graph_rl: bool = False,
        graph_rl_manager=None,
        use_role_management: bool = True,
        use_task_allocation: bool = True,
        use_coalition_formation: bool = True,
        use_coalition_learning: bool = True,
        num_agents: int = 1,
        use_meta_learning: bool = False,
        use_dynamic_tasks: bool = True,
        curriculum_config: Optional[Dict[str, Any]] = None,
        meta_config: Optional[Dict[str, Any]] = None,
        coalition_config: Optional[Dict[str, Any]] = None,
        learning_config: Optional[Dict[str, Any]] = None,
        task_config: Optional[Dict[str, Any]] = None,
        communication_config: Optional[Dict[str, Any]] = None,
        monitoring_config: Optional[Dict[str, Any]] = None
    ):
        self.agent = learning_agent
        self.memory_client = memory_client
        self.task_type = task_type
        self.reward_manager = TaskRewardManager(task_type, reward_config)
        
        # Initialize curriculum manager if enabled
        self.use_curriculum = use_curriculum
        if use_curriculum:
            self.curriculum_manager = CurriculumManager(
                **curriculum_config if curriculum_config else {}
            )
        
        # Initialize action space
        self.action_space = ActionSpaceFactory.create_action_space(
            task_type=task_type.value,
            num_agents=getattr(reward_config, 'num_agents', 1)
        )
        
        self.max_episodes = max_episodes
        self.max_steps_per_episode = max_steps_per_episode
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.checkpoint_frequency = checkpoint_frequency
        self.checkpoint_dir = checkpoint_dir
        self.device = device
        
        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Training metrics
        self.writer = SummaryWriter(log_dir=os.path.join("logs", "training"))
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.best_reward = float("-inf")
        self.invalid_actions_count = 0
        
        self.use_graph_rl = use_graph_rl
        self.graph_rl_manager = graph_rl_manager
        
        # Initialize role manager if enabled
        self.use_role_management = use_role_management
        if use_role_management:
            self.role_manager = RoleManager(num_agents=num_agents)
            self.agent_capabilities = {}  # Will be populated during training
        
        # Initialize task allocator if enabled
        self.use_task_allocation = use_task_allocation
        if use_task_allocation:
            self.task_allocator = TaskAllocator()
        
        # Initialize coalition manager if enabled
        self.use_coalition_formation = use_coalition_formation
        if use_coalition_formation:
            self.coalition_manager = CoalitionManager(num_agents=num_agents)
            self.active_coalitions: Dict[str, Dict[str, Any]] = {}
        
        # Initialize meta-learning if enabled
        self.use_meta_learning = use_meta_learning
        if use_meta_learning:
            self.meta_learning_manager = MetaLearningManager(
                **meta_config if meta_config else {}
            )
        
        # Initialize coalition communication if enabled
        self.communication_manager = None
        if use_coalition_formation:
            self.communication_manager = CoalitionCommunicationManager(
                config=communication_config
            )
        
        # Initialize coalition learning if enabled
        self.use_coalition_learning = use_coalition_learning
        if use_coalition_learning:
            self.coalition_learning_manager = CoalitionLearningManager(
                config=LearningConfig(**learning_config) if learning_config else None
            )
        
        # Training state
        self.current_episode = 0
        self.best_performance = float('-inf')
        self.training_history: List[Dict[str, float]] = []
        self.metrics = MetricsTracker()
        
        # Initialize task generator if enabled
        self.use_dynamic_tasks = use_dynamic_tasks
        if use_dynamic_tasks:
            self.task_generator = TaskGenerator(
                base_difficulty=TaskDifficulty.BEGINNER,
                **task_config if task_config else {}
            )
        
        # Initialize system monitor
        self.system_monitor = SystemMonitor(
            config=MonitoringConfig(
                level=MonitoringLevel.DETAILED,
                update_interval=1.0,
                history_size=1000,
                alert_thresholds={
                    "cpu_usage": 90,
                    "memory_usage": 90,
                    "error_rate": 0.1
                },
                visualization_enabled=True,
                log_to_file=True,
                profile_enabled=True,
                **monitoring_config if monitoring_config else {}
            )
        )
        
    async def _create_training_task(self, episode: int) -> str:
        """Create a training task with appropriate requirements."""
        if not self.use_task_allocation:
            return None
            
        task_id = f"training_episode_{episode}"
        
        # Define task requirements based on current training state
        requirements = TaskRequirements(
            min_skill_level=0.3 + 0.4 * (episode / self.max_episodes),  # Increases with training
            required_capabilities={
                "learning_rate": 0.5,
                "adaptability": 0.3,
                "skill_mastery": 0.2 + 0.4 * (episode / self.max_episodes)
            },
            preferred_role=RoleType.LEARNER,
            min_agents=1,
            max_agents=1,
            estimated_duration=self.max_steps_per_episode,
            resource_requirements={}
        )
        
        # Create task with appropriate priority
        task = await self.task_allocator.create_task(
            task_id=task_id,
            priority=TaskPriority.HIGH,
            requirements=requirements
        )
        
        return task_id

    async def _update_task_progress(
        self,
        task_id: str,
        step: int,
        reward: float,
        info: Dict[str, Any]
    ):
        """Update task progress during training."""
        if not self.use_task_allocation or not task_id:
            return
            
        # Calculate progress as percentage of steps completed
        progress = step / self.max_steps_per_episode
        
        # Collect task metrics
        metrics = {
            "step_reward": reward,
            "cumulative_reward": sum(self.episode_rewards) / max(1, len(self.episode_rewards)),
            "exploration_rate": self.epsilon
        }
        metrics.update(info.get("task_metrics", {}))
        
        await self.task_allocator.update_task_progress(
            task_id=task_id,
            progress=progress,
            metrics=metrics
        )

    async def train(
        self,
        num_episodes: int,
        task_config: Dict[str, Any],
        agent_configs: Dict[str, Dict[str, Any]],
        agent_roles: Dict[str, str]
    ) -> Dict[str, Any]:
        """Train agents using curriculum learning and coalition formation."""
        try:
            for episode in range(num_episodes):
                self.current_episode = episode
                
                # Get task parameters from curriculum if enabled
                if self.use_curriculum:
                    task_params = self.curriculum_manager.get_current_parameters()
                    task_config.update(task_params)
                
                # Form coalitions if enabled
                coalition_id = None
                if self.use_coalition_formation:
                    coalition_id = await self._form_training_coalition(
                        task_config,
                        agent_configs,
                        agent_roles
                    )
                    if not coalition_id:
                        logger.warning(f"Failed to form coalition for episode {episode}")
                        continue
                    
                    # Get coalition members
                    coalition_members = self.coalition_manager.get_coalition_members(coalition_id)
                    if not coalition_members:
                        logger.warning(f"No members found for coalition {coalition_id}")
                        continue
                    
                    # Update agent configs to only include coalition members
                    episode_agent_configs = {
                        agent_id: config
                        for agent_id, config in agent_configs.items()
                        if agent_id in coalition_members
                    }
                    
                    # Generate dynamic task if enabled
                    if self.use_dynamic_tasks:
                        coalition = self.coalition_manager.coalitions[coalition_id]
                        performance_history = coalition.performance_history
                        
                        generated_task = self.task_generator.generate_task(
                            coalition=coalition,
                            performance_history=performance_history
                        )
                        
                        # Update task configuration
                        task_config.update(generated_task["parameters"])
                        task_config["id"] = generated_task["id"]
                        task_config["difficulty"] = generated_task["difficulty"]
                else:
                    episode_agent_configs = agent_configs
                    
                    # Generate dynamic task without coalition if enabled
                    if self.use_dynamic_tasks:
                        generated_task = self.task_generator.generate_task()
                        task_config.update(generated_task["parameters"])
                        task_config["id"] = generated_task["id"]
                        task_config["difficulty"] = generated_task["difficulty"]
                
                # Run training episode
                episode_results = await self._run_training_episode(
                    task_config,
                    episode_agent_configs
                )
                
                # Update task metrics if using dynamic tasks
                if self.use_dynamic_tasks and "id" in task_config:
                    self.task_generator.update_task_metrics(
                        task_id=task_config["id"],
                        coalition_id=coalition_id if coalition_id else "default",
                        metrics=episode_results
                    )
                
                # Update coalition performance if enabled
                if self.use_coalition_formation and coalition_id:
                    await self._update_coalition_performance(
                        coalition_id,
                        episode_results
                    )
                    
                    # Dissolve coalition if performance is poor
                    if self._should_dissolve_coalition(episode_results):
                        await self.coalition_manager.dissolve_coalition(
                            coalition_id,
                            reason="poor_performance"
                        )
                
                # Update curriculum if enabled
                if self.use_curriculum:
                    self.curriculum_manager.update_progress(episode_results)
                
                # Log episode results
                self._log_episode_results(episode_results, coalition_id)
                
                # Save checkpoint if best performance
                if episode_results["mean_reward"] > self.best_performance:
                    self.best_performance = episode_results["mean_reward"]
                    await self._save_checkpoint()
                    
            return self._get_training_summary()
            
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    async def _form_training_coalition(
        self,
        task_config: Dict[str, Any],
        agent_configs: Dict[str, Dict[str, Any]],
        agent_roles: Dict[str, str]
    ) -> Optional[str]:
        """Form a coalition for training."""
        try:
            # Get available agents (those not in other coalitions)
            available_agents = {
                agent_id: config
                for agent_id, config in agent_configs.items()
                if not self.coalition_manager.get_agent_coalition(agent_id)
            }
            
            if not available_agents:
                logger.warning("No available agents for coalition formation")
                return None
            
            # Form coalition
            coalition_id = await self.coalition_manager.form_coalition(
                task=task_config,
                available_agents=available_agents,
                agent_roles=agent_roles
            )
            
            # Initialize coalition learning if enabled
            if coalition_id and self.use_coalition_learning:
                coalition = self.coalition_manager.coalitions[coalition_id]
                learning_agents = {
                    agent_id: self.agent
                    for agent_id in available_agents.keys()
                }
                await self.coalition_learning_manager.initialize_coalition_learning(
                    coalition=coalition,
                    learning_agents=learning_agents
                )
            
            if coalition_id and self.communication_manager:
                # Notify other coalitions about the new formation
                existing_coalitions = self.coalition_manager.get_active_coalitions()
                for other_coalition in existing_coalitions:
                    if other_coalition.id != coalition_id:
                        await self.communication_manager.send_message(
                            sender_coalition=self.coalition_manager.coalitions[coalition_id],
                            receiver_coalition=other_coalition,
                            msg_type=MessageType.STATUS_UPDATE,
                            content={
                                "event": "coalition_formed",
                                "coalition_info": {
                                    "id": coalition_id,
                                    "size": len(available_agents),
                                    "task_type": task_config.get("type"),
                                    "resource_requirements": task_config.get("resource_requirements", {})
                                }
                            },
                            priority=MessagePriority.MEDIUM
                        )
            
            return coalition_id
            
        except Exception as e:
            logger.error(f"Error forming training coalition: {str(e)}")
            return None
    
    async def _update_coalition_performance(
        self,
        coalition_id: str,
        episode_results: Dict[str, Any]
    ) -> None:
        """Update coalition performance metrics."""
        try:
            # Extract relevant metrics
            task_metrics = {
                "success_rate": episode_results.get("success_rate", 0.0),
                "completion_time": episode_results.get("completion_time", 0.0),
                "goal_distance": episode_results.get("goal_distance", 0.0)
            }
            
            communication_stats = {
                "messages_sent": episode_results.get("messages_sent", 0),
                "coordination_score": episode_results.get("coordination_score", 0.0),
                "efficiency": episode_results.get("communication_efficiency", 1.0)
            }
            
            agent_rewards = episode_results.get("agent_rewards", {})
            
            # Update coalition performance
            await self.coalition_manager.update_coalition_performance(
                coalition_id=coalition_id,
                task_metrics=task_metrics,
                communication_stats=communication_stats,
                agent_rewards=agent_rewards
            )
            
            # Update coalition learning if enabled
            if self.use_coalition_learning:
                # Get current policy state
                policy_state = self.agent.network.state_dict()
                
                # Update policy and get learning status
                await self.coalition_learning_manager.update_coalition_policy(
                    coalition_id=coalition_id,
                    policy_state=policy_state,
                    performance=episode_results.get("mean_reward", 0.0),
                    metadata={
                        "task_metrics": task_metrics,
                        "communication_stats": communication_stats
                    }
                )
                
                # Optimize learning strategy
                await self.coalition_learning_manager.optimize_learning_strategy(
                    coalition_id=coalition_id,
                    performance_metrics={
                        **task_metrics,
                        **communication_stats,
                        "mean_reward": episode_results.get("mean_reward", 0.0),
                        "task_complexity": episode_results.get("task_complexity", 0.5),
                        "coordination_requirement": episode_results.get("coordination_requirement", 0.5),
                        "knowledge_share_benefit": episode_results.get("knowledge_share_benefit", 0.5)
                    }
                )
                
                # Consider knowledge transfer to other coalitions
                if episode_results.get("success_rate", 0.0) > 0.8:
                    other_coalitions = [
                        c.id for c in self.coalition_manager.get_active_coalitions()
                        if c.id != coalition_id
                    ]
                    for other_coalition in other_coalitions:
                        await self.coalition_learning_manager.transfer_knowledge(
                            source_coalition=coalition_id,
                            target_coalition=other_coalition,
                            task_type=str(self.task_type)
                        )
            
            if self.communication_manager:
                # Share performance report with other coalitions
                other_coalitions = [
                    c for c in self.coalition_manager.get_active_coalitions()
                    if c.id != coalition_id
                ]
                
                for other_coalition in other_coalitions:
                    await self.communication_manager.send_message(
                        sender_coalition=self.coalition_manager.coalitions[coalition_id],
                        receiver_coalition=other_coalition,
                        msg_type=MessageType.PERFORMANCE_REPORT,
                        content={
                            "coalition_id": coalition_id,
                            "task_metrics": task_metrics,
                            "communication_stats": communication_stats,
                            "avg_reward": np.mean(list(agent_rewards.values())),
                            "episode": self.current_episode
                        },
                        priority=MessagePriority.LOW
                    )
                
                # Request resources if performance is poor
                if episode_results.get("success_rate", 0.0) < 0.3:
                    for other_coalition in other_coalitions:
                        await self.communication_manager.send_message(
                            sender_coalition=self.coalition_manager.coalitions[coalition_id],
                            receiver_coalition=other_coalition,
                            msg_type=MessageType.RESOURCE_REQUEST,
                            content={
                                "requested_resources": {
                                    "computation": 100,
                                    "memory": 50,
                                    "bandwidth": 30
                                },
                                "reason": "poor_performance",
                                "duration": "temporary"
                            },
                            priority=MessagePriority.HIGH
                        )
            
            # Update system monitor with coalition metrics
            await self.system_monitor.update_coalition_metrics(
                coalition_id=coalition_id,
                metrics={
                    "success_rate": task_metrics["success_rate"],
                    "completion_time": task_metrics["completion_time"],
                    "goal_distance": task_metrics["goal_distance"],
                    "messages_sent": communication_stats["messages_sent"],
                    "coordination_score": communication_stats["coordination_score"],
                    "efficiency": communication_stats["efficiency"],
                    "mean_reward": episode_results.get("mean_reward", 0.0)
                }
            )
            
            # Update learning metrics if enabled
            if self.use_coalition_learning:
                learning_status = self.coalition_learning_manager.get_coalition_learning_status(coalition_id)
                if learning_status:
                    await self.system_monitor.update_learning_metrics(
                        coalition_id=coalition_id,
                        metrics={
                            "episodes_trained": learning_status["episodes_trained"],
                            "current_performance": learning_status["current_performance"],
                            "best_performance": learning_status["best_performance"],
                            "learning_rate": self.agent.optimizer.param_groups[0]["lr"],
                            "policy_updates": len(self.coalition_learning_manager.coalition_policies[coalition_id].policies)
                        }
                    )
            
        except Exception as e:
            logger.error(f"Error updating coalition performance: {str(e)}")
    
    def _should_dissolve_coalition(
        self,
        episode_results: Dict[str, Any]
    ) -> bool:
        """Determine if a coalition should be dissolved based on performance."""
        # Check if success rate is below threshold
        if episode_results.get("success_rate", 0.0) < 0.2:
            return True
            
        # Check if mean reward is significantly negative
        if episode_results.get("mean_reward", 0.0) < -10.0:
            return True
            
        # Check if communication efficiency is poor
        if episode_results.get("communication_efficiency", 1.0) < 0.3:
            return True
            
        return False
    
    def _log_episode_results(
        self,
        episode_results: Dict[str, Any],
        coalition_id: Optional[str] = None
    ) -> None:
        """Log episode results and update metrics."""
        # Update metrics
        self.metrics.update({
            "episode": self.current_episode,
            "mean_reward": episode_results["mean_reward"],
            "success_rate": episode_results.get("success_rate", 0.0),
            "completion_time": episode_results.get("completion_time", 0.0)
        })
        
        # Add coalition metrics if applicable
        if coalition_id:
            coalition_metrics = self.coalition_manager.get_coalition_metrics()
            if coalition_id in coalition_metrics:
                self.metrics.update({
                    "coalition_reward": coalition_metrics[coalition_id]["avg_reward"],
                    "coalition_size": coalition_metrics[coalition_id]["num_members"],
                    "coalition_performance": coalition_metrics[coalition_id]["avg_performance"]
                })
        
        # Log metrics
        logger.info(
            f"Episode {self.current_episode} - "
            f"Mean Reward: {episode_results['mean_reward']:.2f}, "
            f"Success Rate: {episode_results.get('success_rate', 0.0):.2f}"
        )
        
        # Store in history
        self.training_history.append(episode_results)
    
    async def _save_checkpoint(self) -> None:
        """Save training checkpoint."""
        try:
            checkpoint = {
                "episode": self.current_episode,
                "best_performance": self.best_performance,
                "metrics": self.metrics.get_all(),
                "history": self.training_history
            }
            
            # Add curriculum state if enabled
            if self.use_curriculum:
                checkpoint["curriculum_state"] = self.curriculum_manager.get_state()
            
            # Add meta-learning state if enabled
            if self.use_meta_learning:
                checkpoint["meta_learning_state"] = self.meta_learning_manager.get_state()
            
            # Add coalition state if enabled
            if self.use_coalition_formation:
                checkpoint["coalition_state"] = {
                    "coalitions": self.coalition_manager.coalitions,
                    "agent_coalitions": self.coalition_manager.agent_coalitions,
                    "history": self.coalition_manager.coalition_history
                }
                
                # Add coalition learning state if enabled
                if self.use_coalition_learning:
                    learning_state = {}
                    for coalition_id in self.coalition_manager.coalitions:
                        learning_state[coalition_id] = self.coalition_learning_manager.get_coalition_learning_status(coalition_id)
                    checkpoint["coalition_learning_state"] = learning_state
                
                # Add communication state if enabled
                if self.communication_manager:
                    checkpoint["communication_state"] = {
                        "stats": self.communication_manager.communication_stats,
                        "message_history": self.communication_manager.message_history,
                        "bandwidth_usage": self.communication_manager.bandwidth_usage
                    }
            
            # Add system monitoring data
            system_status = self.system_monitor.get_system_status()
            checkpoint["system_status"] = system_status
            
            for coalition_id in self.coalition_manager.coalitions:
                coalition_status = self.system_monitor.get_coalition_status(coalition_id)
                if coalition_status:
                    checkpoint.setdefault("coalition_status", {})[coalition_id] = coalition_status
            
            # Save checkpoint (implementation depends on storage system)
            logger.info(f"Saved checkpoint at episode {self.current_episode}")
            
        except Exception as e:
            logger.error(f"Error saving checkpoint: {str(e)}")
    
    def _get_training_summary(self) -> Dict[str, Any]:
        """Get summary of training progress."""
        try:
            summary = {
                "episodes_completed": self.current_episode + 1,
                "best_performance": self.best_performance,
                "final_metrics": self.metrics.get_all(),
                "training_time": None  # TODO: Add training time tracking
            }
            
            # Add curriculum summary if enabled
            if self.use_curriculum:
                summary["curriculum_progress"] = self.curriculum_manager.get_progress()
            
            # Add meta-learning summary if enabled
            if self.use_meta_learning:
                summary["meta_learning_status"] = self.meta_learning_manager.get_optimization_status()
            
            # Add coalition summary if enabled
            if self.use_coalition_formation:
                summary["coalition_metrics"] = self.coalition_manager.get_coalition_metrics()
                
                # Add coalition learning summary if enabled
                if self.use_coalition_learning:
                    learning_summary = {}
                    for coalition_id in self.coalition_manager.coalitions:
                        status = self.coalition_learning_manager.get_coalition_learning_status(coalition_id)
                        if status:
                            learning_summary[coalition_id] = {
                                "strategy": self.coalition_learning_manager.config.strategy.name,
                                "episodes_trained": status.get("episodes_trained", 0),
                                "best_performance": status.get("best_performance", 0.0),
                                "recent_trend": status.get("recent_performance", {}).get("trend", 0.0),
                                "knowledge_base_size": sum(
                                    status.get("knowledge_base", {}).values()
                                )
                            }
                    summary["coalition_learning"] = learning_summary
                
                # Add communication summary if enabled
                if self.communication_manager:
                    comm_metrics = self.communication_manager.get_communication_metrics()
                    summary["communication_metrics"] = {
                        "total_messages": comm_metrics["total_messages"],
                        "total_bytes": comm_metrics["total_bytes_transferred"],
                        "avg_latency": comm_metrics["average_latency"],
                        "success_rate": np.mean([
                            stats["success_rate"]
                            for stats in comm_metrics["detailed_stats"].values()
                        ])
                    }
            
            # Add system monitoring summary
            system_status = self.system_monitor.get_system_status()
            summary["system_health"] = {
                "cpu_usage": system_status["health"]["cpu_usage"],
                "memory_usage": system_status["health"]["memory_usage"],
                "error_counts": system_status["health"]["error_counts"],
                "recent_alerts": [
                    alert["type"] for alert in system_status["alerts"]
                ]
            }
            
            if system_status.get("profiling"):
                summary["performance_profile"] = system_status["profiling"]
            
            # Add task generation summary if enabled
            if self.use_dynamic_tasks:
                task_history = self.task_generator.get_task_history()
                summary["task_generation"] = {
                    "total_tasks": len(task_history),
                    "difficulty_distribution": {
                        difficulty: len([
                            t for t in task_history
                            if t["difficulty"] == difficulty
                        ])
                        for difficulty in TaskDifficulty.__members__
                    },
                    "completion_rate": len([
                        t for t in task_history
                        if t.get("status") == "completed"
                    ]) / max(1, len(task_history))
                }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting training summary: {str(e)}")
            return {"status": "error", "message": str(e)}

    async def _select_action(self, state: np.ndarray) -> np.ndarray:
        """Select action using epsilon-greedy policy with validation."""
        if np.random.random() < self.epsilon:
            # Random action within bounds
            action = self.action_space.sample()
        else:
            # Use network to select action
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                raw_action = self.agent.network(state_tensor).cpu().numpy().squeeze()
                # Denormalize action if needed
                action = self.action_space.denormalize_action(raw_action)
        
        # Validate and adjust action if needed
        if not self.action_space.validate_action(action):
            self.invalid_actions_count += 1
            # Log invalid action
            self.writer.add_scalar(
                "Training/Invalid_Actions",
                self.invalid_actions_count,
                self.agent.training_steps
            )
            # Clip action to valid range
            if hasattr(self.action_space, "bounds"):
                action = np.clip(
                    action,
                    self.action_space.bounds.low,
                    self.action_space.bounds.high
                )
        
        return action

    def _update_metrics(
        self,
        episode: int,
        step: int,
        reward: float,
        reward_components: Dict[str, float],
        info: Dict[str, Any]
    ):
        """Update training metrics including graph RL and communication metrics."""
        # Episode-specific metrics
        self.writer.add_scalar("Training/Step_Reward", reward, self.agent.training_steps)
        self.writer.add_scalar("Training/Episode", episode, self.agent.training_steps)
        self.writer.add_scalar("Training/Epsilon", self.epsilon, self.agent.training_steps)
        
        # Add reward component breakdowns
        for component, value in reward_components.items():
            self.writer.add_scalar(f"Rewards/{component}", value, self.agent.training_steps)
            
        # Add environment info metrics
        for key, value in info.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(f"Environment/{key}", value, self.agent.training_steps)

        # Add graph RL and communication metrics if enabled
        if self.use_graph_rl:
            # Get graph metrics
            graph_metrics = self.graph_rl_manager.get_metrics()
            for key, value in graph_metrics.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(f"Graph/{key}", value, self.agent.training_steps)
                elif isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        self.writer.add_scalar(
                            f"Graph/{key}/{subkey}",
                            subvalue,
                            self.agent.training_steps
                        )

            # Get communication metrics
            comm_stats = self.graph_rl_manager.get_communication_stats()
            for key, value in comm_stats.items():
                self.writer.add_scalar(
                    f"Communication/{key}",
                    value,
                    self.agent.training_steps
                )

    def _log_curriculum_progress(self, progress: Dict[str, Any]):
        """Log curriculum learning progress."""
        # Log current difficulty level
        self.writer.add_text(
            "Curriculum/Current_Difficulty",
            progress["current_difficulty"],
            self.agent.training_steps
        )
        
        # Log success rate
        self.writer.add_scalar(
            "Curriculum/Success_Rate",
            progress["success_rate"],
            self.agent.training_steps
        )
        
        # Log episodes completed in current stage
        self.writer.add_scalar(
            "Curriculum/Stage_Episodes",
            progress["episodes_completed"],
            self.agent.training_steps
        )
        
        # Log completion criteria progress
        for criterion, threshold in progress["completion_criteria"].items():
            if progress["recent_metrics"]:
                current_value = np.mean([
                    m.get(criterion, 0) for m in progress["recent_metrics"]
                ])
                self.writer.add_scalar(
                    f"Curriculum/Criteria/{criterion}",
                    current_value,
                    self.agent.training_steps
                )
                self.writer.add_scalar(
                    f"Curriculum/Criteria/{criterion}_Target",
                    threshold,
                    self.agent.training_steps
                )
    
    async def evaluate(self, num_episodes: int = 10) -> Tuple[float, float]:
        """Evaluate agent performance."""
        logger.info(f"Starting evaluation for {num_episodes} episodes...")
        evaluation_rewards = []
        task_metrics_list = []
        
        for episode in range(num_episodes):
            task = create_task(self.task_type)
            state = task.reset()
            episode_reward = 0
            
            # Reset reward manager for new episode
            self.reward_manager.reset()
            
            for step in range(self.max_steps_per_episode):
                # Select best action (no exploration)
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                    action = self.agent.network(state_tensor)
                    action = action.cpu().numpy().squeeze()
                    
                # Execute action
                next_state, _, done, info = task.step(action)
                
                # Calculate task-specific reward
                reward, _ = self.reward_manager.calculate_reward(
                    state=state,
                    action=action,
                    next_state=next_state,
                    info=info
                )
                episode_reward += reward
                state = next_state
                
                if done:
                    break
                    
            evaluation_rewards.append(episode_reward)
            task_metrics_list.append(self.reward_manager.get_metrics())
            
            # Log evaluation metrics
            metrics_str = ", ".join([
                f"{k}: {v:.2f}" for k, v in task_metrics_list[-1].items()
                if isinstance(v, (int, float))
            ])
            logger.info(
                f"Evaluation Episode {episode + 1} - "
                f"Reward: {episode_reward:.2f}\n"
                f"Task Metrics: {metrics_str}"
            )
            
        mean_reward = np.mean(evaluation_rewards)
        std_reward = np.std(evaluation_rewards)
        
        # Calculate average task metrics
        avg_task_metrics = {}
        for metric in task_metrics_list[0].keys():
            if isinstance(task_metrics_list[0][metric], (int, float)):
                values = [m[metric] for m in task_metrics_list]
                avg_task_metrics[metric] = np.mean(values)
                
        logger.info(
            f"Evaluation Results - "
            f"Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}\n"
            f"Average Task Metrics: {avg_task_metrics}"
        )
        
        return mean_reward, std_reward

    async def _update_agent_capabilities(self, agent_id: int, state: np.ndarray, action: np.ndarray, reward: float, info: Dict[str, Any]):
        """Update agent capabilities based on performance."""
        if not self.use_role_management:
            return

        # Extract capabilities from state and performance
        capabilities = {
            "mobility": float(np.mean(np.abs(action[:3]))),  # Movement capability
            "perception": float(np.mean(state[:3])),         # Perception of environment
            "skill_mastery": float(reward) if reward > 0 else 0.0,
            "communication": float(info.get("communication_success", 0.0)),
            "cooperation": float(info.get("cooperation_score", 0.0)),
            "leadership": float(info.get("influence_score", 0.0)),
            "adaptability": float(info.get("adaptation_rate", 0.0)),
            "resource_management": float(info.get("resource_efficiency", 0.0)),
            "learning_rate": float(info.get("learning_progress", 0.0))
        }
        
        # Update agent capabilities
        if agent_id not in self.agent_capabilities:
            self.agent_capabilities[agent_id] = capabilities
        else:
            # Exponential moving average
            alpha = 0.1
            for key, value in capabilities.items():
                self.agent_capabilities[agent_id][key] = (
                    alpha * value +
                    (1 - alpha) * self.agent_capabilities[agent_id].get(key, value)
                )

    async def _update_role_assignments(self, agent_id: int, metrics: Dict[str, float]):
        """Update role assignments based on performance."""
        if not self.use_role_management:
            return

        # Update role performance
        new_role = await self.role_manager.update_role_performance(agent_id, metrics)
        
        if new_role is not None:
            # Role transition occurred
            self.logger.info(f"Agent {agent_id} transitioned to role {new_role.name}")
            
            # Update agent's policy based on new role
            await self._adapt_policy_to_role(agent_id, new_role)
            
            # Log transition
            self.writer.add_text(
                f"Role_Transition/Agent_{agent_id}",
                f"Transitioned to {new_role.name}",
                self.agent.training_steps
            )

    async def _adapt_policy_to_role(self, agent_id: int, role: RoleType):
        """Adapt agent's policy based on its new role."""
        role_info = self.role_manager.get_agent_role_info(agent_id)
        if role_info is None:
            return
            
        # Adjust exploration parameters based on role
        if role == RoleType.EXPLORER:
            self.epsilon = max(self.epsilon, 0.3)  # Maintain higher exploration
        elif role == RoleType.SPECIALIST:
            self.epsilon = min(self.epsilon, 0.1)  # Focus on exploitation
            
        # Adjust network parameters based on role priorities
        priorities = role_info["priorities"]
        self.agent.network.adapt_to_role(priorities)
        
        # Update communication parameters
        if self.use_graph_rl:
            comm_priority = role_info["communication_priority"]
            self.graph_rl_manager.comm_config.message_size = int(
                64 * (1 + comm_priority)
            )

    async def _update_graph_state(self, state: np.ndarray, action: np.ndarray, reward: float):
        """Update graph state based on current interaction with message types and roles."""
        if not self.use_graph_rl:
            return

        try:
            # Update agent node features based on state and action
            agent_features = torch.cat([
                torch.from_numpy(state).float(),
                torch.from_numpy(action).float(),
                torch.tensor([reward]).float()
            ])
            agent_features = F.pad(
                agent_features,
                (0, self.graph_rl_manager.config.node_feature_dim - len(agent_features))
            )
            self.graph_rl_manager.node_features[0] = agent_features

            # Get role information if available
            role_info = (
                self.role_manager.get_agent_role_info(0)
                if self.use_role_management
                else None
            )
            
            # Process any pending messages
            await self.graph_rl_manager.process_messages()

            # Get neighboring nodes for communication
            neighbors = self.graph_rl_manager.get_node_neighbors(0)
            if neighbors:
                # Create state update message with role context
                state_embedding = self.agent.network.encode_state(
                    torch.from_numpy(state).float().unsqueeze(0)
                ).squeeze()
                
                metadata = {
                    "step": self.agent.training_steps,
                    "role": role_info["role"] if role_info else None,
                    "confidence": role_info["confidence"] if role_info else 1.0
                }
                
                # Adjust message priority based on role
                base_priority = 0.5
                if role_info:
                    base_priority *= role_info["communication_priority"]
                
                await self.graph_rl_manager.send_message(
                    sender_id=0,
                    receiver_ids=list(neighbors),
                    content=state_embedding,
                    msg_type=MessageType.STATE_UPDATE,
                    priority=base_priority,
                    metadata=metadata
                )

                # Send action proposal if reward is positive
                if reward > 0:
                    action_embedding = torch.from_numpy(action).float()
                    await self.graph_rl_manager.send_message(
                        sender_id=0,
                        receiver_ids=list(neighbors),
                        content=action_embedding,
                        msg_type=MessageType.ACTION_PROPOSAL,
                        priority=min(1.0, reward),
                        metadata={"reward": float(reward)}
                    )

                # Send reward signal
                reward_tensor = torch.tensor([reward]).float()
                await self.graph_rl_manager.send_message(
                    sender_id=0,
                    receiver_ids=list(neighbors),
                    content=reward_tensor,
                    msg_type=MessageType.REWARD_SIGNAL,
                    priority=min(1.0, abs(reward)),
                    metadata={"step": self.agent.training_steps}
                )

                # Send coordination request if needed
                if self.agent.training_steps % 100 == 0:
                    coord_embedding = torch.cat([
                        state_embedding,
                        action_embedding,
                        reward_tensor
                    ])
                    await self.graph_rl_manager.send_message(
                        sender_id=0,
                        receiver_ids=list(neighbors),
                        content=coord_embedding,
                        msg_type=MessageType.COORDINATION_REQUEST,
                        priority=0.7,
                        metadata={
                            "step": self.agent.training_steps,
                            "purpose": "periodic_sync"
                        }
                    )

            # Update embeddings for the local subgraph
            subgraph_nodes = self.graph_rl_manager.get_subgraph(center_node=0, depth=2)
            await self.graph_rl_manager.update_node_embeddings(subgraph_nodes)

            # Optimize communication if needed
            if self.agent.training_steps % 100 == 0:
                comm_stats = self.graph_rl_manager.get_communication_stats()
                self.graph_rl_manager.optimize_communication({
                    "bandwidth_utilization": comm_stats["bandwidth_used"] / self.graph_rl_manager.comm_config.bandwidth_limit,
                    "average_latency": comm_stats["avg_latency"],
                    "message_success_rate": 1.0 - comm_stats.get("retry_rate", 0.0)
                })

        except Exception as e:
            self.logger.error(f"Failed to update graph state: {str(e)}")
            raise 