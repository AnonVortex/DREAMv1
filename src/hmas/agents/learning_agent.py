"""Learning Agent implementation for H-MAS.

This module implements a learning agent capable of multiple types of learning:
- Reinforcement learning
- Supervised learning
- Unsupervised learning
- Meta-learning

The agent can adapt its learning strategies based on performance and experience.
"""

from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
import torch
import numpy as np
import logging
from pathlib import Path
import asyncio
from datetime import datetime
import json

from hmas.core.agent import Agent
from hmas.core.learning import (
    LearningSystem,
    LearningConfig,
    LearningMode
)
from hmas.core.memory import MemorySystem
from hmas.core.reasoning import ReasoningEngine
from hmas.core.consciousness import ConsciousnessCore
from hmas.core.perception import PerceptionCore

@dataclass
class AgentConfig:
    """Configuration for learning agent."""
    name: str
    learning_config: LearningConfig
    observation_space: Dict[str, Any]
    action_space: Dict[str, Any]
    max_steps_per_episode: int = 1000
    evaluation_frequency: int = 100
    checkpoint_frequency: int = 1000
    save_dir: str = "agent_data"

class LearningAgent(Agent):
    """An agent capable of multiple types of learning and adaptation."""

    def __init__(
        self,
        config: AgentConfig,
        memory: MemorySystem,
        reasoning: ReasoningEngine,
        consciousness: ConsciousnessCore,
        perception: PerceptionCore
    ):
        """Initialize the learning agent.
        
        Args:
            config: Configuration for the agent
            memory: Memory system for the agent
            reasoning: Reasoning engine for the agent
            consciousness: Consciousness core for the agent
            perception: Perception core for the agent
        """
        super().__init__(name=config.name)
        
        self.config = config
        self.logger = logging.getLogger(f"agent_{config.name}")
        
        # Create save directory
        Path(config.save_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize core systems
        self.memory = memory
        self.reasoning = reasoning
        self.consciousness = consciousness
        self.perception = perception
        
        # Initialize learning system
        self.learning_system = LearningSystem(
            config=config.learning_config,
            memory=memory,
            reasoning=reasoning,
            consciousness=consciousness
        )
        
        # Initialize agent state
        self.current_task: Optional[Dict[str, Any]] = None
        self.episode_count: int = 0
        self.total_steps: int = 0
        self.performance_history: List[Dict[str, Any]] = []
        
        # Add learning capability
        self.capabilities.add("learning")
        
        # Initialize state
        self.state = {
            "models": {ltype: {} for ltype in config.learning_config.learning_types},
            "performance_history": {ltype: [] for ltype in config.learning_config.learning_types},
            "learning_progress": {
                "episodes": 0,
                "improvements": {ltype: 0.0 for ltype in config.learning_config.learning_types},
                "adaptation_rate": {ltype: 0.0 for ltype in config.learning_config.learning_types}
            },
            "meta_learning": {
                "strategies": {ltype: "default" for ltype in config.learning_config.learning_types},
                "effectiveness": {ltype: 0.0 for ltype in config.learning_config.learning_types}
            }
        }

    async def initialize(self) -> bool:
        """Initialize the agent's learning components."""
        try:
            # Initialize perception system
            await self.perception.initialize()
            
            # Initialize memory systems
            await self.memory.initialize()
            
            # Initialize reasoning engine
            await self.reasoning.initialize()
            
            # Initialize consciousness core
            await self.consciousness.initialize()
            
            # Initialize learning models for each type
            for learning_type in self.config.learning_config.learning_types:
                self.state["models"][learning_type] = self._create_model(learning_type)
                
                # Initialize meta-learning components
                self.state["meta_learning"]["strategies"][learning_type] = "default"
                self.state["meta_learning"]["effectiveness"][learning_type] = 0.0
                
                # Initialize progress tracking
                self.state["learning_progress"]["improvements"][learning_type] = 0.0
                self.state["learning_progress"]["adaptation_rate"][learning_type] = 0.0
            
            self.logger.info(f"Agent {self.config.name} initialized successfully")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize learning agent: {str(e)}")
            return False

    def _create_model(self, learning_type: str) -> Dict:
        """Create a new learning model of the specified type."""
        if learning_type == "reinforcement":
            return {"q_table": {}, "policy": "epsilon_greedy"}
        elif learning_type == "supervised":
            return {"weights": np.random.randn(10), "bias": 0.0}
        elif learning_type == "unsupervised":
            return {"clusters": [], "embeddings": []}
        elif learning_type == "meta":
            return {"strategies": {}, "adaptations": []}
        else:
            raise ValueError(f"Unsupported learning type: {learning_type}")

    async def process(self, input_data: Dict) -> Dict:
        """Process input data based on learning type."""
        learning_type = input_data["type"]
        data = input_data["data"]
        context = input_data.get("context", {})
        
        if learning_type == "reinforcement_learning":
            return await self._process_reinforcement(data, context)
        elif learning_type == "supervised_learning":
            return await self._process_supervised(data, context)
        elif learning_type == "unsupervised_learning":
            return await self._process_unsupervised(data, context)
        elif learning_type == "meta_learning":
            return await self._process_meta(data, context)
        else:
            raise ValueError(f"Unsupported learning type: {learning_type}")

    async def _process_reinforcement(self, data: Dict, context: Dict) -> Dict:
        """Process reinforcement learning input."""
        state = data["state"]
        action = data["action"]
        reward = data["reward"]
        next_state = data["next_state"]
        
        # Update Q-value
        model = self.state["models"]["reinforcement"]
        state_key = str(state.tobytes())
        next_state_key = str(next_state.tobytes())
        
        if state_key not in model["q_table"]:
            model["q_table"][state_key] = np.zeros(10)  # Assuming 10 possible actions
        
        # Q-learning update
        current_q = model["q_table"][state_key][action]
        next_q = max(model["q_table"].get(next_state_key, np.zeros(10)))
        new_q = current_q + self.config.learning_config.learning_rate * (reward + 0.9 * next_q - current_q)
        model["q_table"][state_key][action] = new_q
        
        # Select next action
        if np.random.random() < self.config.learning_config.exploration_rate:
            next_action = np.random.randint(10)
        else:
            next_action = np.argmax(model["q_table"].get(next_state_key, np.zeros(10)))
        
        return {
            "next_action": next_action,
            "model_update": {"q_value": new_q},
            "exploration_rate": self.config.learning_config.exploration_rate
        }

    async def _process_supervised(self, data: Dict, context: Dict) -> Dict:
        """Process supervised learning input."""
        inputs = data["inputs"]
        targets = data["targets"]
        
        model = self.state["models"]["supervised"]
        
        # Simple linear model update
        predictions = np.dot(inputs, model["weights"]) + model["bias"]
        loss = np.mean((predictions - targets) ** 2)
        
        # Gradient descent update
        gradients = np.dot(inputs.T, (predictions - targets)) / len(targets)
        model["weights"] -= self.config.learning_config.learning_rate * gradients
        model["bias"] -= self.config.learning_config.learning_rate * np.mean(predictions - targets)
        
        performance = 1.0 - loss  # Simple performance metric
        
        return {
            "loss": loss,
            "performance": performance,
            "model_update": {
                "weights": model["weights"].tolist(),
                "bias": float(model["bias"])
            }
        }

    async def _process_unsupervised(self, data: Dict, context: Dict) -> Dict:
        """Process unsupervised learning input."""
        inputs = data["inputs"]
        model = self.state["models"]["unsupervised"]
        
        # Simple clustering
        if not model["clusters"]:
            model["clusters"] = [inputs[0]]  # Initialize first cluster
            
        patterns = []
        for point in inputs:
            min_dist = float('inf')
            closest_cluster = None
            
            for cluster in model["clusters"]:
                dist = np.linalg.norm(point - cluster)
                if dist < min_dist:
                    min_dist = dist
                    closest_cluster = cluster
            
            if min_dist > 2.0:  # Distance threshold
                model["clusters"].append(point)
            patterns.append(closest_cluster)
        
        quality = len(model["clusters"]) / len(inputs)  # Simple quality metric
        
        return {
            "patterns": [p.tolist() for p in patterns],
            "quality_metrics": {"cluster_quality": quality},
            "model_update": {"num_clusters": len(model["clusters"])}
        }

    async def _process_meta(self, data: Dict, context: Dict) -> Dict:
        """Process meta-learning input."""
        learning_type = data["learning_type"]
        performance_data = data["performance_data"]
        
        # Analyze performance trend
        analysis = self._analyze_learning_performance(learning_type, performance_data)
        
        # Adapt strategy based on analysis
        new_strategy = self._adapt_learning_strategy(learning_type, analysis)
        
        # Apply adaptation
        self.state["meta_learning"]["strategies"][learning_type] = new_strategy["name"]
        self.config.learning_config.learning_rate = new_strategy["learning_rate"]
        self.config.learning_config.exploration_rate = new_strategy["exploration_rate"]
        
        return {
            "analysis": analysis,
            "new_strategy": new_strategy,
            "adaptation": {
                "learning_rate": self.config.learning_config.learning_rate,
                "exploration_rate": self.config.learning_config.exploration_rate
            }
        }

    def _analyze_learning_performance(self, learning_type: str, performance_data: List[Dict]) -> Dict:
        """Analyze learning performance trends."""
        performances = [d["performance"] for d in performance_data]
        
        if len(performances) < 2:
            return {
                "trend": "insufficient_data",
                "improvement_rate": 0.0,
                "stability": 1.0
            }
        
        # Calculate metrics
        trend = "improving" if performances[-1] > performances[0] else "declining"
        improvement_rate = (performances[-1] - performances[0]) / len(performances)
        stability = 1.0 - np.std(performances)
        
        return {
            "trend": trend,
            "improvement_rate": float(improvement_rate),
            "stability": float(stability)
        }

    def _adapt_learning_strategy(self, learning_type: str, analysis: Dict) -> Dict:
        """Adapt learning strategy based on performance analysis."""
        if analysis["trend"] == "improving":
            if analysis["stability"] > 0.8:
                # Stable improvement - reduce exploration
                return {
                    "name": "adaptive",
                    "learning_rate": max(0.001, self.config.learning_config.learning_rate * 0.95),
                    "exploration_rate": max(0.01, self.config.learning_config.exploration_rate * 0.9)
                }
            else:
                # Unstable improvement - maintain exploration
                return {
                    "name": "default",
                    "learning_rate": self.config.learning_config.learning_rate,
                    "exploration_rate": self.config.learning_config.exploration_rate
                }
        else:
            # Declining performance - increase exploration
            return {
                "name": "exploratory",
                "learning_rate": min(0.1, self.config.learning_config.learning_rate * 1.1),
                "exploration_rate": min(0.5, self.config.learning_config.exploration_rate * 1.2)
            }

    async def communicate(self, message: Dict, target_id: UUID) -> bool:
        """Send a message to another agent."""
        try:
            # In a real implementation, this would use the communication system
            # For testing, we just return success
            return True
        except Exception as e:
            self.logger.error(f"Communication failed: {str(e)}")
            return False

    async def learn(self, experience: Dict) -> bool:
        """Learn from experience."""
        try:
            feedback = experience["feedback"]
            learning_type = feedback["type"]
            performance = feedback["performance"]
            
            # Record performance
            self.state["performance_history"][learning_type].append({
                "performance": performance,
                "timestamp": datetime.now()
            })
            
            # Update learning progress
            self.state["learning_progress"]["episodes"] += 1
            
            # Trim history if needed
            if len(self.state["performance_history"][learning_type]) > self.config.learning_config.memory_size:
                self.state["performance_history"][learning_type].pop(0)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Learning failed: {str(e)}")
            return False

    async def reflect(self) -> Dict:
        """Perform self-reflection and analysis."""
        reflection = {
            "status": "active",
            "learning_types": self.config.learning_config.learning_types,
            "performance_summary": {},
            "meta_learning": {
                "strategies": self.state["meta_learning"]["strategies"],
                "effectiveness": self.state["meta_learning"]["effectiveness"]
            }
        }
        
        # Analyze performance for each learning type
        for ltype in self.config.learning_config.learning_types:
            history = self.state["performance_history"][ltype]
            if history:
                analysis = self._analyze_learning_performance(ltype, history)
                reflection["performance_summary"][ltype] = analysis
        
        return reflection

    async def perceive(
        self,
        observation: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process observation through perception system."""
        # Process raw observation
        processed_obs = await self.perception.process_input(observation)
        
        # Store observation in working memory
        await self.memory.update_working_memory({
            "observation": processed_obs,
            "timestamp": datetime.now().isoformat()
        })
        
        # Update consciousness with new observation
        await self.consciousness.update_state({
            "perception": processed_obs
        })
        
        return processed_obs
        
    async def reason(
        self,
        observation: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply reasoning to current situation."""
        # Retrieve relevant memories
        memories = await self.memory.retrieve_relevant(
            observation,
            limit=5
        )
        
        # Generate reasoning context
        context = {
            "observation": observation,
            "memories": memories,
            "consciousness_state": await self.consciousness.get_state()
        }
        
        # Apply reasoning
        reasoning_results = await self.reasoning.analyze(context)
        
        # Update consciousness with reasoning results
        await self.consciousness.update_state({
            "reasoning": reasoning_results
        })
        
        return reasoning_results
        
    async def decide_action(
        self,
        observation: Dict[str, Any],
        reasoning_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Decide next action based on observation and reasoning."""
        # Combine information for decision making
        decision_context = {
            "observation": observation,
            "reasoning": reasoning_results,
            "consciousness": await self.consciousness.get_state(),
            "working_memory": await self.memory.get_working_memory()
        }
        
        # Use reasoning engine to select action
        action_selection = await self.reasoning.select_action(
            decision_context
        )
        
        # Validate action against action space
        action = self._validate_action(action_selection["action"])
        
        # Update consciousness with decision
        await self.consciousness.update_state({
            "decision": action_selection
        })
        
        return action
        
    async def learn_from_experience(
        self,
        experience: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Learn from interaction experience."""
        # Determine learning mode based on experience
        mode = self._determine_learning_mode(experience)
        
        # Prepare learning task
        task_data = await self._prepare_learning_task(
            experience,
            mode
        )
        
        # Execute learning
        results = await self.learning_system.learn(
            task_data,
            mode=mode
        )
        
        # Update consciousness with learning results
        await self.consciousness.update_state({
            "learning": results
        })
        
        return results
        
    async def adapt(
        self,
        new_task: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Adapt to new task."""
        # Store current task
        if self.current_task:
            await self.memory.store_episodic(
                self.current_task,
                context={"type": "task_transition"}
            )
        
        # Update current task
        self.current_task = new_task
        
        # Adapt learning system
        adaptation_results = await self.learning_system.adapt_to_task(
            new_task
        )
        
        # Update consciousness with adaptation
        await self.consciousness.update_state({
            "adaptation": adaptation_results
        })
        
        return adaptation_results
        
    async def evaluate(self) -> Dict[str, Any]:
        """Evaluate agent's current performance."""
        if not self.current_task:
            return {"success": False, "error": "No current task"}
            
        # Perform evaluation
        eval_results = await self.learning_system.evaluate_performance(
            self.current_task
        )
        
        # Store evaluation results
        self.performance_history.append({
            "timestamp": datetime.now().isoformat(),
            "task": self.current_task,
            "results": eval_results
        })
        
        # Update consciousness with evaluation
        await self.consciousness.update_state({
            "evaluation": eval_results
        })
        
        return eval_results
        
    async def run_episode(
        self,
        environment: Any,
        task: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Run a single episode in the environment."""
        if task and task != self.current_task:
            await self.adapt(task)
            
        episode_memory = []
        total_reward = 0
        done = False
        steps = 0
        
        # Get initial observation
        observation = await self._get_observation(environment)
        
        while not done and steps < self.config.max_steps_per_episode:
            # Process observation
            processed_obs = await self.perceive(observation)
            
            # Apply reasoning
            reasoning_results = await self.reason(processed_obs)
            
            # Select action
            action = await self.decide_action(
                processed_obs,
                reasoning_results
            )
            
            # Execute action
            next_observation, reward, done, info = await self._execute_action(
                environment,
                action
            )
            
            # Store experience
            experience = {
                "observation": observation,
                "action": action,
                "reward": reward,
                "next_observation": next_observation,
                "done": done,
                "info": info
            }
            episode_memory.append(experience)
            
            # Learn from experience
            if len(episode_memory) >= self.config.learning_config.batch_size:
                await self.learn_from_experience({
                    "experiences": episode_memory,
                    "task": self.current_task
                })
                episode_memory = []
                
            # Update state
            observation = next_observation
            total_reward += reward
            steps += 1
            self.total_steps += 1
            
            # Periodic evaluation
            if self.total_steps % self.config.learning_config.evaluation_frequency == 0:
                await self.evaluate()
                
            # Periodic checkpointing
            if self.total_steps % self.config.learning_config.checkpoint_frequency == 0:
                await self.save_checkpoint()
                
        self.episode_count += 1
        
        # Final learning update with remaining experiences
        if episode_memory:
            await self.learn_from_experience({
                "experiences": episode_memory,
                "task": self.current_task
            })
            
        return {
            "total_reward": total_reward,
            "steps": steps,
            "episode": self.episode_count
        }
        
    def _validate_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Validate action against action space."""
        validated_action = {}
        
        for key, value in action.items():
            if key in self.config.action_space:
                space = self.config.action_space[key]
                
                if isinstance(space, dict):
                    if "low" in space and "high" in space:
                        validated_action[key] = np.clip(
                            value,
                            space["low"],
                            space["high"]
                        )
                    elif "discrete" in space:
                        validated_action[key] = int(value) % space["discrete"]
                else:
                    validated_action[key] = value
                    
        return validated_action
        
    def _determine_learning_mode(
        self,
        experience: Dict[str, Any]
    ) -> LearningMode:
        """Determine appropriate learning mode."""
        if "labels" in experience:
            return LearningMode.SUPERVISED
        elif "reward" in experience:
            return LearningMode.REINFORCEMENT
        elif "task_sequence" in experience:
            return LearningMode.CONTINUAL
        elif "support_set" in experience:
            return LearningMode.META
        else:
            return LearningMode.UNSUPERVISED
            
    async def _prepare_learning_task(
        self,
        experience: Dict[str, Any],
        mode: LearningMode
    ) -> Dict[str, Any]:
        """Prepare experience data for learning."""
        task_data = {
            "mode": mode,
            "timestamp": datetime.now().isoformat()
        }
        
        if mode == LearningMode.SUPERVISED:
            task_data.update({
                "inputs": experience["inputs"],
                "targets": experience["labels"]
            })
        elif mode == LearningMode.REINFORCEMENT:
            task_data.update({
                "experiences": experience["experiences"],
                "environment": experience.get("environment", {})
            })
        elif mode == LearningMode.META:
            task_data.update({
                "support_set": experience["support_set"],
                "query_set": experience["query_set"],
                "task_encoding": await self._encode_task(experience["task"])
            })
        elif mode == LearningMode.CONTINUAL:
            task_data.update({
                "task_sequence": experience["task_sequence"]
            })
        else:
            task_data.update({
                "data": experience["data"]
            })
            
        return task_data
        
    async def _encode_task(
        self,
        task: Dict[str, Any]
    ) -> torch.Tensor:
        """Encode task for meta-learning."""
        # Use perception system to encode task description
        task_encoding = await self.perception.encode_task(task)
        return task_encoding
        
    async def _get_observation(
        self,
        environment: Any
    ) -> Dict[str, Any]:
        """Get observation from environment."""
        if hasattr(environment, "get_observation"):
            return await environment.get_observation()
        return environment.reset()
        
    async def _execute_action(
        self,
        environment: Any,
        action: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Execute action in environment."""
        if hasattr(environment, "step_async"):
            return await environment.step_async(action)
        return environment.step(action)
        
    async def save_checkpoint(self) -> None:
        """Save agent checkpoint."""
        checkpoint_path = Path(self.config.save_dir) / f"checkpoint_{self.total_steps}"
        
        # Save learning system state
        self.learning_system.save_state(
            str(checkpoint_path / "learning_system.json")
        )
        
        # Save agent state
        agent_state = {
            "episode_count": self.episode_count,
            "total_steps": self.total_steps,
            "current_task": self.current_task,
            "performance_history": self.performance_history
        }
        
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        with open(checkpoint_path / "agent_state.json", "w") as f:
            json.dump(agent_state, f, indent=2, default=str)
            
        self.logger.info(f"Saved checkpoint at step {self.total_steps}")
        
    async def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load agent checkpoint."""
        checkpoint_dir = Path(checkpoint_path)
        
        # Load learning system state
        self.learning_system.load_state(
            str(checkpoint_dir / "learning_system.json")
        )
        
        # Load agent state
        with open(checkpoint_dir / "agent_state.json", "r") as f:
            agent_state = json.load(f)
            
        self.episode_count = agent_state["episode_count"]
        self.total_steps = agent_state["total_steps"]
        self.current_task = agent_state["current_task"]
        self.performance_history = agent_state["performance_history"]
        
        self.logger.info(f"Loaded checkpoint from step {self.total_steps}") 