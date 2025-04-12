"""Visualization tools for monitoring H-MAS training environments."""

from typing import Dict, List, Optional, Any, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import seaborn as sns
from datetime import datetime
import pandas as pd
from pathlib import Path
import json
from dataclasses import dataclass
from .environments import EnvironmentType

@dataclass
class VisualizationConfig:
    """Configuration for visualization settings."""
    output_dir: str = "visualizations"
    update_interval: int = 100
    max_history: int = 10000
    plot_style: str = "darkgrid"
    save_format: str = "png"
    dpi: int = 300

class PerformanceTracker:
    """Tracks and visualizes agent performance metrics."""
    
    def __init__(self, config: VisualizationConfig):
        """Initialize performance tracker."""
        self.config = config
        self.metrics_history = {
            env_type.value: {
                "rewards": [],
                "steps": [],
                "timestamps": [],
                "task_success": [],
                "learning_curves": []
            }
            for env_type in EnvironmentType
        }
        
        # Create output directory
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Set plot style
        sns.set_style(config.plot_style)
        
    def update_metrics(
        self,
        env_type: EnvironmentType,
        step_metrics: Dict[str, Any]
    ) -> None:
        """Update performance metrics."""
        metrics = self.metrics_history[env_type.value]
        
        # Update basic metrics
        metrics["rewards"].append(step_metrics.get("reward", 0.0))
        metrics["steps"].append(step_metrics.get("step", 0))
        metrics["timestamps"].append(datetime.now())
        metrics["task_success"].append(step_metrics.get("success", False))
        
        # Update learning curve
        if "learning_progress" in step_metrics:
            metrics["learning_curves"].append(step_metrics["learning_progress"])
            
        # Trim history if needed
        if len(metrics["rewards"]) > self.config.max_history:
            for key in metrics:
                metrics[key] = metrics[key][-self.config.max_history:]
                
    def plot_reward_history(
        self,
        env_type: EnvironmentType,
        window_size: int = 100
    ) -> Figure:
        """Plot reward history with moving average."""
        metrics = self.metrics_history[env_type.value]
        rewards = np.array(metrics["rewards"])
        steps = np.array(metrics["steps"])
        
        # Calculate moving average
        moving_avg = np.convolve(
            rewards,
            np.ones(window_size) / window_size,
            mode="valid"
        )
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(steps[window_size-1:], moving_avg, label="Moving Average")
        ax.plot(steps, rewards, alpha=0.3, label="Raw Rewards")
        
        ax.set_title(f"Reward History - {env_type.value}")
        ax.set_xlabel("Steps")
        ax.set_ylabel("Reward")
        ax.legend()
        
        return fig
        
    def plot_task_success_rate(
        self,
        env_type: EnvironmentType,
        window_size: int = 1000
    ) -> Figure:
        """Plot task success rate over time."""
        metrics = self.metrics_history[env_type.value]
        success = np.array(metrics["task_success"], dtype=float)
        steps = np.array(metrics["steps"])
        
        # Calculate success rate
        success_rate = np.convolve(
            success,
            np.ones(window_size) / window_size,
            mode="valid"
        )
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(steps[window_size-1:], success_rate)
        
        ax.set_title(f"Task Success Rate - {env_type.value}")
        ax.set_xlabel("Steps")
        ax.set_ylabel("Success Rate")
        ax.set_ylim(0, 1)
        
        return fig
        
    def plot_learning_curves(
        self,
        env_type: EnvironmentType
    ) -> Figure:
        """Plot learning curves showing skill acquisition."""
        metrics = self.metrics_history[env_type.value]
        learning_curves = np.array(metrics["learning_curves"])
        steps = np.array(metrics["steps"])
        
        if len(learning_curves) == 0:
            return None
            
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot multiple skills if available
        if learning_curves.ndim > 1:
            for i in range(learning_curves.shape[1]):
                ax.plot(steps, learning_curves[:, i], label=f"Skill {i+1}")
            ax.legend()
        else:
            ax.plot(steps, learning_curves)
            
        ax.set_title(f"Learning Curves - {env_type.value}")
        ax.set_xlabel("Steps")
        ax.set_ylabel("Skill Level")
        ax.set_ylim(0, 1)
        
        return fig
        
    def plot_performance_comparison(
        self,
        window_size: int = 1000
    ) -> Figure:
        """Plot performance comparison across environments."""
        fig, ax = plt.subplots(figsize=(15, 8))
        
        for env_type in EnvironmentType:
            metrics = self.metrics_history[env_type.value]
            if len(metrics["rewards"]) > window_size:
                rewards = np.array(metrics["rewards"])
                steps = np.array(metrics["steps"])
                
                # Calculate moving average
                moving_avg = np.convolve(
                    rewards,
                    np.ones(window_size) / window_size,
                    mode="valid"
                )
                
                ax.plot(
                    steps[window_size-1:],
                    moving_avg,
                    label=env_type.value
                )
                
        ax.set_title("Performance Comparison Across Environments")
        ax.set_xlabel("Steps")
        ax.set_ylabel("Average Reward")
        ax.legend()
        
        return fig
        
    def save_visualization(
        self,
        figure: Figure,
        name: str
    ) -> None:
        """Save visualization to file."""
        output_path = Path(self.config.output_dir) / f"{name}.{self.config.save_format}"
        figure.savefig(
            output_path,
            dpi=self.config.dpi,
            bbox_inches="tight"
        )
        plt.close(figure)
        
    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        report = {}
        
        for env_type in EnvironmentType:
            metrics = self.metrics_history[env_type.value]
            
            if len(metrics["rewards"]) > 0:
                recent_window = min(1000, len(metrics["rewards"]))
                recent_rewards = metrics["rewards"][-recent_window:]
                recent_success = metrics["task_success"][-recent_window:]
                
                env_report = {
                    "average_reward": float(np.mean(recent_rewards)),
                    "reward_std": float(np.std(recent_rewards)),
                    "success_rate": float(np.mean(recent_success)),
                    "total_steps": int(metrics["steps"][-1]),
                    "learning_progress": float(
                        np.mean(metrics["learning_curves"][-1])
                        if len(metrics["learning_curves"]) > 0
                        else 0.0
                    )
                }
                
                report[env_type.value] = env_report
                
        return report
        
    def save_metrics_history(self) -> None:
        """Save metrics history to file."""
        output_path = Path(self.config.output_dir) / "metrics_history.json"
        
        # Convert numpy arrays and timestamps to serializable format
        serializable_history = {}
        for env_type, metrics in self.metrics_history.items():
            serializable_history[env_type] = {
                "rewards": [float(r) for r in metrics["rewards"]],
                "steps": [int(s) for s in metrics["steps"]],
                "timestamps": [ts.isoformat() for ts in metrics["timestamps"]],
                "task_success": [bool(s) for s in metrics["task_success"]],
                "learning_curves": [
                    [float(v) for v in curve]
                    for curve in metrics["learning_curves"]
                ]
            }
            
        with open(output_path, "w") as f:
            json.dump(serializable_history, f, indent=2)
            
    def load_metrics_history(self, file_path: str) -> None:
        """Load metrics history from file."""
        with open(file_path, "r") as f:
            loaded_history = json.load(f)
            
        # Convert loaded data back to appropriate types
        for env_type, metrics in loaded_history.items():
            self.metrics_history[env_type] = {
                "rewards": np.array(metrics["rewards"]),
                "steps": np.array(metrics["steps"]),
                "timestamps": [
                    datetime.fromisoformat(ts)
                    for ts in metrics["timestamps"]
                ],
                "task_success": np.array(metrics["task_success"]),
                "learning_curves": np.array(metrics["learning_curves"])
            }
            
class EnvironmentVisualizer:
    """Visualizes environment states and agent interactions."""
    
    def __init__(self, config: VisualizationConfig):
        """Initialize environment visualizer."""
        self.config = config
        
    def visualize_perception_state(
        self,
        state: np.ndarray,
        task: str
    ) -> Figure:
        """Visualize perception environment state."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if task == "object_recognition":
            # Visualize object features
            sns.heatmap(
                state.reshape(-1, int(np.sqrt(state.size))),
                ax=ax,
                cmap="viridis"
            )
            ax.set_title("Object Features")
            
        elif task == "pattern_matching":
            # Visualize pattern
            ax.plot(state)
            ax.set_title("Pattern")
            
        elif task == "anomaly_detection":
            # Visualize sample with potential anomaly
            ax.plot(state)
            ax.set_title("Sample")
            
        else:  # feature_extraction
            # Visualize feature importance
            sns.barplot(
                x=np.arange(len(state)),
                y=state,
                ax=ax
            )
            ax.set_title("Feature Importance")
            
        return fig
        
    def visualize_communication_state(
        self,
        state: Dict[str, np.ndarray],
        task: str
    ) -> Figure:
        """Visualize communication environment state."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        if task == "message_passing":
            # Visualize message encoding
            sns.heatmap(
                state["message"].reshape(-1, int(np.sqrt(state["message"].size))),
                ax=ax,
                cmap="viridis"
            )
            ax.set_title("Message Encoding")
            
        elif task == "consensus_building":
            # Visualize agent opinions
            for i, opinion in enumerate(state["opinions"]):
                ax.plot(opinion, label=f"Agent {i+1}")
            ax.set_title("Agent Opinions")
            ax.legend()
            
        elif task == "information_sharing":
            # Visualize information distribution
            sns.heatmap(state["information"], ax=ax, cmap="viridis")
            ax.set_title("Information Distribution")
            
        else:  # coordination
            # Visualize coordination state
            ax.plot(state["coordination"])
            ax.set_title("Coordination State")
            
        return fig
        
    def visualize_planning_state(
        self,
        state: Dict[str, np.ndarray],
        task: str
    ) -> Figure:
        """Visualize planning environment state."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        if task == "goal_decomposition":
            # Visualize goal hierarchy
            for i, subgoal in enumerate(state["subgoals"]):
                ax.plot(subgoal, label=f"Subgoal {i+1}")
            ax.plot(state["main_goal"], "k--", label="Main Goal")
            ax.set_title("Goal Decomposition")
            ax.legend()
            
        elif task == "resource_allocation":
            # Visualize resource distribution
            sns.heatmap(state["resources"], ax=ax, cmap="viridis")
            ax.set_title("Resource Allocation")
            
        elif task == "sequential_decision":
            # Visualize action sequence
            ax.plot(state["current"], label="Current")
            ax.plot(state["target"], "r--", label="Target")
            ax.set_title("Sequential Decision Making")
            ax.legend()
            
        else:  # contingency_planning
            # Visualize contingency plans
            for i, plan in enumerate(state["plans"]):
                ax.plot(plan, label=f"Plan {i+1}")
            ax.set_title("Contingency Plans")
            ax.legend()
            
        return fig
        
    def visualize_reasoning_state(
        self,
        state: Dict[str, np.ndarray],
        task: str
    ) -> Figure:
        """Visualize reasoning environment state."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        if task == "logical_inference":
            # Visualize premises and conclusion
            for i, premise in enumerate(state["premises"]):
                ax.plot(premise, label=f"Premise {i+1}")
            ax.plot(state["conclusion"], "r--", label="Conclusion")
            ax.set_title("Logical Inference")
            ax.legend()
            
        elif task == "causal_reasoning":
            # Visualize causal chain
            sns.heatmap(state["causality"], ax=ax, cmap="viridis")
            ax.set_title("Causal Chain")
            
        elif task == "analogical_reasoning":
            # Visualize source and target domains
            ax.plot(state["source"], label="Source")
            ax.plot(state["target"], "r--", label="Target")
            ax.set_title("Analogical Reasoning")
            ax.legend()
            
        else:  # deductive_reasoning
            # Visualize logical steps
            for i, step in enumerate(state["steps"]):
                ax.plot(step, label=f"Step {i+1}")
            ax.set_title("Deductive Reasoning")
            ax.legend()
            
        return fig
        
    def visualize_creativity_state(
        self,
        state: Dict[str, np.ndarray],
        task: str
    ) -> Figure:
        """Visualize creativity environment state."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        if task == "divergent_thinking":
            # Visualize creative variations
            for i, variation in enumerate(state["variations"]):
                ax.plot(variation, label=f"Variation {i+1}")
            ax.plot(state["original"], "k--", label="Original")
            ax.set_title("Divergent Thinking")
            ax.legend()
            
        elif task == "pattern_innovation":
            # Visualize innovative pattern
            ax.plot(state["pattern"])
            ax.set_title("Pattern Innovation")
            
        elif task == "concept_synthesis":
            # Visualize concept space
            sns.heatmap(state["concepts"], ax=ax, cmap="viridis")
            ax.set_title("Concept Synthesis")
            
        else:  # adaptive_design
            # Visualize design adaptation
            ax.plot(state["design"], label="Current")
            ax.plot(state["target"], "r--", label="Target")
            ax.set_title("Adaptive Design")
            ax.legend()
            
        return fig
        
    def visualize_teaching_state(
        self,
        state: Dict[str, np.ndarray],
        task: str
    ) -> Figure:
        """Visualize teaching environment state."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        if task == "knowledge_distillation":
            # Visualize knowledge transfer
            sns.heatmap(
                np.vstack([state["teacher"], state["student"]]),
                ax=ax,
                cmap="viridis"
            )
            ax.set_title("Knowledge Transfer")
            ax.set_ylabel("Teacher / Student")
            
        elif task == "curriculum_design":
            # Visualize curriculum progression
            sns.heatmap(state["curriculum"], ax=ax, cmap="viridis")
            ax.set_title("Curriculum Design")
            
        elif task == "feedback_generation":
            # Visualize feedback history
            for i, feedback in enumerate(state["feedback"]):
                ax.plot(feedback, label=f"Time {i+1}")
            ax.set_title("Feedback History")
            ax.legend()
            
        else:  # adaptive_instruction
            # Visualize adaptation
            ax.plot(state["performance"], label="Performance")
            ax.plot(state["target"], "r--", label="Target")
            ax.set_title("Adaptive Instruction")
            ax.legend()
            
        return fig
        
    def save_visualization(
        self,
        figure: Figure,
        name: str
    ) -> None:
        """Save visualization to file."""
        output_path = Path(self.config.output_dir) / f"{name}.{self.config.save_format}"
        figure.savefig(
            output_path,
            dpi=self.config.dpi,
            bbox_inches="tight"
        )
        plt.close(figure) 