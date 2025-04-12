"""Hyperparameter optimization module for H-MAS."""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Callable, Union
from dataclasses import dataclass
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import ray
from ray import tune
from ray.tune.schedulers import PopulationBasedTraining
import json
from pathlib import Path
import logging
from datetime import datetime
import copy
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from .environments import EnvironmentType

@dataclass
class OptimizationConfig:
    """Configuration for hyperparameter optimization."""
    num_trials: int = 100
    num_epochs: int = 50
    population_size: int = 4
    perturbation_interval: int = 10
    exploration_decay: float = 0.7
    num_parallel: int = 4
    save_dir: str = "optimization_results"
    seed: Optional[int] = None
    evaluation_episodes: int = 10
    checkpoint_freq: int = 5
    early_stopping_patience: int = 10
    
class ParameterSpace:
    """Defines searchable parameter spaces for different components."""
    
    @staticmethod
    def get_replay_space() -> Dict[str, Any]:
        """Get parameter space for experience replay."""
        return {
            "capacity": tune.loguniform(1e4, 1e6),
            "alpha": tune.uniform(0.4, 0.8),
            "beta": tune.uniform(0.3, 0.7),
            "beta_increment": tune.loguniform(1e-4, 1e-2),
            "n_step": tune.randint(1, 5),
            "gamma": tune.uniform(0.95, 0.999),
            "batch_size": tune.choice([32, 64, 128, 256]),
            "hindsight_k": tune.randint(2, 8)
        }
        
    @staticmethod
    def get_environment_space(env_type: EnvironmentType) -> Dict[str, Any]:
        """Get parameter space for specific environment type."""
        base_space = {
            "learning_rate": tune.loguniform(1e-5, 1e-2),
            "reward_scale": tune.uniform(0.1, 10.0),
            "difficulty_increment": tune.uniform(0.05, 0.3)
        }
        
        if env_type == EnvironmentType.PERCEPTION:
            base_space.update({
                "input_resolution": tune.choice([(64, 64), (128, 128), (256, 256)]),
                "noise_std": tune.uniform(0.0, 0.2),
                "augmentation_prob": tune.uniform(0.0, 0.5)
            })
        elif env_type == EnvironmentType.COMMUNICATION:
            base_space.update({
                "vocab_size": tune.randint(1000, 10000),
                "max_sequence_length": tune.randint(50, 200),
                "embedding_dim": tune.choice([64, 128, 256])
            })
        elif env_type == EnvironmentType.PLANNING:
            base_space.update({
                "max_steps": tune.randint(50, 200),
                "branching_factor": tune.randint(2, 5),
                "horizon_length": tune.randint(5, 20)
            })
        elif env_type == EnvironmentType.REASONING:
            base_space.update({
                "num_premises": tune.randint(2, 8),
                "rule_complexity": tune.uniform(1.0, 5.0),
                "inference_steps": tune.randint(3, 10)
            })
        
        return base_space

class HyperparameterOptimizer:
    """Manages hyperparameter optimization across different components."""
    
    def __init__(self, config: OptimizationConfig):
        """Initialize optimizer."""
        self.config = config
        self.logger = logging.getLogger("hyperparameter_optimizer")
        self.study_results: Dict[str, Any] = {}
        
        # Create save directory
        Path(config.save_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize optuna study
        self.sampler = TPESampler(seed=config.seed)
        self.pruner = MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=10,
            interval_steps=1
        )
        
        # Initialize Ray for distributed optimization
        if not ray.is_initialized():
            ray.init(num_cpus=config.num_parallel)
            
    def optimize_replay(
        self,
        evaluation_fn: Callable[[Dict[str, Any]], float]
    ) -> Dict[str, Any]:
        """Optimize experience replay parameters."""
        study_name = f"replay_opt_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        study = optuna.create_study(
            study_name=study_name,
            sampler=self.sampler,
            pruner=self.pruner,
            direction="maximize"
        )
        
        def objective(trial: optuna.Trial) -> float:
            """Optimization objective."""
            params = {
                "capacity": trial.suggest_loguniform("capacity", 1e4, 1e6),
                "alpha": trial.suggest_uniform("alpha", 0.4, 0.8),
                "beta": trial.suggest_uniform("beta", 0.3, 0.7),
                "beta_increment": trial.suggest_loguniform("beta_increment", 1e-4, 1e-2),
                "n_step": trial.suggest_int("n_step", 1, 5),
                "gamma": trial.suggest_uniform("gamma", 0.95, 0.999),
                "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
                "hindsight_k": trial.suggest_int("hindsight_k", 2, 8)
            }
            
            return evaluation_fn(params)
            
        study.optimize(objective, n_trials=self.config.num_trials)
        self.study_results[study_name] = study
        
        return study.best_params
        
    def optimize_environment(
        self,
        env_type: EnvironmentType,
        evaluation_fn: Callable[[Dict[str, Any]], float]
    ) -> Dict[str, Any]:
        """Optimize environment-specific parameters."""
        scheduler = PopulationBasedTraining(
            time_attr="training_iteration",
            metric="reward",
            mode="max",
            perturbation_interval=self.config.perturbation_interval,
            hyperparam_mutations=self.get_environment_space(env_type)
        )
        
        # Define trainable function for Ray Tune
        def train_env(config: Dict[str, Any]) -> None:
            """Training function for environment optimization."""
            for i in range(self.config.num_epochs):
                reward = evaluation_fn(config)
                tune.report(reward=reward, training_iteration=i)
                
        # Run population-based training
        analysis = tune.run(
            train_env,
            name=f"{env_type.value}_opt_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            scheduler=scheduler,
            num_samples=self.config.population_size,
            config=self.get_environment_space(env_type),
            resources_per_trial={"cpu": 1, "gpu": 0.25},
            local_dir=self.config.save_dir,
            checkpoint_freq=self.config.checkpoint_freq,
            checkpoint_at_end=True,
            keep_checkpoints_num=2,
            checkpoint_score_attr="reward"
        )
        
        return analysis.get_best_config(metric="reward", mode="max")
        
    def optimize_curriculum(
        self,
        evaluation_fn: Callable[[Dict[str, Any]], float]
    ) -> Dict[str, Any]:
        """Optimize curriculum learning parameters."""
        study_name = f"curriculum_opt_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        study = optuna.create_study(
            study_name=study_name,
            sampler=self.sampler,
            pruner=self.pruner,
            direction="maximize"
        )
        
        def objective(trial: optuna.Trial) -> float:
            """Optimization objective."""
            params = {
                "initial_difficulty": trial.suggest_uniform("initial_difficulty", 0.1, 0.5),
                "difficulty_increment": trial.suggest_uniform("difficulty_increment", 0.05, 0.3),
                "success_threshold": trial.suggest_uniform("success_threshold", 0.7, 0.9),
                "window_size": trial.suggest_int("window_size", 10, 50),
                "min_steps_per_level": trial.suggest_int("min_steps_per_level", 1000, 5000),
                "max_difficulty": trial.suggest_uniform("max_difficulty", 0.8, 1.0)
            }
            
            return evaluation_fn(params)
            
        study.optimize(objective, n_trials=self.config.num_trials)
        self.study_results[study_name] = study
        
        return study.best_params
        
    def save_results(self, path: Optional[str] = None) -> None:
        """Save optimization results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = path or f"{self.config.save_dir}/results_{timestamp}.json"
        
        results = {
            "config": self.config.__dict__,
            "studies": {
                name: {
                    "best_params": study.best_params,
                    "best_value": study.best_value,
                    "trials": [
                        {
                            "params": trial.params,
                            "value": trial.value,
                            "state": trial.state.name
                        }
                        for trial in study.trials
                    ]
                }
                for name, study in self.study_results.items()
            }
        }
        
        with open(path, "w") as f:
            json.dump(results, f, indent=2)
            
    def load_results(self, path: str) -> None:
        """Load optimization results."""
        with open(path, "r") as f:
            results = json.load(f)
            
        self.config = OptimizationConfig(**results["config"])
        
        # Recreate studies from results
        for name, study_data in results["studies"].items():
            study = optuna.create_study(study_name=name)
            study.best_params = study_data["best_params"]
            study.best_value = study_data["best_value"]
            
            for trial_data in study_data["trials"]:
                trial = optuna.trial.create_trial(
                    params=trial_data["params"],
                    value=trial_data["value"],
                    state=getattr(optuna.trial.TrialState, trial_data["state"])
                )
                study.add_trial(trial)
                
            self.study_results[name] = study
            
    def visualize_results(
        self,
        study_name: Optional[str] = None
    ) -> None:
        """Visualize optimization results."""
        import plotly.graph_objects as go
        from optuna.visualization import plot_optimization_history
        from optuna.visualization import plot_parallel_coordinate
        from optuna.visualization import plot_param_importances
        
        if study_name is None:
            # Visualize all studies
            for name, study in self.study_results.items():
                self._create_study_visualizations(name, study)
        else:
            # Visualize specific study
            study = self.study_results.get(study_name)
            if study is not None:
                self._create_study_visualizations(study_name, study)
                
    def _create_study_visualizations(
        self,
        name: str,
        study: optuna.Study
    ) -> None:
        """Create and save visualizations for a study."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = Path(self.config.save_dir) / f"visualizations_{timestamp}"
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Optimization history
        fig = plot_optimization_history(study)
        fig.write_html(str(save_dir / f"{name}_history.html"))
        
        # Parallel coordinate plot
        fig = plot_parallel_coordinate(study)
        fig.write_html(str(save_dir / f"{name}_parallel.html"))
        
        # Parameter importances
        fig = plot_param_importances(study)
        fig.write_html(str(save_dir / f"{name}_importance.html"))
        
        # Create parameter correlation heatmap
        params_df = pd.DataFrame([
            trial.params for trial in study.trials if trial.value is not None
        ])
        values = np.array([
            trial.value for trial in study.trials if trial.value is not None
        ])
        
        correlations = pd.DataFrame(index=params_df.columns, columns=params_df.columns)
        for i in params_df.columns:
            for j in params_df.columns:
                if i == j:
                    correlations.loc[i, j] = 1.0
                else:
                    correlations.loc[i, j] = np.corrcoef(params_df[i], params_df[j])[0, 1]
                    
        fig = go.Figure(data=go.Heatmap(
            z=correlations.values,
            x=correlations.columns,
            y=correlations.index,
            colorscale="RdBu"
        ))
        fig.update_layout(title=f"Parameter Correlations - {name}")
        fig.write_html(str(save_dir / f"{name}_correlations.html"))
        
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of optimization results."""
        summary = {}
        
        for name, study in self.study_results.items():
            summary[name] = {
                "best_params": study.best_params,
                "best_value": study.best_value,
                "n_trials": len(study.trials),
                "study_duration": sum(
                    (t.datetime_complete - t.datetime_start).total_seconds()
                    for t in study.trials
                    if t.datetime_complete is not None
                ),
                "param_importance": self._compute_param_importance(study)
            }
            
        return summary
        
    def _compute_param_importance(
        self,
        study: optuna.Study
    ) -> Dict[str, float]:
        """Compute parameter importance scores."""
        importance = optuna.importance.get_param_importances(study)
        return {name: float(score) for name, score in importance.items()} 