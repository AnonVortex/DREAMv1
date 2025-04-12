import pytest
from unittest.mock import Mock, patch
import numpy as np

from ..adaptation_service import EvolutionaryArchitect, ArchitectureConfig

@pytest.fixture
def evolutionary_architect():
    return EvolutionaryArchitect()

@pytest.fixture
def sample_architecture_config():
    return ArchitectureConfig(
        num_agents=3,
        communication_pattern="mesh",
        learning_rate=0.01,
        batch_size=32,
        model_architecture={
            "type": "transformer",
            "num_layers": 6,
            "hidden_size": 512
        }
    )

@pytest.fixture
def sample_constraints():
    return {
        "min_agents": 2,
        "max_agents": 10,
        "min_learning_rate": 0.0001,
        "max_learning_rate": 0.1,
        "min_batch_size": 16,
        "max_batch_size": 128,
        "allowed_architectures": ["transformer", "lstm"],
        "min_layers": 2,
        "max_layers": 12
    }

def test_generate_population(evolutionary_architect, sample_architecture_config, sample_constraints):
    population_size = 10
    population = evolutionary_architect.generate_population(
        sample_architecture_config,
        sample_constraints,
        population_size
    )
    
    assert len(population) == population_size
    
    # Verify constraints are met
    for config in population:
        assert sample_constraints["min_agents"] <= config.num_agents <= sample_constraints["max_agents"]
        assert sample_constraints["min_learning_rate"] <= config.learning_rate <= sample_constraints["max_learning_rate"]
        assert sample_constraints["min_batch_size"] <= config.batch_size <= sample_constraints["max_batch_size"]
        assert config.model_architecture["type"] in sample_constraints["allowed_architectures"]
        assert sample_constraints["min_layers"] <= config.model_architecture["num_layers"] <= sample_constraints["max_layers"]

def test_evaluate_fitness(evolutionary_architect, sample_architecture_config):
    metrics = {
        "accuracy": 0.95,
        "latency": 100,
        "resource_usage": 0.7,
        "throughput": 1000
    }
    
    fitness = evolutionary_architect.evaluate_fitness(sample_architecture_config, metrics)
    assert isinstance(fitness, float)
    assert 0 <= fitness <= 1

def test_select_parents(evolutionary_architect, sample_architecture_config):
    population = [sample_architecture_config.copy() for _ in range(10)]
    fitness_scores = np.random.random(10)
    
    parents = evolutionary_architect.select_parents(population, fitness_scores)
    assert len(parents) == 2
    assert all(parent in population for parent in parents)

def test_crossover(evolutionary_architect, sample_architecture_config):
    parent1 = sample_architecture_config.copy()
    parent2 = sample_architecture_config.copy()
    parent2.num_agents = 5
    parent2.learning_rate = 0.02
    
    offspring = evolutionary_architect.crossover(parent1, parent2)
    
    assert isinstance(offspring, ArchitectureConfig)
    assert offspring.num_agents in [parent1.num_agents, parent2.num_agents]
    assert offspring.learning_rate in [parent1.learning_rate, parent2.learning_rate]

def test_mutate(evolutionary_architect, sample_architecture_config, sample_constraints):
    mutated = evolutionary_architect.mutate(
        sample_architecture_config,
        sample_constraints,
        mutation_rate=0.5
    )
    
    assert isinstance(mutated, ArchitectureConfig)
    # Verify constraints are still met
    assert sample_constraints["min_agents"] <= mutated.num_agents <= sample_constraints["max_agents"]
    assert sample_constraints["min_learning_rate"] <= mutated.learning_rate <= sample_constraints["max_learning_rate"]

def test_evolve_architecture(evolutionary_architect, sample_architecture_config, sample_constraints):
    metrics = {
        "accuracy": 0.95,
        "latency": 100,
        "resource_usage": 0.7,
        "throughput": 1000
    }
    
    evolved_config = evolutionary_architect.evolve_architecture(
        sample_architecture_config,
        sample_constraints,
        metrics,
        generations=5,
        population_size=10
    )
    
    assert isinstance(evolved_config, ArchitectureConfig)
    # Verify evolved config meets constraints
    assert sample_constraints["min_agents"] <= evolved_config.num_agents <= sample_constraints["max_agents"]
    assert sample_constraints["min_learning_rate"] <= evolved_config.learning_rate <= sample_constraints["max_learning_rate"]

def test_constraint_violation_handling(evolutionary_architect, sample_architecture_config, sample_constraints):
    # Test handling of invalid configurations
    invalid_config = sample_architecture_config.copy()
    invalid_config.num_agents = sample_constraints["max_agents"] + 1
    
    # Verify mutation fixes constraint violations
    mutated = evolutionary_architect.mutate(invalid_config, sample_constraints)
    assert sample_constraints["min_agents"] <= mutated.num_agents <= sample_constraints["max_agents"]

def test_fitness_calculation_weights(evolutionary_architect):
    # Test different weight configurations
    metrics = {
        "accuracy": 0.95,
        "latency": 100,
        "resource_usage": 0.7,
        "throughput": 1000
    }
    
    # Test with different weight configurations
    weights = [
        {"accuracy": 0.4, "latency": 0.2, "resource_usage": 0.2, "throughput": 0.2},
        {"accuracy": 0.25, "latency": 0.25, "resource_usage": 0.25, "throughput": 0.25},
        {"accuracy": 0.7, "latency": 0.1, "resource_usage": 0.1, "throughput": 0.1}
    ]
    
    fitness_scores = [
        evolutionary_architect.evaluate_fitness(sample_architecture_config, metrics, w)
        for w in weights
    ]
    
    assert len(set(fitness_scores)) == len(weights)  # Different weights should yield different scores

def test_evolution_convergence(evolutionary_architect, sample_architecture_config, sample_constraints):
    metrics = {
        "accuracy": 0.95,
        "latency": 100,
        "resource_usage": 0.7,
        "throughput": 1000
    }
    
    # Track fitness over generations
    fitness_history = []
    
    for _ in range(5):
        evolved_config = evolutionary_architect.evolve_architecture(
            sample_architecture_config,
            sample_constraints,
            metrics,
            generations=5,
            population_size=10
        )
        fitness = evolutionary_architect.evaluate_fitness(evolved_config, metrics)
        fitness_history.append(fitness)
    
    # Verify improvement or stability in fitness
    assert fitness_history[-1] >= fitness_history[0]

def test_architecture_diversity(evolutionary_architect, sample_architecture_config, sample_constraints):
    population_size = 10
    population = evolutionary_architect.generate_population(
        sample_architecture_config,
        sample_constraints,
        population_size
    )
    
    # Check diversity in various parameters
    unique_agents = len(set(config.num_agents for config in population))
    unique_learning_rates = len(set(config.learning_rate for config in population))
    unique_architectures = len(set(config.model_architecture["type"] for config in population))
    
    # Ensure some diversity in the population
    assert unique_agents > 1
    assert unique_learning_rates > 1
    assert unique_architectures > 0 