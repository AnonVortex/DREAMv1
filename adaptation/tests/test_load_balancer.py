import pytest
from datetime import datetime
from unittest.mock import Mock, patch
import numpy as np

from ..adaptation_service import LoadBalancer, ResourceMetrics, Task, Agent

@pytest.fixture
def load_balancer():
    return LoadBalancer()

@pytest.fixture
def sample_tasks():
    return [
        Task(
            task_id="task1",
            requirements={
                "cpu": 0.5,
                "memory": "512M",
                "capabilities": ["vision"]
            },
            priority=1
        ),
        Task(
            task_id="task2",
            requirements={
                "cpu": 1.0,
                "memory": "1G",
                "capabilities": ["nlp"]
            },
            priority=2
        )
    ]

@pytest.fixture
def sample_agents():
    return [
        Agent(
            agent_id="agent1",
            capabilities=["vision", "nlp"],
            current_load={
                "cpu": 0.3,
                "memory": "256M",
                "tasks": 2
            },
            status="running"
        ),
        Agent(
            agent_id="agent2",
            capabilities=["vision"],
            current_load={
                "cpu": 0.1,
                "memory": "128M",
                "tasks": 1
            },
            status="running"
        )
    ]

def test_calculate_agent_score(load_balancer, sample_agents):
    agent = sample_agents[0]
    score = load_balancer.calculate_agent_score(agent)
    
    assert isinstance(score, float)
    assert 0 <= score <= 1.0  # Score should be normalized

def test_check_task_compatibility(load_balancer, sample_tasks, sample_agents):
    task = sample_tasks[0]
    agent = sample_agents[0]
    
    # Test compatible agent
    assert load_balancer.check_task_compatibility(task, agent)
    
    # Test incompatible agent (missing capability)
    agent.capabilities = ["nlp"]
    assert not load_balancer.check_task_compatibility(task, agent)

def test_distribute_load(load_balancer, sample_tasks, sample_agents):
    # Test load distribution
    distribution = load_balancer.distribute_load(sample_tasks, sample_agents)
    
    assert isinstance(distribution, dict)
    assert len(distribution) > 0
    
    # Verify task assignments
    for agent_id, tasks in distribution.items():
        assert isinstance(tasks, list)
        assert all(isinstance(task, Task) for task in tasks)

def test_distribute_load_no_agents(load_balancer, sample_tasks):
    # Test distribution with no available agents
    with pytest.raises(Exception) as exc_info:
        load_balancer.distribute_load(sample_tasks, [])
    
    assert "No available agents" in str(exc_info.value)

def test_distribute_load_no_tasks(load_balancer, sample_agents):
    # Test distribution with no tasks
    distribution = load_balancer.distribute_load([], sample_agents)
    
    assert isinstance(distribution, dict)
    assert len(distribution) == 0

def test_handle_agent_failure(load_balancer, sample_tasks, sample_agents):
    # Setup initial distribution
    initial_distribution = load_balancer.distribute_load(sample_tasks, sample_agents)
    
    # Simulate agent failure
    failed_agent_id = "agent1"
    new_distribution = load_balancer.handle_agent_failure(
        failed_agent_id,
        initial_distribution,
        sample_agents
    )
    
    assert failed_agent_id not in new_distribution
    assert len(new_distribution) < len(initial_distribution)

def test_optimize_distribution(load_balancer, sample_tasks, sample_agents):
    # Create initial unbalanced distribution
    unbalanced_distribution = {
        "agent1": sample_tasks,  # All tasks assigned to one agent
        "agent2": []
    }
    
    # Test optimization
    optimized_distribution = load_balancer.optimize_distribution(
        unbalanced_distribution,
        sample_agents
    )
    
    assert len(optimized_distribution["agent1"]) < len(sample_tasks)
    assert len(optimized_distribution["agent2"]) > 0

def test_calculate_load_metrics(load_balancer, sample_agents):
    metrics = load_balancer.calculate_load_metrics(sample_agents)
    
    assert "average_cpu_load" in metrics
    assert "average_memory_load" in metrics
    assert "total_tasks" in metrics
    assert isinstance(metrics["average_cpu_load"], float)

def test_check_overload_threshold(load_balancer, sample_agents):
    # Test overload detection
    agent = sample_agents[0]
    agent.current_load["cpu"] = 0.9  # High CPU load
    
    is_overloaded = load_balancer.check_overload_threshold(agent)
    assert is_overloaded
    
    # Test normal load
    agent.current_load["cpu"] = 0.5
    is_overloaded = load_balancer.check_overload_threshold(agent)
    assert not is_overloaded

def test_prioritize_tasks(load_balancer, sample_tasks):
    # Add a high-priority task
    high_priority_task = Task(
        task_id="task3",
        requirements={
            "cpu": 0.5,
            "memory": "512M",
            "capabilities": ["vision"]
        },
        priority=3
    )
    tasks = sample_tasks + [high_priority_task]
    
    prioritized_tasks = load_balancer.prioritize_tasks(tasks)
    
    assert len(prioritized_tasks) == len(tasks)
    assert prioritized_tasks[0].task_id == "task3"  # Highest priority first

def test_estimate_completion_time(load_balancer, sample_tasks, sample_agents):
    # Test completion time estimation
    estimates = load_balancer.estimate_completion_time(sample_tasks, sample_agents)
    
    assert isinstance(estimates, dict)
    assert all(isinstance(time, float) for time in estimates.values())

def test_handle_new_agent(load_balancer, sample_tasks, sample_agents):
    # Create initial distribution
    initial_distribution = load_balancer.distribute_load(sample_tasks, sample_agents)
    
    # Add new agent
    new_agent = Agent(
        agent_id="agent3",
        capabilities=["vision", "nlp"],
        current_load={
            "cpu": 0.0,
            "memory": "0M",
            "tasks": 0
        },
        status="running"
    )
    
    # Test redistribution with new agent
    new_distribution = load_balancer.handle_new_agent(
        new_agent,
        initial_distribution,
        sample_agents + [new_agent]
    )
    
    assert "agent3" in new_distribution
    assert len(new_distribution) > len(initial_distribution)

def test_calculate_agent_efficiency(load_balancer, sample_agents):
    agent = sample_agents[0]
    efficiency = load_balancer.calculate_agent_efficiency(agent)
    
    assert isinstance(efficiency, float)
    assert 0 <= efficiency <= 1.0

def test_get_agent_recommendations(load_balancer, sample_agents):
    recommendations = load_balancer.get_agent_recommendations(sample_agents)
    
    assert isinstance(recommendations, list)
    assert all(isinstance(rec, dict) for rec in recommendations)
    assert all("agent_id" in rec for rec in recommendations)
    assert all("recommendation" in rec for rec in recommendations)

def test_calculate_agent_scores(load_balancer, sample_agents):
    scores = load_balancer.calculate_agent_scores(sample_agents)
    
    assert len(scores) == len(sample_agents)
    assert all(0 <= score <= 1 for score in scores)
    
    # Higher performance and lower load should result in higher scores
    agent_scores = list(zip(sample_agents, scores))
    agent_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Agent3 should have highest score (high performance, low load)
    assert agent_scores[0][0]["id"] == "agent3"

def test_load_balancing_priority(load_balancer, sample_tasks, sample_agents):
    # Modify tasks to have different priorities
    high_priority_task = sample_tasks[3]  # Planning task with priority 3
    
    distribution = load_balancer.distribute_load(sample_tasks, sample_agents)
    
    # High priority task should be assigned to the best compatible agent
    assigned_agent = next(agent for agent in sample_agents 
                        if agent["id"] == distribution[high_priority_task["id"]])
    assert "planning" in assigned_agent["capabilities"]
    assert assigned_agent["id"] == "agent2"  # Only agent with planning capability

def test_load_distribution_fairness(load_balancer, sample_tasks, sample_agents):
    # Run multiple distributions
    num_trials = 10
    agent_assignment_counts = {agent["id"]: 0 for agent in sample_agents}
    
    for _ in range(num_trials):
        distribution = load_balancer.distribute_load(sample_tasks, sample_agents)
        for agent_id in distribution.values():
            agent_assignment_counts[agent_id] += 1
    
    # Check if load is somewhat evenly distributed
    counts = list(agent_assignment_counts.values())
    assert max(counts) - min(counts) <= num_trials * 0.5  # Allow some variance

def test_overloaded_agents(load_balancer, sample_tasks, sample_agents):
    # Simulate an overloaded agent
    overloaded_agent = sample_agents[0]
    overloaded_agent["current_load"] = 0.9
    
    distribution = load_balancer.distribute_load(sample_tasks, sample_agents)
    
    # Count tasks assigned to overloaded agent
    overloaded_assignments = sum(1 for agent_id in distribution.values()
                               if agent_id == overloaded_agent["id"])
    
    # Overloaded agent should receive fewer tasks
    assert overloaded_assignments <= len(sample_tasks) / len(sample_agents)

def test_performance_based_distribution(load_balancer, sample_tasks, sample_agents):
    # Modify agent performance scores
    high_performer = sample_agents[2]
    high_performer["performance_score"] = 1.0
    high_performer["capabilities"] = ["nlp", "vision", "planning"]
    
    distribution = load_balancer.distribute_load(sample_tasks, sample_agents)
    
    # Count tasks assigned to high performer
    high_performer_tasks = sum(1 for agent_id in distribution.values()
                             if agent_id == high_performer["id"])
    
    # High performer should get more tasks
    assert high_performer_tasks >= len(sample_tasks) / len(sample_agents)

def test_handle_no_compatible_agents(load_balancer, sample_tasks, sample_agents):
    # Add a task with requirements that no agent can meet
    impossible_task = {
        "id": "impossible_task",
        "complexity": 0.5,
        "priority": 1,
        "requirements": ["quantum_computing"]
    }
    tasks = sample_tasks + [impossible_task]
    
    with pytest.raises(ValueError):
        load_balancer.distribute_load(tasks, sample_agents)

def test_handle_empty_inputs(load_balancer):
    # Test with empty task list
    distribution = load_balancer.distribute_load([], sample_agents)
    assert len(distribution) == 0
    
    # Test with empty agent list
    with pytest.raises(ValueError):
        load_balancer.distribute_load(sample_tasks, [])

def test_dynamic_load_adjustment(load_balancer, sample_tasks, sample_agents):
    # First distribution
    initial_distribution = load_balancer.distribute_load(sample_tasks, sample_agents)
    
    # Simulate changed load conditions
    sample_agents[0]["current_load"] = 0.8
    sample_agents[1]["current_load"] = 0.2
    
    # Second distribution
    new_distribution = load_balancer.distribute_load(sample_tasks, sample_agents)
    
    # Distribution should change based on new load conditions
    assert new_distribution != initial_distribution

def test_check_task_compatibility(load_balancer, sample_tasks, sample_agents):
    # Test task compatibility check
    task = sample_tasks[0]
    agent_metrics = sample_agents["agent1"]
    
    is_compatible = load_balancer.check_task_compatibility(task, agent_metrics)
    assert isinstance(is_compatible, bool)
    
    # Test with incompatible task
    incompatible_task = {
        "id": "heavy_task",
        "priority": 1,
        "resource_requirements": {"cpu": 0.9, "memory": 0.9}
    }
    is_compatible = load_balancer.check_task_compatibility(incompatible_task, agent_metrics)
    assert not is_compatible

def test_calculate_load_score(load_balancer, sample_agents):
    # Test load score calculation
    agent_metrics = sample_agents["agent1"]
    load_score = load_balancer.calculate_load_score(agent_metrics)
    
    # Verify score is within expected range
    assert 0 <= load_score <= 1
    
    # Compare scores of different agents
    agent1_score = load_balancer.calculate_load_score(sample_agents["agent1"])
    agent2_score = load_balancer.calculate_load_score(sample_agents["agent2"])
    assert agent1_score < agent2_score  # agent1 has lower resource usage

def test_priority_based_distribution(load_balancer, sample_tasks, sample_agents):
    # Add priority weights
    load_balancer.priority_weights = {1: 0.5, 2: 1.0, 3: 2.0}
    
    # Test priority-based distribution
    distribution = load_balancer.distribute_load(sample_tasks, sample_agents)
    
    # Verify high-priority tasks are assigned to less loaded agents
    for agent_id, tasks in distribution.items():
        agent_load = sample_agents[agent_id].cpu_usage
        if agent_load < 50:  # Less loaded agent
            assert any(task["priority"] == 3 for task in tasks)  # Should have high-priority tasks

def test_resource_threshold_handling(load_balancer, sample_tasks, sample_agents):
    # Set resource thresholds
    load_balancer.cpu_threshold = 80.0
    load_balancer.memory_threshold = 85.0
    
    # Add a heavily loaded agent
    sample_agents["agent3"] = ResourceMetrics(
        timestamp=datetime.now(),
        cpu_usage=85.0,
        memory_usage=90.0,
        network_io={'rx_bytes': 5000, 'tx_bytes': 6000},
        agent_id="agent3"
    )
    
    # Test distribution with threshold consideration
    distribution = load_balancer.distribute_load(sample_tasks, sample_agents)
    
    # Verify no new tasks are assigned to overloaded agent
    assert "agent3" not in distribution or len(distribution["agent3"]) == 0 