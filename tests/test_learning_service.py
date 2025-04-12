import pytest
from fastapi.testclient import TestClient
from learning.learning_service import app, LearningManager
import numpy as np
import json
from datetime import datetime

@pytest.fixture
def test_client():
    return TestClient(app)

@pytest.fixture
def learning_manager():
    return LearningManager()

def test_register_agent(test_client):
    agent_config = {
        "agent_id": "test_agent",
        "learning_type": "reinforcement",
        "model_config": {
            "state_size": 10,
            "action_size": 5,
            "hidden_layers": [64, 32]
        },
        "hyperparameters": {
            "learning_rate": 0.001,
            "discount_factor": 0.99
        }
    }
    
    response = test_client.post("/agents/register", json=agent_config)
    assert response.status_code == 200
    data = response.json()
    assert data["agent_id"] == "test_agent"
    assert "model_id" in data

def test_process_experience(test_client):
    # First register an agent
    agent_config = {
        "agent_id": "test_agent_exp",
        "learning_type": "reinforcement",
        "model_config": {
            "state_size": 3,
            "action_size": 2
        }
    }
    test_client.post("/agents/register", json=agent_config)
    
    # Submit experience
    experience = {
        "agent_id": "test_agent_exp",
        "state": [0.1, 0.2, 0.3],
        "action": 1,
        "reward": 1.0,
        "next_state": [0.2, 0.3, 0.4],
        "done": False
    }
    
    response = test_client.post("/experience", json=experience)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "experience_id" in data

def test_get_action(test_client):
    # First register an agent
    agent_config = {
        "agent_id": "test_agent_action",
        "learning_type": "reinforcement",
        "model_config": {
            "state_size": 3,
            "action_size": 2
        }
    }
    test_client.post("/agents/register", json=agent_config)
    
    # Get action for state
    request = {
        "agent_id": "test_agent_action",
        "state": [0.1, 0.2, 0.3]
    }
    
    response = test_client.post("/action", json=request)
    assert response.status_code == 200
    data = response.json()
    assert "action" in data
    assert isinstance(data["action"], int)
    assert 0 <= data["action"] < 2

def test_train_step(test_client):
    # First register an agent
    agent_config = {
        "agent_id": "test_agent_train",
        "learning_type": "reinforcement",
        "model_config": {
            "state_size": 3,
            "action_size": 2
        }
    }
    test_client.post("/agents/register", json=agent_config)
    
    # Submit multiple experiences
    experiences = [
        {
            "agent_id": "test_agent_train",
            "state": [0.1, 0.2, 0.3],
            "action": 1,
            "reward": 1.0,
            "next_state": [0.2, 0.3, 0.4],
            "done": False
        }
        for _ in range(5)
    ]
    
    for exp in experiences:
        test_client.post("/experience", json=exp)
    
    # Trigger training step
    train_request = {
        "agent_id": "test_agent_train",
        "batch_size": 4
    }
    
    response = test_client.post("/train", json=train_request)
    assert response.status_code == 200
    data = response.json()
    assert "loss" in data
    assert "metrics" in data

def test_meta_learning(test_client):
    meta_config = {
        "agent_id": "test_meta_agent",
        "meta_learning_type": "maml",
        "task_config": {
            "n_tasks": 3,
            "n_samples_per_task": 10
        },
        "model_config": {
            "state_size": 5,
            "action_size": 2
        }
    }
    
    response = test_client.post("/meta/register", json=meta_config)
    assert response.status_code == 200
    data = response.json()
    assert "meta_learner_id" in data

def test_curriculum_learning(test_client):
    curriculum_config = {
        "agent_id": "test_curriculum_agent",
        "curriculum_type": "progressive",
        "stages": [
            {
                "difficulty": 1,
                "requirements": {"success_rate": 0.8}
            },
            {
                "difficulty": 2,
                "requirements": {"success_rate": 0.7}
            }
        ]
    }
    
    response = test_client.post("/curriculum/register", json=curriculum_config)
    assert response.status_code == 200
    data = response.json()
    assert "curriculum_id" in data

def test_coalition_learning(test_client):
    coalition_config = {
        "coalition_id": "test_coalition",
        "agents": ["agent1", "agent2"],
        "cooperation_type": "knowledge_sharing",
        "sharing_frequency": 100
    }
    
    response = test_client.post("/coalition/register", json=coalition_config)
    assert response.status_code == 200
    data = response.json()
    assert "coalition_id" in data

def test_get_agent_performance(test_client):
    # First register an agent
    agent_config = {
        "agent_id": "test_agent_perf",
        "learning_type": "reinforcement",
        "model_config": {
            "state_size": 3,
            "action_size": 2
        }
    }
    test_client.post("/agents/register", json=agent_config)
    
    response = test_client.get("/agents/test_agent_perf/performance")
    assert response.status_code == 200
    data = response.json()
    
    assert "metrics" in data
    assert "training_episodes" in data
    assert "average_reward" in data

def test_save_agent_model(test_client):
    # First register an agent
    agent_config = {
        "agent_id": "test_agent_save",
        "learning_type": "reinforcement",
        "model_config": {
            "state_size": 3,
            "action_size": 2
        }
    }
    test_client.post("/agents/register", json=agent_config)
    
    response = test_client.post("/agents/test_agent_save/save")
    assert response.status_code == 200
    data = response.json()
    assert "model_path" in data

def test_load_agent_model(test_client):
    # First register and save an agent
    agent_config = {
        "agent_id": "test_agent_load",
        "learning_type": "reinforcement",
        "model_config": {
            "state_size": 3,
            "action_size": 2
        }
    }
    test_client.post("/agents/register", json=agent_config)
    save_response = test_client.post("/agents/test_agent_load/save")
    model_path = save_response.json()["model_path"]
    
    # Load the model
    load_request = {
        "model_path": model_path
    }
    response = test_client.post("/agents/test_agent_load/load", json=load_request)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"

def test_learning_config(test_client):
    config_data = {
        "default_learning_rate": 0.001,
        "default_batch_size": 32,
        "experience_buffer_size": 10000,
        "training_frequency": 4
    }
    
    response = test_client.post("/config", json=config_data)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"

def test_get_learning_stats(test_client):
    response = test_client.get("/stats")
    assert response.status_code == 200
    data = response.json()
    
    assert "total_agents" in data
    assert "total_experiences" in data
    assert "training_metrics" in data

def test_health_check(test_client):
    response = test_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    
    assert data["status"] == "healthy"
    assert "gpu_status" in data
    assert "memory_usage" in data 