import pytest
from fastapi.testclient import TestClient
from learning.learning_service import app, CurriculumManager, TransferLearningManager, MetaLearner
from learning.config import LEARNING_RATE, BATCH_SIZE, EPOCHS

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_curriculum_manager():
    manager = CurriculumManager()
    task = {
        "type": "classification",
        "difficulty": "easy",
        "data": "test_data"
    }
    
    # Test adding task
    manager.add_task(level=1, task=task)
    assert len(manager.get_tasks(level=1)) == 1
    
    # Test task completion
    manager.mark_task_complete(level=1, task_id=0)
    assert manager.get_task_status(level=1, task_id=0) == "completed"

def test_transfer_learning():
    manager = TransferLearningManager()
    source_domain = "domain1"
    target_domain = "domain2"
    knowledge = {"weights": [0.1, 0.2, 0.3]}
    
    # Test storing knowledge
    manager.store_knowledge(source_domain, knowledge)
    assert manager.has_knowledge(source_domain)
    
    # Test transferring knowledge
    transferred = manager.transfer_knowledge(source_domain, target_domain)
    assert transferred is not None
    assert "weights" in transferred

def test_meta_learning():
    learner = MetaLearner()
    task = {
        "type": "regression",
        "data": "test_data"
    }
    
    # Test meta-learning update
    initial_params = learner.get_parameters()
    learner.update_parameters(task)
    updated_params = learner.get_parameters()
    assert initial_params != updated_params

def test_curriculum_endpoint():
    response = client.post(
        "/curriculum/add_task",
        json={
            "level": 1,
            "task": {
                "type": "classification",
                "difficulty": "easy"
            }
        }
    )
    assert response.status_code == 200
    assert "task_id" in response.json()

def test_transfer_endpoint():
    response = client.post(
        "/transfer/store_knowledge",
        json={
            "domain": "source_domain",
            "knowledge": {"weights": [0.1, 0.2, 0.3]}
        }
    )
    assert response.status_code == 200
    assert "success" in response.json()

def test_meta_endpoint():
    response = client.post(
        "/meta/update_parameters",
        json={
            "task": {
                "type": "regression",
                "data": "test_data"
            }
        }
    )
    assert response.status_code == 200
    assert "parameters" in response.json()

def test_performance_metrics():
    response = client.get("/metrics")
    assert response.status_code == 200
    metrics = response.json()
    assert "accuracy" in metrics
    assert "loss" in metrics
    assert "learning_speed" in metrics

def test_invalid_task():
    response = client.post(
        "/curriculum/add_task",
        json={
            "level": -1,  # Invalid level
            "task": {
                "type": "invalid_type",
                "difficulty": "invalid"
            }
        }
    )
    assert response.status_code == 400

def test_resource_limits():
    # Test with very large data to check resource limits
    large_data = "x" * 1000000  # 1MB of data
    response = client.post(
        "/transfer/store_knowledge",
        json={
            "domain": "test_domain",
            "knowledge": {"data": large_data}
        }
    )
    assert response.status_code == 413  # Payload Too Large 