import pytest
import os
import docker
from unittest.mock import Mock
from fastapi.testclient import TestClient

from adaptation.adaptation_service import app
from adaptation.config import TestConfig

@pytest.fixture(scope="session")
def test_config():
    """Provides test configuration settings."""
    return TestConfig()

@pytest.fixture(scope="session")
def test_client():
    """Creates a FastAPI test client."""
    return TestClient(app)

@pytest.fixture(scope="session")
def mock_docker_client():
    """Creates a mock Docker client for testing."""
    client = Mock(spec=docker.DockerClient)
    
    # Mock container methods
    container = Mock()
    container.id = "test_container_id"
    container.name = "test_container"
    container.status = "running"
    container.attrs = {
        "State": {
            "Status": "running",
            "Running": True,
            "Pid": 1234
        },
        "HostConfig": {
            "NanoCpus": 1000000000,  # 1 CPU
            "Memory": 1073741824  # 1GB
        },
        "NetworkSettings": {
            "Networks": {
                "bridge": {
                    "IPAddress": "172.17.0.2"
                }
            }
        }
    }
    
    # Set up container operations
    container.start = Mock(return_value=None)
    container.stop = Mock(return_value=None)
    container.remove = Mock(return_value=None)
    container.update = Mock(return_value=None)
    container.exec_run = Mock(return_value=(0, b"test output"))
    
    # Set up client operations
    client.containers = Mock()
    client.containers.get = Mock(return_value=container)
    client.containers.list = Mock(return_value=[container])
    client.containers.run = Mock(return_value=container)
    
    return client

@pytest.fixture(scope="session")
def test_data_dir(tmp_path_factory):
    """Creates a temporary directory for test data."""
    test_dir = tmp_path_factory.mktemp("test_data")
    return test_dir

@pytest.fixture(scope="function")
def cleanup_test_data(test_data_dir):
    """Cleans up test data after each test."""
    yield
    for file in test_data_dir.iterdir():
        if file.is_file():
            file.unlink()

@pytest.fixture(scope="session")
def sample_resource_metrics():
    """Provides sample resource metrics data."""
    return {
        "cpu_usage": 45.5,
        "memory_usage": 1024 * 1024 * 512,  # 512MB
        "disk_usage": 1024 * 1024 * 1024 * 10,  # 10GB
        "network_usage": 1024 * 1024  # 1MB
    }

@pytest.fixture(scope="session")
def sample_agent_config():
    """Provides sample agent configuration."""
    return {
        "name": "test_agent",
        "image": "test_image:latest",
        "cpu_limit": 1.0,
        "memory_limit": "1G",
        "environment": {
            "ENV_VAR1": "value1",
            "ENV_VAR2": "value2"
        },
        "volumes": {
            "/host/path": {"bind": "/container/path", "mode": "rw"}
        }
    }

@pytest.fixture(scope="session")
def sample_architecture_config():
    """Provides sample architecture configuration."""
    return {
        "components": ["comp1", "comp2", "comp3"],
        "connections": [
            ("comp1", "comp2"),
            ("comp2", "comp3")
        ],
        "resources": {
            "comp1": {"cpu": 0.5, "memory": "512M"},
            "comp2": {"cpu": 1.0, "memory": "1G"},
            "comp3": {"cpu": 0.7, "memory": "768M"}
        }
    }

@pytest.fixture(scope="session")
def sample_adaptation_rule():
    """Provides sample adaptation rule."""
    return {
        "condition": "high_load",
        "action": "scale_up",
        "target_component": "comp2",
        "parameters": {
            "scale_factor": 1.5,
            "max_instances": 3
        }
    }

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers",
        "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers",
        "slow: mark test as slow running"
    )

def pytest_collection_modifyitems(config, items):
    """Modify test collection based on markers and conditions."""
    if not config.getoption("--run-integration"):
        skip_integration = pytest.mark.skip(reason="need --run-integration option to run")
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(skip_integration)
    
    if not config.getoption("--run-slow"):
        skip_slow = pytest.mark.skip(reason="need --run-slow option to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)

def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="run integration tests"
    )
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="run slow tests"
    )

@pytest.fixture
def resource_manager():
    from adaptation.adaptation_service import ResourceManager
    return ResourceManager()

@pytest.fixture
def dynamic_scaler():
    from adaptation.adaptation_service import DynamicScaler
    return DynamicScaler()

@pytest.fixture
def load_balancer():
    from adaptation.adaptation_service import LoadBalancer
    return LoadBalancer()

@pytest.fixture
def self_modifier():
    from adaptation.adaptation_service import SelfModifier
    return SelfModifier()

@pytest.fixture
def test_tasks():
    return ["task1", "task2", "task3", "task4", "task5"]

@pytest.fixture
def test_configuration():
    return {
        "param1": "value1",
        "param2": "value2",
        "param3": "value3"
    } 