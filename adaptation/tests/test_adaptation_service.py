import pytest
import docker
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
from datetime import datetime, timedelta
from fastapi import status

from adaptation.adaptation_service import (
    ResourceManager, DynamicScaler, LoadBalancer,
    SelfModifier, EvolutionaryArchitect,
    ResourceMetrics, AgentConfig, AdaptationRule,
    ArchitectureConfig, app
)

@pytest.fixture
def test_client():
    return TestClient(app)

@pytest.fixture
def mock_docker_client():
    with patch('docker.from_env') as mock_client:
        yield mock_client

class TestResourceManager:
    @pytest.fixture
    def resource_manager(self):
        return ResourceManager()

    def test_collect_metrics(self, resource_manager, mock_docker_client, sample_resource_metrics):
        """Test collecting resource metrics from containers."""
        metrics = resource_manager.collect_metrics()
        assert isinstance(metrics, dict)
        assert "cpu_usage" in metrics
        assert "memory_usage" in metrics
        assert "disk_usage" in metrics
        assert "network_usage" in metrics
        
        # Verify metrics are within expected ranges
        assert 0 <= metrics["cpu_usage"] <= 100
        assert metrics["memory_usage"] > 0
        assert metrics["disk_usage"] > 0
        assert metrics["network_usage"] >= 0

    def test_analyze_trends(self, resource_manager):
        """Test analyzing resource usage trends."""
        # Add sample historical data
        historical_data = [
            {"cpu_usage": 45.5, "memory_usage": 512 * 1024 * 1024},
            {"cpu_usage": 65.2, "memory_usage": 768 * 1024 * 1024},
            {"cpu_usage": 85.7, "memory_usage": 1024 * 1024 * 1024}
        ]
        
        trends = resource_manager.analyze_trends(historical_data)
        assert isinstance(trends, dict)
        assert "cpu_trend" in trends
        assert "memory_trend" in trends
        assert trends["cpu_trend"] > 0  # Increasing trend
        assert trends["memory_trend"] > 0  # Increasing trend

class TestDynamicScaler:
    @pytest.fixture
    def dynamic_scaler(self, mock_docker_client):
        return DynamicScaler(docker_client=mock_docker_client)

    def test_scale_agent(self, dynamic_scaler, mock_docker_client, sample_agent_config):
        """Test scaling an agent's resources."""
        # Test scaling up
        success = dynamic_scaler.scale_agent(
            agent_name="test_agent",
            scale_factor=1.5
        )
        assert success is True
        
        # Verify Docker client calls
        mock_docker_client.containers.get.assert_called_once_with("test_agent")
        mock_docker_client.containers.get().update.assert_called_once()

    def test_create_agent(self, dynamic_scaler, mock_docker_client, sample_agent_config):
        """Test creating a new agent."""
        agent_id = dynamic_scaler.create_agent(sample_agent_config)
        assert isinstance(agent_id, str)
        assert agent_id == "test_container_id"
        
        # Verify Docker client calls
        mock_docker_client.containers.run.assert_called_once()

class TestLoadBalancer:
    @pytest.fixture
    def load_balancer(self):
        return LoadBalancer()

    def test_distribute_load(self, load_balancer, sample_agent_config):
        """Test load distribution among agents."""
        agents = [
            {"name": "agent1", "load": 0.3},
            {"name": "agent2", "load": 0.7},
            {"name": "agent3", "load": 0.5}
        ]
        
        distribution = load_balancer.distribute_load(agents)
        assert isinstance(distribution, dict)
        assert len(distribution) == len(agents)
        assert all(0 <= load <= 1 for load in distribution.values())

    def test_calculate_task_compatibility(self, load_balancer):
        """Test calculating task compatibility with agents."""
        task = {"requirements": ["python", "tensorflow"]}
        agent = {"capabilities": ["python", "tensorflow", "pytorch"]}
        
        compatibility = load_balancer.calculate_task_compatibility(task, agent)
        assert isinstance(compatibility, float)
        assert 0 <= compatibility <= 1
        assert compatibility == 1.0  # Perfect match

class TestSelfModifier:
    @pytest.fixture
    def self_modifier(self):
        return SelfModifier()

    def test_modify_architecture(self, self_modifier, sample_architecture_config, sample_adaptation_rule):
        """Test modifying system architecture."""
        modified_config = self_modifier.modify_architecture(
            sample_architecture_config,
            sample_adaptation_rule
        )
        
        assert isinstance(modified_config, dict)
        assert "components" in modified_config
        assert "connections" in modified_config
        assert "resources" in modified_config
        
        # Verify the modification was applied
        target_comp = sample_adaptation_rule["target_component"]
        assert modified_config["resources"][target_comp]["cpu"] > sample_architecture_config["resources"][target_comp]["cpu"]

class TestEvolutionaryArchitect:
    @pytest.fixture
    def evolutionary_architect(self):
        return EvolutionaryArchitect()

    def test_evolve_architecture(self, evolutionary_architect, sample_architecture_config):
        """Test evolving system architecture."""
        constraints = {
            "max_components": 5,
            "max_connections": 6,
            "total_cpu": 4.0,
            "total_memory": "4G"
        }
        
        evolved_config = evolutionary_architect.evolve_architecture(
            sample_architecture_config,
            constraints
        )
        
        assert isinstance(evolved_config, dict)
        assert "components" in evolved_config
        assert "connections" in evolved_config
        assert "resources" in evolved_config
        
        # Verify constraints are met
        assert len(evolved_config["components"]) <= constraints["max_components"]
        assert len(evolved_config["connections"]) <= constraints["max_connections"]
        
        total_cpu = sum(res["cpu"] for res in evolved_config["resources"].values())
        assert total_cpu <= constraints["total_cpu"]

class TestAPIEndpoints:
    def test_collect_metrics_endpoint(self, test_client):
        """Test the metrics collection endpoint."""
        response = test_client.get("/metrics")
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert isinstance(data, dict)
        assert "cpu_usage" in data
        assert "memory_usage" in data

    def test_scale_agent_endpoint(self, test_client, sample_agent_config):
        """Test the agent scaling endpoint."""
        response = test_client.post(
            "/scale",
            json={
                "agent_name": "test_agent",
                "scale_factor": 1.5
            }
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert isinstance(data, dict)
        assert data["success"] is True

    def test_balance_load_endpoint(self, test_client):
        """Test the load balancing endpoint."""
        agents = [
            {"name": "agent1", "load": 0.3},
            {"name": "agent2", "load": 0.7}
        ]
        response = test_client.post("/balance", json={"agents": agents})
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert isinstance(data, dict)
        assert len(data) == len(agents)

    def test_modify_architecture_endpoint(self, test_client, sample_architecture_config, sample_adaptation_rule):
        """Test the architecture modification endpoint."""
        response = test_client.post(
            "/modify",
            json={
                "architecture": sample_architecture_config,
                "rule": sample_adaptation_rule
            }
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert isinstance(data, dict)
        assert "components" in data
        assert "connections" in data

    def test_evolve_architecture_endpoint(self, test_client, sample_architecture_config):
        """Test the architecture evolution endpoint."""
        constraints = {
            "max_components": 5,
            "max_connections": 6,
            "total_cpu": 4.0,
            "total_memory": "4G"
        }
        response = test_client.post(
            "/evolve",
            json={
                "architecture": sample_architecture_config,
                "constraints": constraints
            }
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert isinstance(data, dict)
        assert "components" in data
        assert "connections" in data 