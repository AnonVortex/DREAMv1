import pytest
from datetime import datetime, timedelta
from typing import Dict, Any
from unittest.mock import Mock, patch, MagicMock
import docker

from ..adaptation_service import DynamicScaler, AgentConfig, ScalingPolicy, ScalingHistory, ResourceMetrics

@pytest.fixture
def mock_docker_client(mock_docker_client):
    # Add specific mocks for container operations
    mock_docker_client.containers.run = Mock(return_value=Mock(id="new_container_id"))
    mock_docker_client.containers.get = Mock(return_value=Mock(
        status="running",
        attrs={
            "State": {"Status": "running"},
            "NetworkSettings": {"IPAddress": "172.17.0.2"}
        }
    ))
    return mock_docker_client

@pytest.fixture
def dynamic_scaler(mock_docker_client):
    scaler = DynamicScaler()
    scaler.docker_client = mock_docker_client
    return scaler

@pytest.fixture
def sample_agent_metrics():
    return {
        "agent1": ResourceMetrics(
            timestamp=datetime.now(),
            cpu_usage=85.0,
            memory_usage=90.0,
            network_io={'rx_bytes': 1000, 'tx_bytes': 2000},
            agent_id="agent1"
        ),
        "agent2": ResourceMetrics(
            timestamp=datetime.now(),
            cpu_usage=30.0,
            memory_usage=40.0,
            network_io={'rx_bytes': 3000, 'tx_bytes': 4000},
            agent_id="agent2"
        )
    }

@pytest.fixture
def sample_agent_config():
    return AgentConfig(
        name="test_agent",
        image="test_image:latest",
        cpu_limit="1",
        memory_limit="1G",
        environment={
            "ENV_VAR": "test_value"
        },
        volumes={
            "/host/path": {"bind": "/container/path", "mode": "rw"}
        }
    )

def test_check_scaling_needs(dynamic_scaler, sample_agent_metrics):
    # Test scaling check with high load
    scale_up, scale_down = dynamic_scaler.check_scaling_needs(sample_agent_metrics)
    assert scale_up  # Should recommend scaling up due to high load on agent1
    assert not scale_down  # Should not recommend scaling down
    
    # Test with low load
    low_load_metrics = {
        "agent1": ResourceMetrics(
            timestamp=datetime.now(),
            cpu_usage=20.0,
            memory_usage=30.0,
            network_io={'rx_bytes': 1000, 'tx_bytes': 2000},
            agent_id="agent1"
        )
    }
    scale_up, scale_down = dynamic_scaler.check_scaling_needs(low_load_metrics)
    assert not scale_up  # Should not recommend scaling up
    assert scale_down  # Should recommend scaling down

def test_create_agent(dynamic_scaler, sample_agent_config, mock_docker_client):
    # Test successful agent creation
    container = dynamic_scaler.create_agent(sample_agent_config)
    
    assert container is not None
    assert container.id == "test_container_id"
    assert container.status == "running"
    
    # Verify Docker client was called with correct parameters
    mock_docker_client.return_value.containers.run.assert_called_once_with(
        image=sample_agent_config.image,
        name=sample_agent_config.name,
        cpu_quota=int(float(sample_agent_config.cpu_limit) * 100000),
        mem_limit=sample_agent_config.memory_limit,
        environment=sample_agent_config.environment,
        volumes=sample_agent_config.volumes,
        detach=True
    )

def test_create_agent_failure(dynamic_scaler, sample_agent_config, mock_docker_client):
    # Mock Docker client to raise an exception
    mock_docker_client.return_value.containers.run.side_effect = docker.errors.APIError("Test error")
    
    with pytest.raises(Exception) as exc_info:
        dynamic_scaler.create_agent(sample_agent_config)
    
    assert "Failed to create agent" in str(exc_info.value)

def test_scale_agent(dynamic_scaler, mock_docker_client):
    # Test scaling up
    mock_container = Mock()
    mock_container.update = Mock()
    mock_docker_client.return_value.containers.get.return_value = mock_container
    
    dynamic_scaler.scale_agent("test_agent", cpu="2", memory="2G")
    
    # Verify container update was called with correct parameters
    mock_container.update.assert_called_once_with(
        cpu_quota=200000,  # 2 CPUs * 100000
        mem_limit="2G"
    )

def test_scale_agent_not_found(dynamic_scaler, mock_docker_client):
    # Mock Docker client to raise NotFound error
    mock_docker_client.return_value.containers.get.side_effect = docker.errors.NotFound("Container not found")
    
    with pytest.raises(Exception) as exc_info:
        dynamic_scaler.scale_agent("nonexistent_agent", cpu="2", memory="2G")
    
    assert "Agent not found" in str(exc_info.value)

def test_get_agent_status(dynamic_scaler, mock_docker_client):
    # Mock container with status
    mock_container = Mock()
    mock_container.status = "running"
    mock_container.attrs = {
        "State": {
            "Status": "running",
            "StartedAt": "2024-01-01T00:00:00Z",
            "Health": {"Status": "healthy"}
        }
    }
    mock_docker_client.return_value.containers.get.return_value = mock_container
    
    status = dynamic_scaler.get_agent_status("test_agent")
    
    assert status["status"] == "running"
    assert "start_time" in status
    assert status["health"] == "healthy"

def test_stop_agent(dynamic_scaler, mock_docker_client):
    # Mock container
    mock_container = Mock()
    mock_container.stop = Mock()
    mock_docker_client.return_value.containers.get.return_value = mock_container
    
    dynamic_scaler.stop_agent("test_agent")
    
    # Verify container stop was called
    mock_container.stop.assert_called_once()

def test_restart_agent(dynamic_scaler, mock_docker_client):
    # Mock container
    mock_container = Mock()
    mock_container.restart = Mock()
    mock_docker_client.return_value.containers.get.return_value = mock_container
    
    dynamic_scaler.restart_agent("test_agent")
    
    # Verify container restart was called
    mock_container.restart.assert_called_once()

def test_list_agents(dynamic_scaler, mock_docker_client):
    # Mock container list
    mock_containers = [
        Mock(name="agent1", status="running"),
        Mock(name="agent2", status="running")
    ]
    mock_docker_client.return_value.containers.list.return_value = mock_containers
    
    agents = dynamic_scaler.list_agents()
    
    assert len(agents) == 2
    assert all(agent["status"] == "running" for agent in agents)
    assert {"agent1", "agent2"} == {agent["name"] for agent in agents}

def test_auto_scaling(dynamic_scaler, mock_docker_client):
    # Mock resource metrics
    metrics = ResourceMetrics(
        timestamp=datetime.now(),
        cpu_usage=90.0,  # High CPU usage
        memory_usage=85.0,  # High memory usage
        network_io={'rx_bytes': 1000, 'tx_bytes': 2000},
        agent_id="test_agent"
    )
    
    # Mock container
    mock_container = Mock()
    mock_container.update = Mock()
    mock_docker_client.return_value.containers.get.return_value = mock_container
    
    # Test auto-scaling based on metrics
    scaling_decision = dynamic_scaler.check_scaling_needs(metrics)
    
    assert scaling_decision["needs_scaling"]
    assert scaling_decision["cpu_scale"] > 1.0
    assert scaling_decision["memory_scale"] > 1.0

def test_agent_recovery(dynamic_scaler, mock_docker_client):
    # Mock failed container
    mock_container = Mock()
    mock_container.status = "exited"
    mock_container.restart = Mock()
    mock_docker_client.return_value.containers.get.return_value = mock_container
    
    # Test recovery of failed agent
    recovery_result = dynamic_scaler.recover_failed_agent("test_agent")
    
    assert recovery_result["recovered"]
    mock_container.restart.assert_called_once()

def test_batch_scaling(dynamic_scaler, mock_docker_client):
    # Mock multiple containers
    mock_containers = [
        Mock(name="agent1", update=Mock()),
        Mock(name="agent2", update=Mock())
    ]
    mock_docker_client.return_value.containers.list.return_value = mock_containers
    
    # Test batch scaling of multiple agents
    scaling_configs = [
        {"agent_id": "agent1", "cpu": "2", "memory": "2G"},
        {"agent_id": "agent2", "cpu": "1.5", "memory": "1.5G"}
    ]
    
    results = dynamic_scaler.batch_scale_agents(scaling_configs)
    
    assert len(results) == 2
    assert all(result["success"] for result in results)
    assert all(container.update.called for container in mock_containers)

def test_resource_optimization(dynamic_scaler, mock_docker_client):
    # Mock container with resource usage
    mock_container = Mock()
    mock_container.stats.return_value = {
        'cpu_stats': {'cpu_usage': {'total_usage': 50000}, 'system_cpu_usage': 100000},
        'memory_stats': {'usage': 512 * 1024 * 1024, 'limit': 1024 * 1024 * 1024}
    }
    mock_docker_client.return_value.containers.get.return_value = mock_container
    
    # Test resource optimization recommendations
    recommendations = dynamic_scaler.optimize_resources("test_agent")
    
    assert "cpu_recommendation" in recommendations
    assert "memory_recommendation" in recommendations
    assert isinstance(recommendations["cpu_recommendation"], float)
    assert isinstance(recommendations["memory_recommendation"], str)

def test_scale_agent_up(dynamic_scaler, sample_agent_config, sample_agent_metrics):
    # Test scaling up
    with patch.object(dynamic_scaler, 'create_agent') as mock_create:
        mock_create.return_value = "new_agent_id"
        new_agents = dynamic_scaler.scale_up(1, sample_agent_config)
        
        assert len(new_agents) == 1
        assert "new_agent_id" in new_agents
        mock_create.assert_called_once_with(sample_agent_config)

def test_scale_agent_down(dynamic_scaler, sample_agent_metrics):
    # Test scaling down
    agent_to_remove = "agent1"
    
    # Mock container object
    mock_container = Mock()
    dynamic_scaler.docker_client.containers.get.return_value = mock_container
    
    # Test scale down
    removed_agents = dynamic_scaler.scale_down([agent_to_remove])
    
    # Verify container stop and removal
    mock_container.stop.assert_called_once()
    mock_container.remove.assert_called_once()
    assert agent_to_remove in removed_agents

def test_handle_failed_scaling(dynamic_scaler, sample_agent_config):
    # Test handling of failed scaling operation
    dynamic_scaler.docker_client.containers.run.side_effect = Exception("Container creation failed")
    
    # Attempt to create agent
    with pytest.raises(Exception) as exc_info:
        dynamic_scaler.create_agent(sample_agent_config)
    
    assert "Container creation failed" in str(exc_info.value)

def test_agent_health_check(dynamic_scaler):
    # Test agent health checking
    healthy_agent = "healthy_agent"
    unhealthy_agent = "unhealthy_agent"
    
    # Mock container responses
    def get_container(agent_id):
        if agent_id == healthy_agent:
            return Mock(
                status="running",
                attrs={"State": {"Status": "running", "Health": {"Status": "healthy"}}}
            )
        else:
            return Mock(
                status="running",
                attrs={"State": {"Status": "running", "Health": {"Status": "unhealthy"}}}
            )
    
    dynamic_scaler.docker_client.containers.get.side_effect = get_container
    
    # Test health checks
    assert dynamic_scaler.check_agent_health(healthy_agent)
    assert not dynamic_scaler.check_agent_health(unhealthy_agent)

def test_resource_based_scaling(dynamic_scaler, sample_agent_metrics):
    # Test resource-based scaling decisions
    dynamic_scaler.cpu_threshold = 80.0
    dynamic_scaler.memory_threshold = 85.0
    
    # Test with high resource usage
    scale_up, scale_down = dynamic_scaler.check_scaling_needs(sample_agent_metrics)
    assert scale_up  # Should recommend scaling up due to high resource usage
    
    # Test with moderate resource usage
    moderate_metrics = {
        "agent1": ResourceMetrics(
            timestamp=datetime.now(),
            cpu_usage=60.0,
            memory_usage=70.0,
            network_io={'rx_bytes': 1000, 'tx_bytes': 2000},
            agent_id="agent1"
        )
    }
    scale_up, scale_down = dynamic_scaler.check_scaling_needs(moderate_metrics)
    assert not scale_up  # Should not recommend scaling
    assert not scale_down

def test_create_agent_with_gpu(dynamic_scaler, sample_agent_config):
    dynamic_scaler.gpu_available = True
    sample_agent_config.gpu_config = {"count": 1}
    
    container_id = dynamic_scaler.create_agent(sample_agent_config)
    assert container_id is not None
    
    # Verify GPU configuration was included
    container_config = dynamic_scaler.docker_client.containers.run.call_args[1]
    assert "device_requests" in container_config
    assert container_config["device_requests"][0]["Driver"] == "nvidia"

def test_scale_agent(dynamic_scaler, sample_agent_config):
    # Create an agent first
    container_id = dynamic_scaler.create_agent(sample_agent_config)
    
    # Test scaling up
    success = dynamic_scaler.scale_agent(
        agent_id=sample_agent_config.agent_id,
        scale_factor=1.5,
        reason="high_load"
    )
    assert success is True
    
    # Verify scaling history
    history = dynamic_scaler.get_scaling_history(agent_id=sample_agent_config.agent_id)
    assert len(history) == 2  # Create + Scale
    assert history[-1].action == "scale"
    assert history[-1].reason == "high_load"

def test_cooldown_period(dynamic_scaler, sample_agent_config):
    # Create an agent
    dynamic_scaler.create_agent(sample_agent_config)
    
    # First scale should succeed
    assert dynamic_scaler.scale_agent(
        agent_id=sample_agent_config.agent_id,
        scale_factor=1.5,
        reason="test"
    ) is True
    
    # Immediate rescale should fail due to cooldown
    assert dynamic_scaler.scale_agent(
        agent_id=sample_agent_config.agent_id,
        scale_factor=2.0,
        reason="test"
    ) is False

def test_get_agent_metrics(dynamic_scaler, sample_agent_config):
    # Create an agent
    container_id = dynamic_scaler.create_agent(sample_agent_config)
    
    # Mock container stats
    stats = {
        'cpu_stats': {
            'cpu_usage': {'total_usage': 100000},
            'system_cpu_usage': 1000000
        },
        'precpu_stats': {
            'cpu_usage': {'total_usage': 90000},
            'system_cpu_usage': 900000
        },
        'memory_stats': {
            'usage': 512000000,
            'limit': 1024000000
        },
        'networks': {
            'eth0': {
                'rx_bytes': 1000,
                'tx_bytes': 2000
            }
        }
    }
    
    with patch.object(dynamic_scaler.docker_client.containers, 'get') as mock_get:
        mock_container = Mock()
        mock_container.stats.return_value = stats
        mock_get.return_value = mock_container
        
        metrics = dynamic_scaler.get_agent_metrics(sample_agent_config.agent_id)
        
        assert 'cpu_usage' in metrics
        assert 'memory_usage' in metrics
        assert 'network_io' in metrics
        assert metrics['memory_usage'] == 50.0  # 512MB / 1024MB * 100

def test_scaling_history_management(dynamic_scaler):
    # Test history size limit
    for i in range(1100):  # More than the 1000 limit
        dynamic_scaler._record_scaling_history(
            agent_id=f"test_agent_{i}",
            action="test",
            old_config={},
            new_config={},
            reason="test",
            success=True
        )
    
    history = dynamic_scaler.get_scaling_history()
    assert len(history) == 1000  # Verify history was trimmed

def test_get_scaling_history_filters(dynamic_scaler):
    # Add some test history entries
    agent_id = "test_agent"
    now = datetime.now()
    
    for i in range(3):
        dynamic_scaler._record_scaling_history(
            agent_id=agent_id,
            action=f"action_{i}",
            old_config={},
            new_config={},
            reason="test",
            success=True
        )
    
    # Test filtering by time range
    start_time = now - timedelta(minutes=1)
    end_time = now + timedelta(minutes=1)
    
    filtered_history = dynamic_scaler.get_scaling_history(
        agent_id=agent_id,
        start_time=start_time,
        end_time=end_time
    )
    
    assert len(filtered_history) == 3
    assert all(h.agent_id == agent_id for h in filtered_history)
    assert all(start_time <= h.timestamp <= end_time for h in filtered_history)

def test_gpu_scaling(dynamic_scaler, sample_agent_config):
    dynamic_scaler.gpu_available = True
    sample_agent_config.gpu_config = {
        "count": 1,
        "Options": {"gpu-memory-limit": "4096"}
    }
    
    # Create agent with GPU
    container_id = dynamic_scaler.create_agent(sample_agent_config)
    
    # Mock container for GPU scaling
    mock_container = Mock()
    mock_container.attrs = {
        'HostConfig': {
            'DeviceRequests': [{
                'Driver': 'nvidia',
                'Count': 1,
                'Options': {'gpu-memory-limit': '4096'}
            }]
        }
    }
    
    with patch.object(dynamic_scaler.docker_client.containers, 'get', return_value=mock_container):
        # Test GPU scaling
        new_gpu_config = dynamic_scaler._calculate_gpu_config(
            agent_id=sample_agent_config.agent_id,
            scale_factor=1.5
        )
        
        assert new_gpu_config is not None
        assert new_gpu_config[0]['Driver'] == 'nvidia'
        assert new_gpu_config[0]['Options']['gpu-memory-limit'] == '6144'  # 4096 * 1.5 