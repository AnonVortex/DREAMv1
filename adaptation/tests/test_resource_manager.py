import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import numpy as np

from ..adaptation_service import ResourceManager, ResourceMetrics

@pytest.fixture
def resource_manager():
    return ResourceManager()

@pytest.fixture
def sample_metrics():
    return ResourceMetrics(
        timestamp=datetime.now(),
        cpu_usage=45.5,
        memory_usage=60.2,
        network_io={'rx_bytes': 1500, 'tx_bytes': 2500},
        agent_id="test_agent"
    )

@pytest.fixture
def historical_metrics():
    base_time = datetime.now()
    return [
        ResourceMetrics(
            timestamp=base_time - timedelta(minutes=i),
            cpu_usage=40.0 + i,
            memory_usage=55.0 + i,
            network_io={'rx_bytes': 1000 + i*100, 'tx_bytes': 2000 + i*100},
            agent_id="test_agent"
        )
        for i in range(10)
    ]

def test_collect_metrics(resource_manager):
    with patch('docker.from_env') as mock_docker:
        # Mock container stats
        mock_stats = {
            'cpu_stats': {
                'cpu_usage': {'total_usage': 100000},
                'system_cpu_usage': 1000000
            },
            'memory_stats': {
                'usage': 1024 * 1024 * 100,  # 100MB
                'limit': 1024 * 1024 * 1000  # 1GB
            },
            'networks': {
                'eth0': {
                    'rx_bytes': 1500,
                    'tx_bytes': 2500
                }
            }
        }
        
        mock_container = Mock()
        mock_container.stats.return_value = mock_stats
        mock_docker.return_value.containers.get.return_value = mock_container
        
        metrics = resource_manager.collect_metrics("test_agent")
        
        assert isinstance(metrics, ResourceMetrics)
        assert metrics.agent_id == "test_agent"
        assert 0 <= metrics.cpu_usage <= 100
        assert metrics.memory_usage > 0
        assert all(key in metrics.network_io for key in ['rx_bytes', 'tx_bytes'])

def test_calculate_resource_usage(resource_manager, sample_metrics):
    usage = resource_manager.calculate_resource_usage(sample_metrics)
    
    assert isinstance(usage, float)
    assert 0 <= usage <= 1
    
    # Test with extreme values
    high_metrics = ResourceMetrics(
        timestamp=datetime.now(),
        cpu_usage=95.0,
        memory_usage=90.0,
        network_io={'rx_bytes': 10000, 'tx_bytes': 10000},
        agent_id="test_agent"
    )
    high_usage = resource_manager.calculate_resource_usage(high_metrics)
    assert high_usage > usage  # Higher metrics should result in higher usage

def test_detect_anomalies(resource_manager, historical_metrics):
    # Test normal metrics
    normal_metrics = ResourceMetrics(
        timestamp=datetime.now(),
        cpu_usage=45.0,
        memory_usage=60.0,
        network_io={'rx_bytes': 1500, 'tx_bytes': 2500},
        agent_id="test_agent"
    )
    
    is_anomaly = resource_manager.detect_anomalies(normal_metrics, historical_metrics)
    assert not is_anomaly
    
    # Test anomalous metrics
    anomalous_metrics = ResourceMetrics(
        timestamp=datetime.now(),
        cpu_usage=95.0,
        memory_usage=95.0,
        network_io={'rx_bytes': 10000, 'tx_bytes': 10000},
        agent_id="test_agent"
    )
    
    is_anomaly = resource_manager.detect_anomalies(anomalous_metrics, historical_metrics)
    assert is_anomaly

def test_predict_resource_trends(resource_manager, historical_metrics):
    predictions = resource_manager.predict_resource_trends(historical_metrics)
    
    assert isinstance(predictions, dict)
    assert all(key in predictions for key in ['cpu_trend', 'memory_trend', 'network_trend'])
    assert all(isinstance(trend, float) for trend in predictions.values())

def test_resource_threshold_alerts(resource_manager, sample_metrics):
    # Test normal metrics
    alerts = resource_manager.check_resource_thresholds(sample_metrics)
    assert not alerts  # No alerts for normal metrics
    
    # Test metrics exceeding thresholds
    high_metrics = ResourceMetrics(
        timestamp=datetime.now(),
        cpu_usage=95.0,
        memory_usage=90.0,
        network_io={'rx_bytes': 10000, 'tx_bytes': 10000},
        agent_id="test_agent"
    )
    
    alerts = resource_manager.check_resource_thresholds(high_metrics)
    assert alerts  # Should generate alerts
    assert any('CPU' in alert for alert in alerts)
    assert any('memory' in alert for alert in alerts)

def test_historical_data_management(resource_manager, historical_metrics):
    # Test adding historical data
    for metrics in historical_metrics:
        resource_manager.add_historical_metrics(metrics)
    
    stored_metrics = resource_manager.get_historical_metrics("test_agent")
    assert len(stored_metrics) == len(historical_metrics)
    
    # Test data retention policy
    old_metrics = ResourceMetrics(
        timestamp=datetime.now() - timedelta(days=8),
        cpu_usage=40.0,
        memory_usage=55.0,
        network_io={'rx_bytes': 1000, 'tx_bytes': 2000},
        agent_id="test_agent"
    )
    
    resource_manager.add_historical_metrics(old_metrics)
    updated_metrics = resource_manager.get_historical_metrics("test_agent")
    assert old_metrics not in updated_metrics  # Old data should be pruned

def test_resource_efficiency_score(resource_manager, sample_metrics):
    score = resource_manager.calculate_efficiency_score(sample_metrics)
    
    assert isinstance(score, float)
    assert 0 <= score <= 1
    
    # Test with different resource profiles
    efficient_metrics = ResourceMetrics(
        timestamp=datetime.now(),
        cpu_usage=30.0,
        memory_usage=40.0,
        network_io={'rx_bytes': 1000, 'tx_bytes': 2000},
        agent_id="test_agent"
    )
    
    inefficient_metrics = ResourceMetrics(
        timestamp=datetime.now(),
        cpu_usage=90.0,
        memory_usage=85.0,
        network_io={'rx_bytes': 5000, 'tx_bytes': 6000},
        agent_id="test_agent"
    )
    
    efficient_score = resource_manager.calculate_efficiency_score(efficient_metrics)
    inefficient_score = resource_manager.calculate_efficiency_score(inefficient_metrics)
    
    assert efficient_score > inefficient_score

def test_handle_missing_metrics(resource_manager):
    # Test with missing network metrics
    partial_metrics = ResourceMetrics(
        timestamp=datetime.now(),
        cpu_usage=45.5,
        memory_usage=60.2,
        network_io=None,
        agent_id="test_agent"
    )
    
    usage = resource_manager.calculate_resource_usage(partial_metrics)
    assert isinstance(usage, float)
    assert 0 <= usage <= 1

def test_concurrent_metrics_collection(resource_manager):
    with patch('docker.from_env') as mock_docker:
        # Mock multiple containers
        containers = ["agent1", "agent2", "agent3"]
        mock_docker.return_value.containers.list.return_value = [
            Mock(id=container_id) for container_id in containers
        ]
        
        # Collect metrics concurrently
        metrics_list = resource_manager.collect_metrics_concurrent(containers)
        
        assert len(metrics_list) == len(containers)
        assert all(isinstance(metrics, ResourceMetrics) for metrics in metrics_list)
        assert all(metrics.agent_id in containers for metrics in metrics_list)

def test_resource_optimization_recommendations(resource_manager, historical_metrics):
    recommendations = resource_manager.generate_optimization_recommendations(historical_metrics)
    
    assert isinstance(recommendations, list)
    assert len(recommendations) > 0
    assert all(isinstance(rec, str) for rec in recommendations)
    
    # Verify recommendations are relevant to resource usage
    relevant_terms = ['CPU', 'memory', 'network', 'resource', 'usage', 'optimization']
    assert any(any(term.lower() in rec.lower() for term in relevant_terms)
              for rec in recommendations) 