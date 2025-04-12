"""Unit tests for the LearningAgent class."""

import pytest
import numpy as np
from datetime import datetime
from uuid import uuid4
from hmas.agents.learning_agent import LearningAgent

@pytest.fixture
def learning_agent():
    """Create a basic learning agent for testing."""
    return LearningAgent(
        name="Test Learning Agent",
        learning_types=["reinforcement", "supervised", "unsupervised", "meta"],
        memory_size=1000,
        learning_rate=0.01,
        exploration_rate=0.1
    )

@pytest.mark.asyncio
async def test_initialization(learning_agent):
    """Test proper initialization of LearningAgent."""
    assert learning_agent.name == "Test Learning Agent"
    assert learning_agent.learning_types == ["reinforcement", "supervised", "unsupervised", "meta"]
    assert learning_agent.learning_rate == 0.01
    assert learning_agent.exploration_rate == 0.1
    assert "learning" in learning_agent.capabilities
    
    # Test state initialization
    assert all(ltype in learning_agent.state["models"] for ltype in learning_agent.learning_types)
    assert all(ltype in learning_agent.state["performance_history"] for ltype in learning_agent.learning_types)
    assert "episodes" in learning_agent.state["learning_progress"]
    assert "strategies" in learning_agent.state["meta_learning"]

@pytest.mark.asyncio
async def test_initialize():
    """Test agent initialization process."""
    agent = LearningAgent("Init Test Agent")
    success = await agent.initialize()
    assert success is True
    
    # Verify initialization of learning components
    for learning_type in agent.learning_types:
        assert learning_type in agent.state["meta_learning"]["strategies"]
        assert learning_type in agent.state["meta_learning"]["effectiveness"]
        assert learning_type in agent.state["learning_progress"]["improvements"]
        assert learning_type in agent.state["learning_progress"]["adaptation_rate"]

@pytest.mark.asyncio
async def test_process_reinforcement_learning(learning_agent):
    """Test reinforcement learning processing."""
    input_data = {
        "type": "reinforcement_learning",
        "data": {
            "state": np.array([1.0, 0.0, 0.0]),
            "action": 1,
            "reward": 1.0,
            "next_state": np.array([0.0, 1.0, 0.0])
        },
        "context": {}
    }
    
    result = await learning_agent.process(input_data)
    assert result is not None
    assert "next_action" in result
    assert "model_update" in result
    assert "exploration_rate" in result

@pytest.mark.asyncio
async def test_process_supervised_learning(learning_agent):
    """Test supervised learning processing."""
    input_data = {
        "type": "supervised_learning",
        "data": {
            "inputs": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            "targets": np.array([0, 1])
        },
        "context": {}
    }
    
    result = await learning_agent.process(input_data)
    assert result is not None
    assert "loss" in result
    assert "performance" in result
    assert "model_update" in result

@pytest.mark.asyncio
async def test_process_unsupervised_learning(learning_agent):
    """Test unsupervised learning processing."""
    input_data = {
        "type": "unsupervised_learning",
        "data": {
            "inputs": np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
        },
        "context": {}
    }
    
    result = await learning_agent.process(input_data)
    assert result is not None
    assert "patterns" in result
    assert "quality_metrics" in result
    assert "model_update" in result

@pytest.mark.asyncio
async def test_process_meta_learning(learning_agent):
    """Test meta-learning processing."""
    input_data = {
        "type": "meta_learning",
        "data": {
            "learning_type": "supervised",
            "performance_data": [
                {"performance": 0.8, "timestamp": datetime.now()},
                {"performance": 0.85, "timestamp": datetime.now()},
                {"performance": 0.9, "timestamp": datetime.now()}
            ]
        },
        "context": {}
    }
    
    result = await learning_agent.process(input_data)
    assert result is not None
    assert "analysis" in result
    assert "new_strategy" in result
    assert "adaptation" in result

@pytest.mark.asyncio
async def test_communicate(learning_agent):
    """Test communication with other agents."""
    target_id = uuid4()
    message = {
        "type": "learning_update",
        "content": {"test": "data"}
    }
    
    success = await learning_agent.communicate(message, target_id)
    assert success is True

@pytest.mark.asyncio
async def test_learn(learning_agent):
    """Test learning from experience."""
    experience = {
        "feedback": {
            "type": "supervised",
            "performance": 0.85
        }
    }
    
    success = await learning_agent.learn(experience)
    assert success is True
    assert len(learning_agent.state["performance_history"]["supervised"]) > 0
    assert learning_agent.state["learning_progress"]["episodes"] > 0

@pytest.mark.asyncio
async def test_reflect(learning_agent):
    """Test agent self-reflection."""
    # Add some learning history
    await learning_agent.learn({
        "feedback": {
            "type": "supervised",
            "performance": 0.85
        }
    })
    
    reflection = await learning_agent.reflect()
    assert reflection is not None
    assert "status" in reflection
    assert "learning_types" in reflection
    assert "performance_summary" in reflection
    assert "meta_learning" in reflection

def test_analyze_learning_performance(learning_agent):
    """Test learning performance analysis."""
    performance_data = [
        {"performance": 0.8, "timestamp": datetime.now()},
        {"performance": 0.85, "timestamp": datetime.now()},
        {"performance": 0.9, "timestamp": datetime.now()}
    ]
    
    analysis = learning_agent._analyze_learning_performance("supervised", performance_data)
    assert "trend" in analysis
    assert "improvement_rate" in analysis
    assert "stability" in analysis
    assert analysis["trend"] in ["improving", "declining"]

def test_adapt_learning_strategy(learning_agent):
    """Test learning strategy adaptation."""
    analysis = {
        "trend": "improving",
        "improvement_rate": 0.05,
        "stability": 0.02
    }
    
    strategy = learning_agent._adapt_learning_strategy("supervised", analysis)
    assert "name" in strategy
    assert "exploration_rate" in strategy
    assert "learning_rate" in strategy
    assert strategy["name"] in ["default", "adaptive", "exploratory"] 