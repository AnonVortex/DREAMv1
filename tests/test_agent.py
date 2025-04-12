"""Tests for the Agent class."""

import pytest
from uuid import UUID
from datetime import datetime

from hmas.core.agent import Agent, AgentState

@pytest.fixture
def test_agent():
    """Create a test agent."""
    return Agent(
        name="test_agent",
        capabilities=["test", "mock"],
        memory_size=100
    )

class TestAgent:
    """Test suite for Agent class."""
    
    def test_agent_initialization(self, test_agent):
        """Test agent initialization."""
        assert isinstance(test_agent.id, UUID)
        assert test_agent.name == "test_agent"
        assert test_agent.capabilities == ["test", "mock"]
        assert test_agent.memory_size == 100
        assert isinstance(test_agent.state, AgentState)
        assert test_agent.state.is_active is True
        
    def test_agent_has_capability(self, test_agent):
        """Test capability checking."""
        assert test_agent.has_capability("test") is True
        assert test_agent.has_capability("mock") is True
        assert test_agent.has_capability("unknown") is False
        
    @pytest.mark.asyncio
    async def test_agent_lifecycle(self, test_agent):
        """Test agent initialization and shutdown."""
        # Test initialization
        await test_agent.initialize()
        assert test_agent.is_active is True
        assert test_agent.state.last_action == "initialization"
        assert isinstance(test_agent.state.last_action_time, datetime)
        
        # Test shutdown
        await test_agent.shutdown()
        assert test_agent.is_active is False
        assert test_agent.state.last_action == "shutdown"
        
    @pytest.mark.asyncio
    async def test_agent_state_update(self, test_agent):
        """Test agent state updates."""
        metrics = {
            "cpu_usage": 50.0,
            "memory_usage": 75.0
        }
        
        await test_agent.update_state(metrics)
        assert test_agent.state.performance_metrics == metrics
        assert test_agent.state.last_action == "state_update"
        
    @pytest.mark.asyncio
    async def test_process_message_not_implemented(self, test_agent):
        """Test process_message raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            await test_agent.process_message({"test": "message"})
            
    def test_agent_representation(self, test_agent):
        """Test agent string representation."""
        repr_str = repr(test_agent)
        assert "Agent" in repr_str
        assert test_agent.name in repr_str
        assert str(test_agent.id) in repr_str
        
class TestAgentState:
    """Test suite for AgentState class."""
    
    def test_agent_state_initialization(self):
        """Test AgentState initialization."""
        state = AgentState()
        assert state.is_active is True
        assert state.memory_usage == 0
        assert state.last_action is None
        assert state.last_action_time is None
        assert state.performance_metrics == {}
        assert state.error_count == 0
        
    def test_agent_state_update(self):
        """Test AgentState updates."""
        state = AgentState()
        
        # Update state
        state.is_active = False
        state.memory_usage = 100
        state.last_action = "test"
        state.last_action_time = datetime.utcnow()
        state.error_count = 1
        
        assert state.is_active is False
        assert state.memory_usage == 100
        assert state.last_action == "test"
        assert isinstance(state.last_action_time, datetime)
        assert state.error_count == 1 