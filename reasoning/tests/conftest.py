import pytest
from fastapi.testclient import TestClient
from reasoning.reasoning_service import app
from reasoning.config import KNOWLEDGE_GRAPH_PATH, RULE_BASE_PATH

@pytest.fixture
def client():
    return TestClient(app)

@pytest.fixture
def knowledge_graph_manager():
    from reasoning.reasoning_service import KnowledgeGraphManager
    return KnowledgeGraphManager()

@pytest.fixture
def symbolic_reasoner():
    from reasoning.reasoning_service import SymbolicReasoner
    return SymbolicReasoner()

@pytest.fixture
def causal_reasoner():
    from reasoning.reasoning_service import CausalReasoner
    return CausalReasoner()

@pytest.fixture
def common_sense_reasoner():
    from reasoning.reasoning_service import CommonSenseReasoner
    return CommonSenseReasoner()

@pytest.fixture
def test_node():
    return {
        "label": "test_node",
        "properties": {
            "type": "entity",
            "color": "red"
        }
    }

@pytest.fixture
def test_rule():
    return {
        "premise": ["A", "B"],
        "conclusion": "C",
        "confidence": 0.9
    } 