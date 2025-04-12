import pytest
from fastapi.testclient import TestClient
from reasoning.reasoning_service import app, KnowledgeGraphManager, SymbolicReasoner, CausalReasoner, CommonSenseReasoner
from reasoning.config import KNOWLEDGE_GRAPH_PATH, RULE_BASE_PATH

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_knowledge_graph():
    manager = KnowledgeGraphManager()
    
    # Test adding node
    node_id = manager.add_node("concept1", {"type": "entity", "properties": {"color": "red"}})
    assert node_id is not None
    
    # Test adding edge
    edge_id = manager.add_edge("concept1", "concept2", "related_to")
    assert edge_id is not None
    
    # Test querying
    result = manager.query("MATCH (n) RETURN n")
    assert len(result) > 0

def test_symbolic_reasoning():
    reasoner = SymbolicReasoner()
    
    # Test adding rule
    rule = {
        "premise": ["A", "B"],
        "conclusion": "C",
        "confidence": 0.9
    }
    rule_id = reasoner.add_rule(rule)
    assert rule_id is not None
    
    # Test inference
    result = reasoner.infer(["A", "B"])
    assert result == "C"

def test_causal_reasoning():
    reasoner = CausalReasoner()
    
    # Test adding causal relation
    relation = {
        "cause": "A",
        "effect": "B",
        "strength": 0.8
    }
    relation_id = reasoner.add_relation(relation)
    assert relation_id is not None
    
    # Test causal inference
    result = reasoner.infer_effects("A")
    assert "B" in result

def test_common_sense():
    reasoner = CommonSenseReasoner()
    
    # Test adding rule
    rule = {
        "context": "general",
        "premise": "if raining",
        "conclusion": "then wet",
        "confidence": 0.95
    }
    rule_id = reasoner.add_rule(rule)
    assert rule_id is not None
    
    # Test inference
    result = reasoner.infer("if raining")
    assert result == "then wet"

def test_knowledge_endpoint():
    response = client.post(
        "/knowledge/add_node",
        json={
            "label": "test_node",
            "properties": {"type": "test"}
        }
    )
    assert response.status_code == 200
    assert "node_id" in response.json()

def test_symbolic_endpoint():
    response = client.post(
        "/symbolic/add_rule",
        json={
            "premise": ["A", "B"],
            "conclusion": "C",
            "confidence": 0.9
        }
    )
    assert response.status_code == 200
    assert "rule_id" in response.json()

def test_causal_endpoint():
    response = client.post(
        "/causal/add_relation",
        json={
            "cause": "A",
            "effect": "B",
            "strength": 0.8
        }
    )
    assert response.status_code == 200
    assert "relation_id" in response.json()

def test_common_sense_endpoint():
    response = client.post(
        "/common_sense/add_rule",
        json={
            "context": "general",
            "premise": "if raining",
            "conclusion": "then wet",
            "confidence": 0.95
        }
    )
    assert response.status_code == 200
    assert "rule_id" in response.json()

def test_query_endpoint():
    response = client.post(
        "/reason/query",
        json={
            "query": "MATCH (n) RETURN n"
        }
    )
    assert response.status_code == 200
    assert "results" in response.json()

def test_invalid_query():
    response = client.post(
        "/reason/query",
        json={
            "query": "INVALID QUERY"
        }
    )
    assert response.status_code == 400

def test_performance_metrics():
    response = client.get("/metrics")
    assert response.status_code == 200
    metrics = response.json()
    assert "inference_time" in metrics
    assert "memory_usage" in metrics
    assert "cache_hits" in metrics 