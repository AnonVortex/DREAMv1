import pytest
from fastapi.testclient import TestClient
from reasoning.reasoning_service import app, ReasoningManager
import json
from datetime import datetime

@pytest.fixture
def test_client():
    return TestClient(app)

@pytest.fixture
def reasoning_manager():
    return ReasoningManager()

def test_add_fact(test_client):
    fact = {
        "type": "fact",
        "subject": "Earth",
        "predicate": "is",
        "object": "planet",
        "confidence": 1.0,
        "source": "astronomy",
        "timestamp": datetime.now().isoformat()
    }
    
    response = test_client.post("/knowledge/facts", json=fact)
    assert response.status_code == 200
    data = response.json()
    assert "fact_id" in data
    assert data["status"] == "added"

def test_add_rule(test_client):
    rule = {
        "type": "rule",
        "if_clause": [
            {
                "subject": "?x",
                "predicate": "is",
                "object": "mammal"
            }
        ],
        "then_clause": [
            {
                "subject": "?x",
                "predicate": "is",
                "object": "animal"
            }
        ],
        "confidence": 1.0,
        "source": "biology"
    }
    
    response = test_client.post("/knowledge/rules", json=rule)
    assert response.status_code == 200
    data = response.json()
    assert "rule_id" in data
    assert data["status"] == "added"

def test_add_concept(test_client):
    concept = {
        "type": "concept",
        "name": "Mammal",
        "properties": [
            "has_fur",
            "warm_blooded",
            "produces_milk"
        ],
        "relationships": [
            {
                "type": "is_a",
                "target": "Animal"
            }
        ],
        "examples": [
            "dog",
            "cat",
            "whale"
        ]
    }
    
    response = test_client.post("/knowledge/concepts", json=concept)
    assert response.status_code == 200
    data = response.json()
    assert "concept_id" in data
    assert data["status"] == "added"

def test_query_knowledge(test_client):
    # First add some knowledge
    fact = {
        "type": "fact",
        "subject": "dog",
        "predicate": "is",
        "object": "mammal"
    }
    test_client.post("/knowledge/facts", json=fact)
    
    # Query the knowledge
    query = {
        "type": "query",
        "pattern": {
            "subject": "dog",
            "predicate": "is",
            "object": "?x"
        }
    }
    
    response = test_client.post("/knowledge/query", json=query)
    assert response.status_code == 200
    data = response.json()
    assert len(data["results"]) > 0
    assert data["results"][0]["object"] == "mammal"

def test_symbolic_reasoning(test_client):
    # Add knowledge for reasoning
    knowledge = [
        {
            "type": "fact",
            "subject": "Socrates",
            "predicate": "is",
            "object": "human"
        },
        {
            "type": "rule",
            "if_clause": [
                {
                    "subject": "?x",
                    "predicate": "is",
                    "object": "human"
                }
            ],
            "then_clause": [
                {
                    "subject": "?x",
                    "predicate": "is",
                    "object": "mortal"
                }
            ]
        }
    ]
    
    for k in knowledge:
        if k["type"] == "fact":
            test_client.post("/knowledge/facts", json=k)
        else:
            test_client.post("/knowledge/rules", json=k)
    
    # Perform reasoning
    query = {
        "type": "symbolic",
        "query": "Is Socrates mortal?"
    }
    
    response = test_client.post("/reason/symbolic", json=query)
    assert response.status_code == 200
    data = response.json()
    assert data["result"] == True
    assert "explanation" in data

def test_causal_reasoning(test_client):
    # Add causal knowledge
    causal_model = {
        "variables": ["rain", "wet_grass", "sprinkler"],
        "relationships": [
            {
                "cause": "rain",
                "effect": "wet_grass",
                "probability": 0.9
            },
            {
                "cause": "sprinkler",
                "effect": "wet_grass",
                "probability": 0.8
            }
        ]
    }
    
    test_client.post("/knowledge/causal", json=causal_model)
    
    # Perform causal reasoning
    query = {
        "type": "causal",
        "evidence": {"wet_grass": True},
        "query_variable": "rain"
    }
    
    response = test_client.post("/reason/causal", json=query)
    assert response.status_code == 200
    data = response.json()
    assert "probability" in data
    assert 0 <= data["probability"] <= 1

def test_common_sense_reasoning(test_client):
    query = {
        "type": "common_sense",
        "context": "A person is hungry",
        "question": "What should they do?"
    }
    
    response = test_client.post("/reason/common_sense", json=query)
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "confidence" in data
    assert "reasoning_steps" in data

def test_pattern_matching(test_client):
    # Add patterns
    pattern = {
        "name": "cause_effect",
        "template": "If ?x then ?y",
        "variables": ["?x", "?y"],
        "constraints": {
            "?x": {"type": "event"},
            "?y": {"type": "event"}
        }
    }
    
    test_client.post("/knowledge/patterns", json=pattern)
    
    # Match pattern
    text = "If it rains then the ground gets wet"
    match_request = {
        "text": text,
        "pattern_name": "cause_effect"
    }
    
    response = test_client.post("/reason/match_pattern", json=match_request)
    assert response.status_code == 200
    data = response.json()
    assert "matches" in data
    assert len(data["matches"]) > 0

def test_knowledge_inference(test_client):
    # Add knowledge for inference
    knowledge = [
        {
            "type": "fact",
            "subject": "cat",
            "predicate": "has",
            "object": "fur"
        },
        {
            "type": "fact",
            "subject": "cat",
            "predicate": "has",
            "object": "whiskers"
        }
    ]
    
    for k in knowledge:
        test_client.post("/knowledge/facts", json=k)
    
    # Request inference
    query = {
        "subject": "cat",
        "inference_type": "properties"
    }
    
    response = test_client.post("/reason/infer", json=query)
    assert response.status_code == 200
    data = response.json()
    assert "inferred_properties" in data
    assert len(data["inferred_properties"]) >= 2

def test_reasoning_chain(test_client):
    chain_request = {
        "initial_fact": "The sky is cloudy",
        "reasoning_steps": [
            "weather_prediction",
            "causal_effects",
            "action_recommendation"
        ],
        "max_depth": 3
    }
    
    response = test_client.post("/reason/chain", json=chain_request)
    assert response.status_code == 200
    data = response.json()
    assert "reasoning_chain" in data
    assert "final_conclusion" in data
    assert len(data["reasoning_chain"]) <= 3

def test_reasoning_config(test_client):
    config_data = {
        "inference_depth": 5,
        "confidence_threshold": 0.7,
        "reasoning_timeout": 10,
        "max_branches": 100,
        "cache_settings": {
            "enabled": True,
            "max_size": 1000,
            "ttl": 3600
        }
    }
    
    response = test_client.post("/config", json=config_data)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"

def test_get_reasoning_stats(test_client):
    response = test_client.get("/stats")
    assert response.status_code == 200
    data = response.json()
    
    assert "total_facts" in data
    assert "total_rules" in data
    assert "inference_stats" in data
    assert "performance_metrics" in data

def test_health_check(test_client):
    response = test_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    
    assert data["status"] == "healthy"
    assert "knowledge_base_status" in data
    assert "inference_engine_status" in data 