import pytest
from fastapi.testclient import TestClient
from communication.communication_service import app, CommunicationManager
import json
from datetime import datetime

@pytest.fixture
def test_client():
    return TestClient(app)

@pytest.fixture
def communication_manager():
    return CommunicationManager()

def test_send_message(test_client):
    message = {
        "sender": "agent1",
        "receiver": "agent2",
        "content": "Test message",
        "message_type": "text",
        "priority": "normal",
        "metadata": {
            "conversation_id": "test_conv_1",
            "timestamp": datetime.now().isoformat()
        }
    }
    
    response = test_client.post("/messages/send", json=message)
    assert response.status_code == 200
    data = response.json()
    assert "message_id" in data
    assert data["status"] == "sent"

def test_get_message(test_client):
    # First send a message
    message = {
        "sender": "agent1",
        "receiver": "agent2",
        "content": "Test message for retrieval",
        "message_type": "text"
    }
    
    send_response = test_client.post("/messages/send", json=message)
    message_id = send_response.json()["message_id"]
    
    # Retrieve the message
    response = test_client.get(f"/messages/{message_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["content"] == "Test message for retrieval"
    assert data["sender"] == "agent1"

def test_get_agent_messages(test_client):
    # Send multiple messages
    messages = [
        {
            "sender": "agent1",
            "receiver": "agent3",
            "content": f"Test message {i}",
            "message_type": "text"
        }
        for i in range(3)
    ]
    
    for message in messages:
        test_client.post("/messages/send", json=message)
    
    # Get messages for agent
    response = test_client.get("/messages/agent/agent3")
    assert response.status_code == 200
    data = response.json()
    assert len(data["messages"]) == 3

def test_broadcast_message(test_client):
    broadcast = {
        "sender": "system",
        "receivers": ["agent1", "agent2", "agent3"],
        "content": "Broadcast test message",
        "message_type": "broadcast",
        "priority": "high"
    }
    
    response = test_client.post("/messages/broadcast", json=broadcast)
    assert response.status_code == 200
    data = response.json()
    assert "broadcast_id" in data
    assert len(data["delivered_to"]) == 3

def test_create_conversation(test_client):
    conversation = {
        "participants": ["agent1", "agent2"],
        "metadata": {
            "topic": "test conversation",
            "created_at": datetime.now().isoformat()
        }
    }
    
    response = test_client.post("/conversations/create", json=conversation)
    assert response.status_code == 200
    data = response.json()
    assert "conversation_id" in data

def test_add_to_conversation(test_client):
    # First create a conversation
    conversation = {
        "participants": ["agent1"],
        "metadata": {"topic": "test"}
    }
    conv_response = test_client.post("/conversations/create", json=conversation)
    conversation_id = conv_response.json()["conversation_id"]
    
    # Add participant
    add_request = {
        "agent_id": "agent2"
    }
    response = test_client.post(
        f"/conversations/{conversation_id}/add",
        json=add_request
    )
    assert response.status_code == 200
    data = response.json()
    assert "agent2" in data["participants"]

def test_leave_conversation(test_client):
    # First create a conversation
    conversation = {
        "participants": ["agent1", "agent2"],
        "metadata": {"topic": "test"}
    }
    conv_response = test_client.post("/conversations/create", json=conversation)
    conversation_id = conv_response.json()["conversation_id"]
    
    # Leave conversation
    leave_request = {
        "agent_id": "agent2"
    }
    response = test_client.post(
        f"/conversations/{conversation_id}/leave",
        json=leave_request
    )
    assert response.status_code == 200
    data = response.json()
    assert "agent2" not in data["participants"]

def test_get_conversation_history(test_client):
    # Create conversation and send messages
    conversation = {
        "participants": ["agent1", "agent2"],
        "metadata": {"topic": "test"}
    }
    conv_response = test_client.post("/conversations/create", json=conversation)
    conversation_id = conv_response.json()["conversation_id"]
    
    messages = [
        {
            "sender": "agent1",
            "receiver": "agent2",
            "content": f"Test message {i}",
            "message_type": "text",
            "metadata": {"conversation_id": conversation_id}
        }
        for i in range(3)
    ]
    
    for message in messages:
        test_client.post("/messages/send", json=message)
    
    # Get history
    response = test_client.get(f"/conversations/{conversation_id}/history")
    assert response.status_code == 200
    data = response.json()
    assert len(data["messages"]) == 3

def test_message_acknowledgment(test_client):
    # Send a message
    message = {
        "sender": "agent1",
        "receiver": "agent2",
        "content": "Test message for ack",
        "message_type": "text",
        "require_ack": True
    }
    
    send_response = test_client.post("/messages/send", json=message)
    message_id = send_response.json()["message_id"]
    
    # Acknowledge message
    ack_request = {
        "agent_id": "agent2",
        "status": "received"
    }
    response = test_client.post(f"/messages/{message_id}/acknowledge", json=ack_request)
    assert response.status_code == 200
    data = response.json()
    assert data["acknowledgment_status"] == "received"

def test_message_encryption(test_client):
    message = {
        "sender": "agent1",
        "receiver": "agent2",
        "content": "Sensitive test message",
        "message_type": "text",
        "encryption": {
            "enabled": True,
            "algorithm": "AES"
        }
    }
    
    response = test_client.post("/messages/send", json=message)
    assert response.status_code == 200
    data = response.json()
    assert data["encryption_status"] == "encrypted"

def test_communication_config(test_client):
    config_data = {
        "max_message_size": 1024,
        "default_priority": "normal",
        "encryption_settings": {
            "default_algorithm": "AES",
            "key_rotation_interval": 3600
        },
        "rate_limiting": {
            "messages_per_minute": 60,
            "burst_size": 10
        }
    }
    
    response = test_client.post("/config", json=config_data)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"

def test_get_communication_stats(test_client):
    response = test_client.get("/stats")
    assert response.status_code == 200
    data = response.json()
    
    assert "total_messages" in data
    assert "active_conversations" in data
    assert "message_types" in data
    assert "delivery_stats" in data

def test_health_check(test_client):
    response = test_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    
    assert data["status"] == "healthy"
    assert "queue_status" in data
    assert "connection_pool" in data 