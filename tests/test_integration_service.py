import pytest
from unittest.mock import Mock, patch
import json
from datetime import datetime

from integration.service import IntegrationService
from shared.models import Integration, IntegrationType, DataSchema
from shared.exceptions import IntegrationError

@pytest.fixture
def integration_service():
    return IntegrationService()

@pytest.fixture
def sample_integration():
    return Integration(
        id="test_integration_1",
        type=IntegrationType.REST_API,
        config={
            "url": "https://api.example.com",
            "auth_type": "bearer",
            "token": "test_token"
        },
        metadata={
            "name": "Test API",
            "description": "Test integration",
            "owner": "test_team"
        }
    )

@pytest.fixture
def sample_schema():
    return DataSchema(
        id="test_schema_1",
        name="user_data",
        version="1.0",
        fields=[
            {"name": "id", "type": "string", "required": True},
            {"name": "name", "type": "string", "required": True},
            {"name": "age", "type": "integer", "required": False}
        ]
    )

class TestIntegrationService:
    def test_create_integration(self, integration_service, sample_integration):
        # Test creating new integration
        integration_id = integration_service.create(sample_integration)
        assert integration_id is not None
        assert isinstance(integration_id, str)

    def test_get_integration(self, integration_service, sample_integration):
        # Test retrieving integration
        integration_id = integration_service.create(sample_integration)
        retrieved = integration_service.get(integration_id)
        assert retrieved.id == integration_id
        assert retrieved.type == sample_integration.type
        assert retrieved.config == sample_integration.config

    def test_update_integration(self, integration_service, sample_integration):
        # Test updating integration
        integration_id = integration_service.create(sample_integration)
        updated_config = {
            "url": "https://api.example.com/v2",
            "auth_type": "oauth2",
            "client_id": "test_client"
        }
        integration_service.update(integration_id, config=updated_config)
        retrieved = integration_service.get(integration_id)
        assert retrieved.config == updated_config

    def test_delete_integration(self, integration_service, sample_integration):
        # Test deleting integration
        integration_id = integration_service.create(sample_integration)
        integration_service.delete(integration_id)
        with pytest.raises(IntegrationError):
            integration_service.get(integration_id)

    def test_list_integrations(self, integration_service, sample_integration):
        # Test listing integrations
        integration_service.create(sample_integration)
        integrations = integration_service.list()
        assert len(integrations) > 0
        assert isinstance(integrations[0], Integration)

    def test_schema_management(self, integration_service, sample_schema):
        # Test schema management
        # Create schema
        schema_id = integration_service.create_schema(sample_schema)
        assert schema_id is not None
        
        # Retrieve schema
        retrieved = integration_service.get_schema(schema_id)
        assert retrieved.name == sample_schema.name
        assert retrieved.fields == sample_schema.fields
        
        # Update schema
        updated_fields = sample_schema.fields + [
            {"name": "email", "type": "string", "required": True}
        ]
        integration_service.update_schema(schema_id, fields=updated_fields)
        
        # Verify update
        updated = integration_service.get_schema(schema_id)
        assert len(updated.fields) == len(updated_fields)

    def test_data_transformation(self, integration_service, sample_schema):
        # Test data transformation
        schema_id = integration_service.create_schema(sample_schema)
        
        input_data = {
            "id": "user_1",
            "name": "Test User",
            "age": 25
        }
        
        transformed = integration_service.transform_data(
            data=input_data,
            schema_id=schema_id
        )
        
        assert transformed["id"] == input_data["id"]
        assert transformed["name"] == input_data["name"]
        assert transformed["age"] == input_data["age"]

    @patch('integration.service.IntegrationService._make_api_call')
    def test_api_integration(self, mock_api_call, integration_service, sample_integration):
        # Test API integration
        mock_api_call.return_value = {"status": "success"}
        
        integration_id = integration_service.create(sample_integration)
        response = integration_service.execute_integration(
            integration_id,
            method="GET",
            endpoint="/users"
        )
        
        assert response["status"] == "success"
        mock_api_call.assert_called_once()

    def test_batch_operations(self, integration_service):
        # Test batch operations
        integrations = [
            Integration(
                type=IntegrationType.REST_API,
                config={"url": f"https://api{i}.example.com"}
            ) for i in range(5)
        ]
        
        # Batch create
        ids = integration_service.batch_create(integrations)
        assert len(ids) == 5
        
        # Batch get
        retrieved = integration_service.batch_get(ids)
        assert len(retrieved) == 5

    def test_integration_validation(self, integration_service):
        # Test integration validation
        invalid_integration = Integration(
            type=IntegrationType.REST_API,
            config={"invalid_config": True}  # Missing required url
        )
        
        with pytest.raises(ValueError):
            integration_service.create(invalid_integration)

    def test_integration_monitoring(self, integration_service, sample_integration):
        # Test integration monitoring
        integration_id = integration_service.create(sample_integration)
        
        # Get health status
        health = integration_service.get_health_status(integration_id)
        assert "status" in health
        assert "last_check" in health
        
        # Get metrics
        metrics = integration_service.get_metrics(integration_id)
        assert "request_count" in metrics
        assert "error_rate" in metrics

    def test_error_handling(self, integration_service, sample_integration):
        # Test error handling
        integration_id = integration_service.create(sample_integration)
        
        with pytest.raises(IntegrationError):
            integration_service.execute_integration(
                integration_id,
                method="INVALID",
                endpoint="/test"
            )

    def test_rate_limiting(self, integration_service, sample_integration):
        # Test rate limiting
        integration_id = integration_service.create(sample_integration)
        
        # Configure rate limit
        integration_service.configure_rate_limit(
            integration_id,
            requests_per_second=10
        )
        
        # Verify rate limit
        config = integration_service.get_rate_limit_config(integration_id)
        assert config["requests_per_second"] == 10

    def test_webhook_integration(self, integration_service):
        # Test webhook integration
        webhook = Integration(
            type=IntegrationType.WEBHOOK,
            config={
                "url": "https://webhook.example.com",
                "method": "POST",
                "headers": {"Content-Type": "application/json"}
            }
        )
        
        webhook_id = integration_service.create(webhook)
        
        # Register webhook
        integration_service.register_webhook(
            webhook_id,
            events=["user.created", "user.updated"]
        )
        
        # Verify registration
        registered = integration_service.get_registered_webhooks()
        assert len(registered) > 0
        assert registered[0].id == webhook_id 