import pytest
from fastapi.testclient import TestClient
from pipelines.pipeline_service import app, PipelineManager, PipelineStep, Pipeline, StepType, RetryStrategy
import asyncio
import json

@pytest.fixture
def test_client():
    return TestClient(app)

@pytest.fixture
def pipeline_manager():
    return PipelineManager()

def test_register_pipeline(test_client):
    pipeline_def = {
        "name": "test_pipeline",
        "description": "Test pipeline",
        "steps": [
            {
                "id": "step1",
                "type": "task",
                "service": "test_service",
                "endpoint": "/test",
                "input_mapping": {"input": "${pipeline.input}"},
                "output_mapping": {"result": "${step.output}"}
            }
        ]
    }
    
    response = test_client.post(
        "/pipelines",
        json=pipeline_def
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "pipeline_id" in data

def test_execute_pipeline(test_client):
    # First register a pipeline
    pipeline_def = {
        "name": "test_execution",
        "description": "Test execution pipeline",
        "steps": [
            {
                "id": "step1",
                "type": "task",
                "service": "test_service",
                "endpoint": "/test",
                "input_mapping": {"input": "${pipeline.input}"}
            }
        ]
    }
    
    reg_response = test_client.post(
        "/pipelines",
        json=pipeline_def
    )
    pipeline_id = reg_response.json()["pipeline_id"]
    
    # Execute pipeline
    response = test_client.post(
        f"/pipelines/{pipeline_id}/execute",
        json={"input": "test_input"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "execution_id" in data

def test_get_execution_status(test_client):
    # First execute a pipeline
    pipeline_def = {
        "name": "test_status",
        "description": "Test status pipeline",
        "steps": [
            {
                "id": "step1",
                "type": "task",
                "service": "test_service",
                "endpoint": "/test"
            }
        ]
    }
    
    reg_response = test_client.post(
        "/pipelines",
        json=pipeline_def
    )
    pipeline_id = reg_response.json()["pipeline_id"]
    
    exec_response = test_client.post(
        f"/pipelines/{pipeline_id}/execute",
        json={}
    )
    execution_id = exec_response.json()["execution_id"]
    
    # Get status
    response = test_client.get(
        f"/pipelines/executions/{execution_id}"
    )
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "steps" in data

def test_cancel_execution(test_client):
    # First execute a pipeline
    pipeline_def = {
        "name": "test_cancel",
        "description": "Test cancel pipeline",
        "steps": [
            {
                "id": "step1",
                "type": "task",
                "service": "test_service",
                "endpoint": "/test"
            }
        ]
    }
    
    reg_response = test_client.post(
        "/pipelines",
        json=pipeline_def
    )
    pipeline_id = reg_response.json()["pipeline_id"]
    
    exec_response = test_client.post(
        f"/pipelines/{pipeline_id}/execute",
        json={}
    )
    execution_id = exec_response.json()["execution_id"]
    
    # Cancel execution
    response = test_client.post(
        f"/pipelines/executions/{execution_id}/cancel"
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "cancelled"

def test_parallel_execution(pipeline_manager):
    # Create a pipeline with parallel steps
    steps = [
        PipelineStep(
            id="parallel",
            type=StepType.PARALLEL,
            steps=[
                PipelineStep(
                    id="step1",
                    type=StepType.TASK,
                    service="test_service",
                    endpoint="/test1"
                ),
                PipelineStep(
                    id="step2",
                    type=StepType.TASK,
                    service="test_service",
                    endpoint="/test2"
                )
            ]
        )
    ]
    
    pipeline = Pipeline(
        name="test_parallel",
        description="Test parallel execution",
        steps=steps
    )
    
    # Execute pipeline
    execution = asyncio.run(pipeline_manager._run_pipeline(pipeline, {}))
    assert execution.status == "completed"
    assert len(execution.steps) == 3  # parallel step + 2 substeps

def test_conditional_execution(pipeline_manager):
    # Create a pipeline with conditional step
    steps = [
        PipelineStep(
            id="condition",
            type=StepType.CONDITION,
            condition="${pipeline.input.value > 5}",
            if_step=PipelineStep(
                id="if_step",
                type=StepType.TASK,
                service="test_service",
                endpoint="/test_if"
            ),
            else_step=PipelineStep(
                id="else_step",
                type=StepType.TASK,
                service="test_service",
                endpoint="/test_else"
            )
        )
    ]
    
    pipeline = Pipeline(
        name="test_condition",
        description="Test conditional execution",
        steps=steps
    )
    
    # Execute pipeline with condition true
    execution = asyncio.run(pipeline_manager._run_pipeline(
        pipeline, {"input": {"value": 10}}
    ))
    assert execution.status == "completed"
    assert execution.steps[0].executed_branch == "if"

def test_retry_strategy(pipeline_manager):
    # Create a pipeline with retry strategy
    steps = [
        PipelineStep(
            id="retry_step",
            type=StepType.TASK,
            service="test_service",
            endpoint="/test_retry",
            retry_strategy=RetryStrategy(
                max_attempts=3,
                delay_seconds=1
            )
        )
    ]
    
    pipeline = Pipeline(
        name="test_retry",
        description="Test retry strategy",
        steps=steps
    )
    
    # Execute pipeline
    execution = asyncio.run(pipeline_manager._run_pipeline(pipeline, {}))
    assert execution.status in ["completed", "failed"]
    assert execution.steps[0].retry_count <= 3

def test_loop_execution(pipeline_manager):
    # Create a pipeline with loop
    steps = [
        PipelineStep(
            id="loop",
            type=StepType.LOOP,
            condition="${pipeline.counter < 3}",
            step=PipelineStep(
                id="loop_step",
                type=StepType.TASK,
                service="test_service",
                endpoint="/test_loop"
            )
        )
    ]
    
    pipeline = Pipeline(
        name="test_loop",
        description="Test loop execution",
        steps=steps
    )
    
    # Execute pipeline
    execution = asyncio.run(pipeline_manager._run_pipeline(
        pipeline, {"counter": 0}
    ))
    assert execution.status == "completed"
    assert len(execution.steps[0].iterations) == 3

def test_variable_mapping(pipeline_manager):
    # Create a pipeline with variable mapping
    steps = [
        PipelineStep(
            id="step1",
            type=StepType.TASK,
            service="test_service",
            endpoint="/test_vars",
            input_mapping={
                "mapped_input": "${pipeline.input.value}",
                "constant": "test_constant"
            },
            output_mapping={
                "result": "${step.output.data}"
            }
        )
    ]
    
    pipeline = Pipeline(
        name="test_mapping",
        description="Test variable mapping",
        steps=steps
    )
    
    # Execute pipeline
    execution = asyncio.run(pipeline_manager._run_pipeline(
        pipeline, {"input": {"value": "test_value"}}
    ))
    assert execution.status == "completed"
    assert "result" in execution.variables

def test_health_check(test_client):
    response = test_client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy" 