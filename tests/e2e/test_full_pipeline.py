import pytest
from datetime import datetime
import json
import asyncio

from perception.service import PerceptionService
from memory.service import MemoryService
from learning.service import LearningService
from reasoning.service import ReasoningService
from communication.service import CommunicationService
from feedback.service import FeedbackService
from integration.service import IntegrationService

from shared.models import (
    Input, Memory, MemoryType,
    LearningModel, Inference,
    Message, Feedback
)

@pytest.fixture
def services():
    return {
        'perception': PerceptionService(),
        'memory': MemoryService(),
        'learning': LearningService(),
        'reasoning': ReasoningService(),
        'communication': CommunicationService(),
        'feedback': FeedbackService(),
        'integration': IntegrationService()
    }

@pytest.fixture
def sample_input():
    return Input(
        type="text",
        content="Hello, how can I help you today?",
        metadata={
            "source": "user",
            "timestamp": datetime.now().isoformat()
        }
    )

class TestFullPipeline:
    async def test_complete_interaction_flow(self, services, sample_input):
        """Test a complete interaction flow through all services"""
        
        # 1. Perception Phase
        perception_result = await services['perception'].process(sample_input)
        assert perception_result.features is not None
        assert perception_result.patterns is not None
        
        # 2. Memory Storage
        memory = Memory(
            type=MemoryType.EPISODIC,
            content={
                "input": sample_input.dict(),
                "perception": perception_result.dict()
            }
        )
        memory_id = await services['memory'].store(memory)
        assert memory_id is not None
        
        # 3. Learning Phase
        model = await services['learning'].get_model("conversation")
        learning_result = await services['learning'].process(
            model_id=model.id,
            input_data=perception_result.features
        )
        assert learning_result.predictions is not None
        
        # 4. Reasoning Phase
        inference = await services['reasoning'].infer(
            context={
                "input": sample_input.dict(),
                "perception": perception_result.dict(),
                "learning": learning_result.dict()
            }
        )
        assert inference.decision is not None
        
        # 5. Communication Phase
        message = Message(
            content=inference.decision.response,
            metadata={
                "input_id": sample_input.id,
                "inference_id": inference.id
            }
        )
        message_id = await services['communication'].send(message)
        assert message_id is not None
        
        # 6. Feedback Collection
        feedback = Feedback(
            type="user_satisfaction",
            content={
                "score": 0.9,
                "message_id": message_id
            }
        )
        feedback_id = await services['feedback'].submit(feedback)
        assert feedback_id is not None
        
        # 7. Integration Check
        integration_result = await services['integration'].execute_integration(
            "metrics_service",
            method="POST",
            data={
                "interaction_id": sample_input.id,
                "metrics": {
                    "response_time": 0.5,
                    "user_satisfaction": 0.9
                }
            }
        )
        assert integration_result["status"] == "success"

    async def test_error_handling_flow(self, services):
        """Test error handling across services"""
        
        # 1. Invalid Input
        with pytest.raises(ValueError):
            await services['perception'].process(None)
        
        # 2. Memory Not Found
        with pytest.raises(Exception):
            await services['memory'].retrieve("nonexistent_id")
        
        # 3. Model Not Found
        with pytest.raises(Exception):
            await services['learning'].get_model("nonexistent_model")
        
        # 4. Invalid Inference
        with pytest.raises(ValueError):
            await services['reasoning'].infer({})
        
        # 5. Message Delivery Failure
        with pytest.raises(Exception):
            await services['communication'].send(None)
        
        # 6. Invalid Feedback
        with pytest.raises(ValueError):
            await services['feedback'].submit(None)
        
        # 7. Integration Failure
        with pytest.raises(Exception):
            await services['integration'].execute_integration(
                "nonexistent_integration",
                method="GET"
            )

    async def test_concurrent_operations(self, services, sample_input):
        """Test concurrent operations across services"""
        
        async def process_interaction(input_data):
            # Parallel processing
            perception_task = asyncio.create_task(
                services['perception'].process(input_data)
            )
            memory_task = asyncio.create_task(
                services['memory'].search(query="recent_interactions")
            )
            model_task = asyncio.create_task(
                services['learning'].get_model("conversation")
            )
            
            # Wait for all tasks
            perception_result = await perception_task
            memory_result = await memory_task
            model = await model_task
            
            return {
                "perception": perception_result,
                "memory": memory_result,
                "model": model
            }
        
        # Run multiple concurrent interactions
        tasks = [
            process_interaction(sample_input)
            for _ in range(5)
        ]
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 5
        assert all(r["perception"] is not None for r in results)
        assert all(r["memory"] is not None for r in results)
        assert all(r["model"] is not None for r in results)

    async def test_performance_metrics(self, services, sample_input):
        """Test performance metrics collection"""
        
        start_time = datetime.now()
        
        # Process interaction
        perception_result = await services['perception'].process(sample_input)
        memory_id = await services['memory'].store(Memory(
            type=MemoryType.EPISODIC,
            content={"input": sample_input.dict()}
        ))
        model = await services['learning'].get_model("conversation")
        inference = await services['reasoning'].infer(
            context={"input": sample_input.dict()}
        )
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        # Record metrics
        metrics = {
            "processing_time": processing_time,
            "perception_confidence": perception_result.confidence,
            "memory_id": memory_id,
            "model_version": model.version,
            "inference_confidence": inference.confidence
        }
        
        # Store metrics
        integration_result = await services['integration'].execute_integration(
            "metrics_service",
            method="POST",
            data=metrics
        )
        
        assert integration_result["status"] == "success"
        assert processing_time > 0

    async def test_service_recovery(self, services, sample_input):
        """Test service recovery after failures"""
        
        # 1. Simulate service crash
        services['perception'].shutdown()
        services['memory'].shutdown()
        
        # 2. Attempt recovery
        await services['perception'].initialize()
        await services['memory'].initialize()
        
        # 3. Verify functionality
        perception_result = await services['perception'].process(sample_input)
        memory_id = await services['memory'].store(Memory(
            type=MemoryType.EPISODIC,
            content={"input": sample_input.dict()}
        ))
        
        assert perception_result is not None
        assert memory_id is not None

    async def test_data_consistency(self, services, sample_input):
        """Test data consistency across services"""
        
        # 1. Store data
        perception_result = await services['perception'].process(sample_input)
        memory_id = await services['memory'].store(Memory(
            type=MemoryType.EPISODIC,
            content={
                "input": sample_input.dict(),
                "perception": perception_result.dict()
            }
        ))
        
        # 2. Verify consistency
        memory = await services['memory'].retrieve(memory_id)
        assert memory.content["input"] == sample_input.dict()
        assert memory.content["perception"] == perception_result.dict()
        
        # 3. Update data
        updated_content = {
            "input": sample_input.dict(),
            "perception": perception_result.dict(),
            "updated": True
        }
        await services['memory'].update(memory_id, content=updated_content)
        
        # 4. Verify update
        updated_memory = await services['memory'].retrieve(memory_id)
        assert updated_memory.content == updated_content 