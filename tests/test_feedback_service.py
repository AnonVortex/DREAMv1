import pytest
from unittest.mock import Mock, patch
from datetime import datetime, timedelta

from feedback.service import FeedbackService
from shared.models import Feedback, FeedbackType, FeedbackSource
from shared.exceptions import FeedbackError

@pytest.fixture
def feedback_service():
    return FeedbackService()

@pytest.fixture
def sample_feedback():
    return Feedback(
        id="test_feedback_1",
        type=FeedbackType.REWARD,
        source=FeedbackSource.HUMAN,
        content={
            "score": 0.8,
            "action": "response_generation",
            "timestamp": datetime.now().isoformat()
        },
        metadata={
            "session_id": "test_session",
            "agent_id": "agent_123",
            "priority": "high"
        }
    )

class TestFeedbackService:
    def test_submit_feedback(self, feedback_service, sample_feedback):
        # Test submitting new feedback
        feedback_id = feedback_service.submit(sample_feedback)
        assert feedback_id is not None
        assert isinstance(feedback_id, str)

    def test_get_feedback(self, feedback_service, sample_feedback):
        # Test retrieving submitted feedback
        feedback_id = feedback_service.submit(sample_feedback)
        retrieved = feedback_service.get(feedback_id)
        assert retrieved.id == feedback_id
        assert retrieved.type == sample_feedback.type
        assert retrieved.content == sample_feedback.content

    def test_invalid_feedback_type(self, feedback_service):
        # Test submitting feedback with invalid type
        invalid_feedback = Feedback(
            type="INVALID_TYPE",
            source=FeedbackSource.HUMAN,
            content={"score": 0.5}
        )
        with pytest.raises(ValueError):
            feedback_service.submit(invalid_feedback)

    def test_batch_feedback(self, feedback_service):
        # Test batch feedback submission
        feedbacks = [
            Feedback(
                type=FeedbackType.REWARD,
                source=FeedbackSource.HUMAN,
                content={"score": i/10}
            ) for i in range(5)
        ]
        
        # Test batch submit
        ids = feedback_service.batch_submit(feedbacks)
        assert len(ids) == 5
        
        # Test batch retrieve
        retrieved = feedback_service.batch_get(ids)
        assert len(retrieved) == 5
        assert all(f.content["score"] == i/10 for i, f in enumerate(retrieved))

    def test_feedback_aggregation(self, feedback_service, sample_feedback):
        # Test feedback aggregation
        feedbacks = [
            Feedback(
                type=FeedbackType.REWARD,
                source=FeedbackSource.HUMAN,
                content={"score": i/10},
                metadata={"agent_id": "agent_123"}
            ) for i in range(10)
        ]
        
        for f in feedbacks:
            feedback_service.submit(f)
        
        aggregated = feedback_service.get_aggregated_feedback(
            agent_id="agent_123",
            start_time=datetime.now() - timedelta(hours=1),
            end_time=datetime.now()
        )
        
        assert "average_score" in aggregated
        assert "total_count" in aggregated
        assert aggregated["total_count"] == 10

    def test_feedback_processing(self, feedback_service, sample_feedback):
        # Test feedback processing pipeline
        feedback_id = feedback_service.submit(sample_feedback)
        
        # Process feedback
        result = feedback_service.process_feedback(feedback_id)
        assert result["status"] == "processed"
        assert "processed_at" in result

    def test_feedback_metrics(self, feedback_service, sample_feedback):
        # Test feedback metrics calculation
        feedback_service.submit(sample_feedback)
        
        metrics = feedback_service.get_metrics(
            start_time=datetime.now() - timedelta(hours=1),
            end_time=datetime.now()
        )
        
        assert "feedback_count" in metrics
        assert "average_score" in metrics
        assert "response_times" in metrics

    @patch('feedback.service.FeedbackService._notify_learning_service')
    def test_learning_service_integration(self, mock_notify, feedback_service, sample_feedback):
        # Test integration with learning service
        feedback_service.submit(sample_feedback)
        mock_notify.assert_called_once()

    def test_feedback_filtering(self, feedback_service, sample_feedback):
        # Test feedback filtering
        feedback_service.submit(sample_feedback)
        
        filtered = feedback_service.filter_feedback(
            feedback_type=FeedbackType.REWARD,
            source=FeedbackSource.HUMAN,
            min_score=0.7,
            max_score=0.9
        )
        
        assert len(filtered) > 0
        assert all(f.type == FeedbackType.REWARD for f in filtered)
        assert all(0.7 <= f.content["score"] <= 0.9 for f in filtered)

    def test_feedback_priority(self, feedback_service):
        # Test feedback priority handling
        high_priority = Feedback(
            type=FeedbackType.REWARD,
            source=FeedbackSource.HUMAN,
            content={"score": 0.9},
            metadata={"priority": "high"}
        )
        
        low_priority = Feedback(
            type=FeedbackType.REWARD,
            source=FeedbackSource.HUMAN,
            content={"score": 0.5},
            metadata={"priority": "low"}
        )
        
        feedback_service.submit(high_priority)
        feedback_service.submit(low_priority)
        
        queue = feedback_service.get_processing_queue()
        assert queue[0].metadata["priority"] == "high"

    def test_feedback_validation(self, feedback_service):
        # Test feedback validation
        invalid_feedback = Feedback(
            type=FeedbackType.REWARD,
            source=FeedbackSource.HUMAN,
            content={"score": 2.0}  # Invalid score > 1.0
        )
        
        with pytest.raises(ValueError):
            feedback_service.submit(invalid_feedback)

    def test_feedback_persistence(self, feedback_service, sample_feedback):
        # Test feedback persistence across service restarts
        feedback_id = feedback_service.submit(sample_feedback)
        
        # Simulate service restart
        new_service = FeedbackService()
        retrieved = new_service.get(feedback_id)
        assert retrieved.id == feedback_id
        assert retrieved.content == sample_feedback.content 