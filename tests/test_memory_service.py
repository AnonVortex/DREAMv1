import pytest
from unittest.mock import Mock, patch
from datetime import datetime
import json

from memory.service import MemoryService
from shared.models import Memory, MemoryType, MemoryQuery
from shared.exceptions import MemoryNotFoundError

@pytest.fixture
def memory_service():
    return MemoryService()

@pytest.fixture
def sample_memory():
    return Memory(
        id="test_memory_1",
        type=MemoryType.EPISODIC,
        content={
            "event": "user_interaction",
            "timestamp": datetime.now().isoformat(),
            "details": {"action": "greeting", "response": "hello"}
        },
        metadata={
            "priority": 1,
            "tags": ["interaction", "greeting"],
            "source": "test"
        },
        created_at=datetime.now(),
        updated_at=datetime.now()
    )

class TestMemoryService:
    def test_store_memory(self, memory_service, sample_memory):
        # Test storing a new memory
        stored_id = memory_service.store(sample_memory)
        assert stored_id is not None
        assert isinstance(stored_id, str)

    def test_retrieve_memory(self, memory_service, sample_memory):
        # Test retrieving a stored memory
        stored_id = memory_service.store(sample_memory)
        retrieved = memory_service.retrieve(stored_id)
        assert retrieved.id == stored_id
        assert retrieved.type == sample_memory.type
        assert retrieved.content == sample_memory.content

    def test_retrieve_nonexistent_memory(self, memory_service):
        # Test retrieving a non-existent memory
        with pytest.raises(MemoryNotFoundError):
            memory_service.retrieve("nonexistent_id")

    def test_update_memory(self, memory_service, sample_memory):
        # Test updating a memory
        stored_id = memory_service.store(sample_memory)
        updated_content = {
            "event": "user_interaction",
            "timestamp": datetime.now().isoformat(),
            "details": {"action": "updated_greeting", "response": "hi"}
        }
        memory_service.update(stored_id, content=updated_content)
        retrieved = memory_service.retrieve(stored_id)
        assert retrieved.content == updated_content

    def test_delete_memory(self, memory_service, sample_memory):
        # Test deleting a memory
        stored_id = memory_service.store(sample_memory)
        memory_service.delete(stored_id)
        with pytest.raises(MemoryNotFoundError):
            memory_service.retrieve(stored_id)

    def test_search_memories(self, memory_service, sample_memory):
        # Test searching memories
        memory_service.store(sample_memory)
        query = MemoryQuery(
            type=MemoryType.EPISODIC,
            tags=["greeting"],
            start_date=datetime.now().date(),
            end_date=datetime.now().date()
        )
        results = memory_service.search(query)
        assert len(results) > 0
        assert results[0].type == MemoryType.EPISODIC

    def test_memory_persistence(self, memory_service, sample_memory):
        # Test memory persistence across service restarts
        stored_id = memory_service.store(sample_memory)
        
        # Simulate service restart
        new_service = MemoryService()
        retrieved = new_service.retrieve(stored_id)
        assert retrieved.id == stored_id
        assert retrieved.content == sample_memory.content

    def test_batch_operations(self, memory_service):
        # Test batch memory operations
        memories = [
            Memory(
                type=MemoryType.EPISODIC,
                content={"event": f"event_{i}"},
                metadata={"priority": i}
            ) for i in range(5)
        ]
        
        # Test batch store
        ids = memory_service.batch_store(memories)
        assert len(ids) == 5
        
        # Test batch retrieve
        retrieved = memory_service.batch_retrieve(ids)
        assert len(retrieved) == 5
        assert all(m.content["event"] == f"event_{i}" for i, m in enumerate(retrieved))

    def test_memory_indexing(self, memory_service, sample_memory):
        # Test memory indexing and fast retrieval
        stored_id = memory_service.store(sample_memory)
        
        # Test indexed search
        results = memory_service.search_by_index(
            field="metadata.tags",
            value="greeting"
        )
        assert len(results) > 0
        assert results[0].id == stored_id

    @patch('memory.service.MemoryService._validate_memory')
    def test_memory_validation(self, mock_validate, memory_service, sample_memory):
        # Test memory validation
        memory_service.store(sample_memory)
        mock_validate.assert_called_once()

    def test_memory_compression(self, memory_service):
        # Test memory compression for large content
        large_content = {"data": "x" * 1000000}  # 1MB of data
        memory = Memory(
            type=MemoryType.SEMANTIC,
            content=large_content,
            metadata={"compressed": True}
        )
        
        stored_id = memory_service.store(memory)
        retrieved = memory_service.retrieve(stored_id)
        assert retrieved.content == large_content

    def test_memory_expiration(self, memory_service, sample_memory):
        # Test memory expiration
        sample_memory.metadata["ttl"] = 0  # Expire immediately
        stored_id = memory_service.store(sample_memory)
        
        # Memory should be expired
        with pytest.raises(MemoryNotFoundError):
            memory_service.retrieve(stored_id)

    def test_memory_type_constraints(self, memory_service):
        # Test memory type constraints
        invalid_memory = Memory(
            type="INVALID_TYPE",
            content={"test": "data"}
        )
        
        with pytest.raises(ValueError):
            memory_service.store(invalid_memory)

    def test_concurrent_operations(self, memory_service, sample_memory):
        # Test concurrent memory operations
        import threading
        
        stored_id = memory_service.store(sample_memory)
        
        def update_memory():
            memory_service.update(
                stored_id,
                content={"event": "concurrent_update"}
            )
        
        threads = [threading.Thread(target=update_memory) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        retrieved = memory_service.retrieve(stored_id)
        assert retrieved.content["event"] == "concurrent_update" 