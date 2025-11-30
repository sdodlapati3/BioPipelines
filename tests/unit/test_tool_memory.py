"""Tests for Phase 6: RAG-Enhanced Tool Selection."""

import pytest
import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

from workflow_composer.agents.tool_memory import (
    ToolMemory,
    ToolMemoryConfig,
    ToolExecutionRecord,
    ToolStats,
    ToolMemoryIndex,
    with_tool_memory,
)


class TestToolMemoryConfig:
    """Tests for ToolMemoryConfig."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = ToolMemoryConfig()
        
        assert config.max_records == 10000
        assert config.similarity_threshold == 0.75
        assert config.top_k == 5
        assert config.confidence_boost == 0.15
        assert config.use_embeddings is True
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = ToolMemoryConfig(
            max_records=5000,
            similarity_threshold=0.8,
            top_k=10,
        )
        
        assert config.max_records == 5000
        assert config.similarity_threshold == 0.8
        assert config.top_k == 10


class TestToolExecutionRecord:
    """Tests for ToolExecutionRecord."""
    
    def test_create_record(self):
        """Test creating a record."""
        record = ToolExecutionRecord.create(
            query="Find RNA-seq data",
            tool_name="search_datasets",
            tool_args={"query": "RNA-seq"},
            success=True,
            result_summary="Found 10 datasets",
            duration_ms=150.0,
        )
        
        assert record.query == "Find RNA-seq data"
        assert record.tool_name == "search_datasets"
        assert record.success is True
        assert record.duration_ms == 150.0
        assert record.record_id is not None
        assert record.timestamp is not None
    
    def test_normalize_query(self):
        """Test query normalization."""
        normalized = ToolExecutionRecord._normalize_query("  Find  RNA-seq  DATA  ")
        
        assert normalized == "find rna-seq data"
    
    def test_to_dict(self):
        """Test converting to dictionary."""
        record = ToolExecutionRecord.create(
            query="Test query",
            tool_name="test_tool",
            tool_args={},
            success=True,
            result_summary="Success",
            duration_ms=100.0,
        )
        
        data = record.to_dict()
        
        assert "record_id" in data
        assert "query" in data
        assert "tool_name" in data
        assert "success" in data
        assert "timestamp" in data
    
    def test_from_dict(self):
        """Test creating from dictionary."""
        data = {
            "record_id": "test123",
            "query": "Test query",
            "query_normalized": "test query",
            "tool_name": "test_tool",
            "tool_args": {},
            "success": True,
            "result_summary": "Success",
            "duration_ms": 100.0,
            "timestamp": datetime.now().isoformat(),
        }
        
        record = ToolExecutionRecord.from_dict(data)
        
        assert record.record_id == "test123"
        assert record.query == "Test query"
        assert record.tool_name == "test_tool"


class TestToolStats:
    """Tests for ToolStats."""
    
    def test_initial_stats(self):
        """Test initial stats values."""
        stats = ToolStats(tool_name="test_tool")
        
        assert stats.total_executions == 0
        assert stats.successful_executions == 0
        assert stats.success_rate == 0.0
    
    def test_success_rate(self):
        """Test success rate calculation."""
        stats = ToolStats(tool_name="test_tool")
        stats.total_executions = 10
        stats.successful_executions = 8
        
        assert stats.success_rate == 0.8
    
    def test_avg_duration(self):
        """Test average duration calculation."""
        stats = ToolStats(tool_name="test_tool")
        stats.total_executions = 5
        stats.total_duration_ms = 500.0
        
        assert stats.avg_duration_ms == 100.0
    
    def test_feedback_score(self):
        """Test feedback score calculation."""
        stats = ToolStats(tool_name="test_tool")
        stats.positive_feedback = 7
        stats.negative_feedback = 3
        
        assert stats.feedback_score == 0.4  # (7 - 3) / 10
    
    def test_record_execution(self):
        """Test recording an execution."""
        stats = ToolStats(tool_name="test_tool")
        
        record = ToolExecutionRecord.create(
            query="Test",
            tool_name="test_tool",
            tool_args={},
            success=True,
            result_summary="Done",
            duration_ms=100.0,
        )
        
        stats.record_execution(record)
        
        assert stats.total_executions == 1
        assert stats.successful_executions == 1
        assert stats.total_duration_ms == 100.0


class TestToolMemoryIndex:
    """Tests for ToolMemoryIndex."""
    
    def test_keyword_similarity(self):
        """Test keyword-based similarity."""
        config = ToolMemoryConfig(use_embeddings=False)
        index = ToolMemoryIndex(config)
        
        # Identical queries
        score = index.keyword_similarity("RNA-seq data", "RNA-seq data")
        assert score == 1.0
        
        # Partial overlap
        score = index.keyword_similarity("RNA-seq data human", "RNA-seq human liver")
        assert 0 < score < 1
        
        # No overlap
        score = index.keyword_similarity("apple banana", "car truck")
        assert score == 0.0
    
    def test_compute_similarity_empty(self):
        """Test similarity with empty embeddings."""
        config = ToolMemoryConfig(use_embeddings=False)
        index = ToolMemoryIndex(config)
        
        score = index.compute_similarity([], [])
        assert score == 0.0


class TestToolMemory:
    """Tests for ToolMemory."""
    
    def test_memory_creation(self):
        """Test creating tool memory."""
        memory = ToolMemory()
        
        assert memory.get_record_count() == 0
    
    def test_record_success(self):
        """Test recording successful execution."""
        config = ToolMemoryConfig(use_embeddings=False)
        memory = ToolMemory(config)
        
        record = memory.record_success(
            query="Find RNA-seq data",
            tool_name="search_datasets",
            tool_args={"query": "RNA-seq"},
            result_summary="Found 10 datasets",
            duration_ms=150.0,
        )
        
        assert record.success is True
        assert memory.get_record_count() == 1
        
        # Check stats updated
        stats = memory.get_tool_stats("search_datasets")
        assert stats is not None
        assert stats.total_executions == 1
        assert stats.successful_executions == 1
    
    def test_record_failure(self):
        """Test recording failed execution."""
        config = ToolMemoryConfig(use_embeddings=False)
        memory = ToolMemory(config)
        
        record = memory.record_failure(
            query="Find data",
            tool_name="search_datasets",
            tool_args={},
            error_message="API error",
            duration_ms=50.0,
        )
        
        assert record.success is False
        assert "Error" in record.result_summary
        
        stats = memory.get_tool_stats("search_datasets")
        assert stats.failed_executions == 1
    
    def test_add_feedback(self):
        """Test adding feedback to a record."""
        config = ToolMemoryConfig(use_embeddings=False)
        memory = ToolMemory(config)
        
        record = memory.record_success(
            query="Test",
            tool_name="test_tool",
            tool_args={},
            result_summary="Done",
            duration_ms=100.0,
        )
        
        # Add positive feedback
        result = memory.add_feedback(record.record_id, "positive")
        assert result is True
        
        stats = memory.get_tool_stats("test_tool")
        assert stats.positive_feedback == 1
        
        # Invalid feedback
        with pytest.raises(ValueError):
            memory.add_feedback(record.record_id, "invalid")
    
    def test_find_similar_keyword(self):
        """Test finding similar records with keyword matching."""
        config = ToolMemoryConfig(use_embeddings=False, similarity_threshold=0.3)
        memory = ToolMemory(config)
        
        # Add some records
        memory.record_success(
            query="Find RNA-seq data for human liver",
            tool_name="search_datasets",
            tool_args={},
            result_summary="Found datasets",
            duration_ms=100.0,
        )
        
        memory.record_success(
            query="Search ChIP-seq experiments",
            tool_name="search_datasets",
            tool_args={},
            result_summary="Found experiments",
            duration_ms=120.0,
        )
        
        # Find similar to RNA-seq query
        similar = memory.find_similar("RNA-seq human data")
        
        assert len(similar) >= 1
        assert similar[0][0].query == "Find RNA-seq data for human liver"
    
    def test_get_tool_suggestion(self):
        """Test getting tool suggestions."""
        config = ToolMemoryConfig(use_embeddings=False, similarity_threshold=0.3)
        memory = ToolMemory(config)
        
        # Add successful record
        memory.record_success(
            query="Find RNA-seq data",
            tool_name="search_datasets",
            tool_args={},
            result_summary="Found datasets",
            duration_ms=100.0,
        )
        
        # Get suggestions
        boosts = memory.get_tool_suggestion(
            query="Find RNA-seq experiments",
            candidate_tools=["search_datasets", "generate_workflow"],
        )
        
        assert "search_datasets" in boosts
        assert "generate_workflow" in boosts
    
    def test_get_recent_records(self):
        """Test getting recent records."""
        config = ToolMemoryConfig(use_embeddings=False)
        memory = ToolMemory(config)
        
        # Add records
        for i in range(5):
            memory.record_success(
                query=f"Query {i}",
                tool_name="test_tool",
                tool_args={},
                result_summary=f"Result {i}",
                duration_ms=100.0,
            )
        
        recent = memory.get_recent_records(limit=3)
        
        assert len(recent) == 3
        # Most recent should be first
        assert "Query 4" in recent[0].query
    
    def test_clear(self):
        """Test clearing memory."""
        config = ToolMemoryConfig(use_embeddings=False)
        memory = ToolMemory(config)
        
        memory.record_success(
            query="Test",
            tool_name="test_tool",
            tool_args={},
            result_summary="Done",
            duration_ms=100.0,
        )
        
        assert memory.get_record_count() == 1
        
        memory.clear()
        
        assert memory.get_record_count() == 0
        assert len(memory.get_all_stats()) == 0
    
    def test_eviction(self):
        """Test record eviction when over limit."""
        config = ToolMemoryConfig(max_records=5, use_embeddings=False)
        memory = ToolMemory(config)
        
        # Add more records than limit
        for i in range(7):
            memory.record_success(
                query=f"Query {i}",
                tool_name="test_tool",
                tool_args={},
                result_summary=f"Result {i}",
                duration_ms=100.0,
            )
        
        # Should be limited to max_records
        assert memory.get_record_count() <= 5
    
    def test_persistence(self):
        """Test saving and loading memory."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        
        try:
            # Create memory and add records
            config = ToolMemoryConfig(
                persistence_path=path,
                use_embeddings=False,
            )
            memory = ToolMemory(config)
            
            memory.record_success(
                query="Test query",
                tool_name="test_tool",
                tool_args={"key": "value"},
                result_summary="Done",
                duration_ms=100.0,
            )
            
            # Load new memory from same path
            memory2 = ToolMemory(config)
            
            assert memory2.get_record_count() >= 1
            
        finally:
            Path(path).unlink(missing_ok=True)
    
    def test_get_all_stats(self):
        """Test getting all tool stats."""
        config = ToolMemoryConfig(use_embeddings=False)
        memory = ToolMemory(config)
        
        memory.record_success(
            query="Query 1",
            tool_name="tool_a",
            tool_args={},
            result_summary="Done",
            duration_ms=100.0,
        )
        
        memory.record_success(
            query="Query 2",
            tool_name="tool_b",
            tool_args={},
            result_summary="Done",
            duration_ms=150.0,
        )
        
        stats = memory.get_all_stats()
        
        assert "tool_a" in stats
        assert "tool_b" in stats


class TestWithToolMemoryDecorator:
    """Tests for with_tool_memory decorator."""
    
    def test_sync_decorator_success(self):
        """Test decorator with sync function success."""
        config = ToolMemoryConfig(use_embeddings=False)
        memory = ToolMemory(config)
        
        @with_tool_memory(memory=memory)
        def test_func(query: str):
            return ["result1", "result2"]
        
        result = test_func("test query")
        
        assert result == ["result1", "result2"]
        assert memory.get_record_count() == 1
        
        record = memory.get_recent_records(1)[0]
        assert record.success is True
        assert "2 items" in record.result_summary
    
    def test_sync_decorator_failure(self):
        """Test decorator with sync function failure."""
        config = ToolMemoryConfig(use_embeddings=False)
        memory = ToolMemory(config)
        
        @with_tool_memory(memory=memory)
        def failing_func(query: str):
            raise ValueError("Test error")
        
        with pytest.raises(ValueError):
            failing_func("test query")
        
        assert memory.get_record_count() == 1
        
        record = memory.get_recent_records(1)[0]
        assert record.success is False
    
    @pytest.mark.asyncio
    async def test_async_decorator_success(self):
        """Test decorator with async function success."""
        config = ToolMemoryConfig(use_embeddings=False)
        memory = ToolMemory(config)
        
        @with_tool_memory(memory=memory)
        async def async_test_func(query: str):
            return {"data": "test"}
        
        result = await async_test_func("test query")
        
        assert result == {"data": "test"}
        assert memory.get_record_count() == 1
        
        record = memory.get_recent_records(1)[0]
        assert record.success is True
        assert "1 keys" in record.result_summary


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
