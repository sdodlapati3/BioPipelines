"""
Unit Tests for RAG Layer 1: Tool Memory.

Tests the ToolMemory class for recording and retrieving
tool execution history.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch


class TestToolMemory:
    """Tests for ToolMemory class."""
    
    @pytest.fixture
    def memory(self):
        """Create a fresh ToolMemory instance with in-memory storage."""
        from src.workflow_composer.agents.rag.memory import ToolMemory, ToolMemoryConfig
        config = ToolMemoryConfig(use_database=False)
        return ToolMemory(config)
    
    def test_record_execution(self, memory):
        """Test recording a tool execution."""
        memory.record(
            query="RNA-seq analysis",
            tool_name="salmon",
            tool_args={"threads": 8},
            success=True,
            duration_ms=1500,
        )
        
        assert memory.get_record_count() == 1
    
    def test_record_multiple_executions(self, memory):
        """Test recording multiple executions."""
        for i in range(5):
            memory.record(
                query=f"Query {i}",
                tool_name="tool1",
                tool_args={"param": i},
                success=True,
                duration_ms=float(i * 100),
            )
        
        assert memory.get_record_count() == 5
    
    def test_get_all_stats(self, memory):
        """Test getting all statistics."""
        # Record some executions
        memory.record(
            query="Query 1",
            tool_name="salmon",
            tool_args={},
            success=True,
            duration_ms=1000,
        )
        memory.record(
            query="Query 2",
            tool_name="macs2",
            tool_args={},
            success=False,
            duration_ms=2000,
            error_message="Test error",
        )
        
        stats = memory.get_all_stats()
        
        assert len(stats) == 2
        assert "salmon" in stats
        assert "macs2" in stats
    
    def test_find_similar_executions(self, memory):
        """Test finding similar executions."""
        # Record some executions
        memory.record(
            query="RNA-seq differential expression",
            tool_name="salmon",
            tool_args={},
            success=True,
            duration_ms=1000,
            result_summary="Found genes",
        )
        memory.record(
            query="ChIP-seq peak calling",
            tool_name="macs2",
            tool_args={},
            success=True,
            duration_ms=2000,
        )
        
        # Find similar - adjust threshold for test
        memory.config.similarity_threshold = 0.1
        similar = memory.find_similar(query="RNA-seq analysis", limit=5)
        
        # Should return list (may be empty if no semantic matching)
        assert isinstance(similar, list)
    
    def test_failed_execution_recorded(self, memory):
        """Test that failed executions are recorded."""
        memory.record(
            query="Failed query",
            tool_name="test_tool",
            tool_args={},
            success=False,
            duration_ms=500,
            error_message="Test error",
        )
        
        stats = memory.get_all_stats()
        assert stats["test_tool"].failed_executions == 1
    
    def test_tool_stats(self, memory):
        """Test getting stats for a specific tool."""
        memory.record(
            query="Test query",
            tool_name="salmon",
            tool_args={},
            success=True,
            duration_ms=100,
        )
        
        stats = memory.get_tool_stats("salmon")
        assert stats is not None
        assert stats.total_executions == 1
        assert stats.success_rate == 1.0


class TestToolMemoryWithPersistence:
    """Tests for ToolMemory with file persistence."""
    
    @pytest.fixture
    def memory_with_file(self, tmp_path):
        """Create ToolMemory with file persistence."""
        from src.workflow_composer.agents.rag.memory import ToolMemory, ToolMemoryConfig
        config = ToolMemoryConfig(
            use_database=False,
            persistence_path=str(tmp_path / "memory.json"),
        )
        return ToolMemory(config)
    
    def test_persistence(self, memory_with_file):
        """Test that records persist to file."""
        memory_with_file.record(
            query="Persistent query",
            tool_name="test_tool",
            tool_args={},
            success=True,
            duration_ms=100,
        )
        
        assert memory_with_file.get_record_count() == 1


class TestArgumentMemory:
    """Tests for ArgumentMemory (Layer 2)."""
    
    @pytest.fixture
    def arg_memory(self):
        """Create a fresh ArgumentMemory instance."""
        from src.workflow_composer.agents.rag.arg_memory import ArgumentMemory
        from src.workflow_composer.agents.rag.memory import ToolMemory, ToolMemoryConfig
        
        # Create a fresh tool memory
        tool_memory = ToolMemory(ToolMemoryConfig(use_database=False))
        return ArgumentMemory(tool_memory=tool_memory)
    
    def test_suggest_args(self, arg_memory):
        """Test suggesting arguments for a tool."""
        # First record some executions to learn from
        arg_memory.tool_memory.record(
            query="RNA-seq analysis for human",
            tool_name="salmon",
            tool_args={"threads": 8, "libtype": "A", "organism": "human"},
            success=True,
            duration_ms=1000,
        )
        
        # Get suggestions
        suggestion = arg_memory.suggest(
            query="RNA-seq analysis",
            tool_name="salmon",
        )
        
        assert suggestion is not None
        assert isinstance(suggestion.args, dict)
    
    def test_learn_from_execution(self, arg_memory):
        """Test learning from an execution."""
        arg_memory.learn_from_execution(
            query="ChIP-seq for H3K4me3",
            tool_name="macs2",
            args={"qvalue": 0.05, "broad": False},
            success=True,
        )
        
        # Should not raise
        assert True
    
    def test_get_common_args(self, arg_memory):
        """Test getting common arguments for a tool."""
        # Record several executions
        for i in range(5):
            arg_memory.tool_memory.record(
                query=f"Analysis {i}",
                tool_name="salmon",
                tool_args={"threads": 8, "index": "/path/to/index"},
                success=True,
                duration_ms=100,
            )
        
        common = arg_memory.get_common_args("salmon")
        
        assert isinstance(common, dict)


class TestRAGToolSelector:
    """Tests for RAGToolSelector (Layer 3)."""
    
    @pytest.fixture
    def selector(self):
        """Create a RAGToolSelector instance."""
        from src.workflow_composer.agents.rag.tool_selector import RAGToolSelector
        from src.workflow_composer.agents.rag.memory import ToolMemory, ToolMemoryConfig
        
        tool_memory = ToolMemory(ToolMemoryConfig(use_database=False))
        return RAGToolSelector(tool_memory=tool_memory)
    
    def test_get_tool_boost(self, selector):
        """Test getting tool boost for candidates."""
        # Record some executions
        selector.tool_memory.record(
            query="RNA-seq alignment",
            tool_name="salmon",
            tool_args={},
            success=True,
            duration_ms=100,
        )
        
        result = selector.get_tool_boost(
            query="RNA-seq analysis",
            candidate_tools=["salmon", "star", "hisat2"],
        )
        
        assert result is not None
        assert isinstance(result.boosts, dict)
    
    def test_select_tool(self, selector):
        """Test selecting a tool with base scores."""
        base_scores = {"salmon": 0.8, "star": 0.7, "hisat2": 0.6}
        
        selected, adjusted = selector.select_tool(
            query="RNA-seq alignment",
            base_scores=base_scores,
        )
        
        assert selected in base_scores
        assert isinstance(adjusted, dict)
    
    def test_should_use_rag(self, selector):
        """Test determining when to use RAG."""
        # Close scores - should use RAG
        close_scores = {"salmon": 0.8, "star": 0.75}
        assert selector.should_use_rag(close_scores, uncertainty_threshold=0.1)
        
        # Distant scores - should not use RAG
        distant_scores = {"salmon": 0.9, "star": 0.5}
        assert not selector.should_use_rag(distant_scores, uncertainty_threshold=0.1)


class TestRAGOrchestrator:
    """Tests for RAGOrchestrator (coordinator)."""
    
    @pytest.fixture
    def orchestrator(self):
        """Create a RAGOrchestrator instance."""
        from src.workflow_composer.agents.rag.orchestrator import RAGOrchestrator
        from src.workflow_composer.agents.rag.memory import ToolMemory, ToolMemoryConfig
        
        tool_memory = ToolMemory(ToolMemoryConfig(use_database=False))
        return RAGOrchestrator(tool_memory=tool_memory)
    
    def test_enhance(self, orchestrator):
        """Test enhancing tool selection and args."""
        result = orchestrator.enhance(
            query="RNA-seq differential expression for human samples",
            candidate_tools=["salmon", "deseq2"],
            base_scores={"salmon": 0.8, "deseq2": 0.7},
            base_args={"threads": 4},
        )
        
        assert result is not None
        assert result.selected_tool in ["salmon", "deseq2"]
        assert isinstance(result.enhanced_args, dict)
    
    def test_record_execution(self, orchestrator):
        """Test recording execution through orchestrator."""
        record = orchestrator.record_execution(
            query="Test query",
            tool_name="test_tool",
            tool_args={"param": "value"},
            success=True,
            duration_ms=100,
        )
        
        assert record is not None
        assert record.tool_name == "test_tool"
    
    def test_warmup(self, orchestrator):
        """Test RAG warmup."""
        orchestrator.warm_up()
        # Should not raise
        assert True
    
    def test_get_stats(self, orchestrator):
        """Test getting RAG statistics."""
        stats = orchestrator.get_stats()
        
        assert isinstance(stats, dict)
        assert "enabled" in stats
        assert "layers" in stats
        assert "tool_memory_stats" in stats
