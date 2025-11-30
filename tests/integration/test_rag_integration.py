"""
Integration Tests for RAG System.

Tests the complete RAG pipeline from query to tool selection.
"""

import pytest
from datetime import datetime
from unittest.mock import patch, MagicMock


class TestRAGPipeline:
    """Integration tests for the RAG pipeline."""
    
    @pytest.fixture
    def rag_orchestrator(self):
        """Create a RAG orchestrator instance."""
        from src.workflow_composer.agents.rag.orchestrator import RAGOrchestrator
        from src.workflow_composer.agents.rag.memory import ToolMemory, ToolMemoryConfig
        
        tool_memory = ToolMemory(ToolMemoryConfig(use_database=False))
        return RAGOrchestrator(tool_memory=tool_memory)
    
    def test_end_to_end_query_processing(self, rag_orchestrator):
        """Test processing a query through the entire RAG pipeline."""
        # Warm up the RAG system
        rag_orchestrator.warm_up()
        
        # Enhance a query
        result = rag_orchestrator.enhance(
            query="RNA-seq differential expression for human samples",
            candidate_tools=["salmon", "deseq2"],
        )
        
        assert result is not None
        assert result.selected_tool in ["salmon", "deseq2"]
    
    def test_execution_recording_and_retrieval(self, rag_orchestrator):
        """Test recording and retrieving executions."""
        # Record an execution
        rag_orchestrator.record_execution(
            query="ChIP-seq peak calling for mouse samples",
            tool_name="macs2",
            tool_args={"qvalue": 0.05},
            success=True,
            duration_ms=2500,
        )
        
        # Get stats
        stats = rag_orchestrator.get_stats()
        
        assert stats["tool_memory_stats"]["macs2"].total_executions > 0
    
    def test_rag_learns_from_executions(self, rag_orchestrator):
        """Test that RAG system learns from recorded executions."""
        # Record several successful executions
        for i in range(3):
            rag_orchestrator.record_execution(
                query=f"RNA-seq analysis batch {i}",
                tool_name="salmon",
                tool_args={"threads": 8},
                success=True,
                duration_ms=1000 + i * 100,
            )
        
        # Enhance a similar query - should benefit from past executions
        result = rag_orchestrator.enhance(
            query="RNA-seq gene expression analysis",
            candidate_tools=["salmon", "star"],
        )
        
        assert result is not None


class TestRAGWithWorkflowGenerator:
    """Integration tests for RAG with workflow generation."""
    
    @pytest.mark.skip(reason="Requires query_parser module")
    def test_full_workflow_generation_with_rag(self):
        """Test full workflow generation with RAG integration."""
        pass


class TestRAGMemoryLayers:
    """Integration tests for RAG memory layers."""
    
    def test_tool_memory_integration(self):
        """Test ToolMemory integration."""
        from src.workflow_composer.agents.rag.memory import ToolMemory, ToolMemoryConfig
        
        memory = ToolMemory(ToolMemoryConfig(use_database=False))
        
        # Record multiple executions
        memory.record(
            query="Test query 1",
            tool_name="salmon",
            tool_args={"threads": 8},
            success=True,
            duration_ms=1000,
        )
        memory.record(
            query="Test query 2",
            tool_name="macs2",
            tool_args={"qvalue": 0.05},
            success=True,
            duration_ms=2000,
        )
        
        # Verify stats
        stats = memory.get_all_stats()
        assert len(stats) == 2
    
    def test_arg_memory_integration(self):
        """Test ArgumentMemory integration."""
        from src.workflow_composer.agents.rag.arg_memory import ArgumentMemory
        from src.workflow_composer.agents.rag.memory import ToolMemory, ToolMemoryConfig
        
        tool_memory = ToolMemory(ToolMemoryConfig(use_database=False))
        memory = ArgumentMemory(tool_memory=tool_memory)
        
        # Learn from execution
        memory.learn_from_execution(
            query="RNA-seq for human",
            tool_name="salmon",
            args={"threads": 8, "libtype": "A"},
            success=True,
        )
        memory.learn_from_execution(
            query="RNA-seq for mouse",
            tool_name="salmon",
            args={"threads": 4, "libtype": "ISR"},
            success=True,
        )
        
        # Get common args
        common = memory.get_common_args("salmon")
        
        assert isinstance(common, dict)
    
    def test_tool_selector_integration(self):
        """Test RAGToolSelector integration."""
        from src.workflow_composer.agents.rag.tool_selector import RAGToolSelector
        from src.workflow_composer.agents.rag.memory import ToolMemory, ToolMemoryConfig
        
        tool_memory = ToolMemory(ToolMemoryConfig(use_database=False))
        selector = RAGToolSelector(tool_memory=tool_memory)
        
        # Get tool boost for candidates
        boost = selector.get_tool_boost(
            query="RNA-seq alignment",
            candidate_tools=["salmon", "star", "hisat2"],
        )
        
        assert isinstance(boost.boosts, dict)


class TestRAGCaching:
    """Integration tests for RAG caching."""
    
    def test_cache_integration(self):
        """Test RAG with semantic cache integration."""
        from src.workflow_composer.agents.rag.orchestrator import RAGOrchestrator
        from src.workflow_composer.agents.rag.memory import ToolMemory, ToolMemoryConfig
        from src.workflow_composer.infrastructure.semantic_cache import SemanticCache
        
        tool_memory = ToolMemory(ToolMemoryConfig(use_database=False))
        rag = RAGOrchestrator(tool_memory=tool_memory)
        cache = SemanticCache()
        
        query = "RNA-seq analysis"
        
        # First execution - not cached
        result1 = rag.enhance(query=query, candidate_tools=["salmon", "star"])
        
        # Store in cache
        cache_data = {"selected_tool": result1.selected_tool}
        cache.set(f"rag:{query}", cache_data)
        
        # Second execution - check cache first
        # cache.get() returns tuple (value, cache_hit)
        cached, was_hit = cache.get(f"rag:{query}")
        
        assert was_hit is True
        assert cached is not None
        assert cached["selected_tool"] == result1.selected_tool
