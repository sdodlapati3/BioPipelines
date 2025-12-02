"""
Tests for Session Memory and Conversation Recovery
===================================================

Tests for the robust agent features:
- Session-wide memory persistence
- Reference resolution ("that path", "the data")
- Conversation recovery (clarification, error handling)
"""

import pytest
from datetime import datetime, timedelta

from src.workflow_composer.agents.intent.session_memory import (
    SessionMemory,
    MemoryEntry,
    ActionRecord,
    MemoryType,
    MemoryPriority,
    get_session_memory,
)
from src.workflow_composer.agents.intent.conversation_recovery import (
    ConversationRecovery,
    RecoveryResponse,
    RecoveryStrategy,
    ErrorCategory,
    ErrorContext,
    get_conversation_recovery,
)


# =============================================================================
# Session Memory Tests
# =============================================================================

class TestSessionMemory:
    """Tests for SessionMemory class."""
    
    def test_initialization(self):
        """Test SessionMemory initializes correctly."""
        memory = SessionMemory(session_id="test_session")
        
        assert memory.session_id == "test_session"
        assert memory.created_at is not None
        assert memory._last_path is None
        assert memory._last_dataset is None
    
    def test_remember_path(self):
        """Test path memory functionality."""
        memory = SessionMemory()
        
        # Remember a path
        memory.remember_path("/data/methylation", context="data")
        
        # Should be retrievable by full path
        assert memory.get_remembered_path() == "/data/methylation"
        
        # Should be retrievable by hint
        assert memory.get_remembered_path("methylation") == "/data/methylation"
        assert memory.get_remembered_path("data") == "/data/methylation"
    
    def test_remember_multiple_paths(self):
        """Test remembering multiple paths."""
        memory = SessionMemory()
        
        memory.remember_path("/data/rna_seq", context="input")
        memory.remember_path("/results/output", context="output")
        
        # Last path is the most recent
        assert memory.get_remembered_path() == "/results/output"
        
        # Can retrieve specific paths by hint
        assert memory.get_remembered_path("rna") == "/data/rna_seq"
        assert memory.get_remembered_path("output") == "/results/output"
    
    def test_remember_dataset(self):
        """Test dataset memory functionality."""
        memory = SessionMemory()
        
        memory.remember_dataset("GSE12345", source="GEO", metadata={"organism": "human"})
        
        assert memory.get_remembered_dataset() == "GSE12345"
        assert memory.get_remembered_dataset("GSE12345") == "GSE12345"
    
    def test_remember_search_results(self):
        """Test search results memory."""
        memory = SessionMemory()
        
        results = [
            {"id": "GSE111", "title": "RNA-seq"},
            {"id": "GSE222", "title": "ChIP-seq"},
        ]
        
        memory.remember_search_results(results, "human RNA-seq")
        
        query, stored_results = memory.get_last_search_results()
        assert query == "human RNA-seq"
        assert stored_results == results
        
        # Datasets should also be remembered
        assert memory.get_remembered_dataset("GSE111") == "GSE111"
    
    def test_preferences(self):
        """Test user preference management."""
        memory = SessionMemory()
        
        memory.set_preference("organism", "human")
        memory.set_preference("assay_type", "RNA-seq")
        
        assert memory.get_preference("organism") == "human"
        assert memory.get_preference("assay_type") == "RNA-seq"
        assert memory.get_preference("unknown") is None
        assert memory.get_preference("unknown", "default") == "default"
    
    def test_action_history(self):
        """Test action history recording."""
        memory = SessionMemory()
        
        memory.record_action(
            action_type="search",
            query="human RNA-seq",
            tool_used="search_databases",
            success=True,
            parameters={"query": "human RNA-seq"},
            result_summary="Found 10 results",
        )
        
        last_action = memory.get_last_action()
        assert last_action is not None
        assert last_action.action_type == "search"
        assert last_action.success is True
        
        # Can filter by type
        last_search = memory.get_last_action("search")
        assert last_search.query == "human RNA-seq"
    
    def test_reference_resolution_simple(self):
        """Test simple reference resolution."""
        memory = SessionMemory()
        
        memory.remember_path("/data/methylation")
        
        # Resolve "that path"
        resolved = memory.resolve_reference("that path")
        assert resolved == "/data/methylation"
        
        # Resolve "the path"
        resolved = memory.resolve_reference("the path")
        assert resolved == "/data/methylation"
    
    def test_reference_resolution_in_query(self):
        """Test reference resolution in full query."""
        memory = SessionMemory()
        
        memory.remember_path("/data/methylation")
        memory.remember_dataset("GSE12345")
        
        # Resolve references in query
        query = "scan that path for FASTQ files"
        resolved_query, refs = memory.resolve_references_in_query(query)
        
        assert "/data/methylation" in resolved_query or "that path" in refs
    
    def test_context_summary(self):
        """Test context summary generation."""
        memory = SessionMemory()
        
        memory.remember_path("/data/rna_seq")
        memory.set_preference("organism", "human")
        
        summary = memory.get_context_summary()
        
        assert "/data/rna_seq" in summary
        assert "human" in summary
    
    def test_forget(self):
        """Test forgetting specific items."""
        memory = SessionMemory()
        
        memory.remember_path("/data/methylation")
        memory.remember_path("/data/rna_seq")
        
        memory.forget("/data/methylation")
        
        # Should not be able to find it anymore
        assert memory.get_remembered_path("methylation") is None
        
        # Other path should still work
        assert memory.get_remembered_path("rna") == "/data/rna_seq"
    
    def test_clear(self):
        """Test clearing all memory."""
        memory = SessionMemory()
        
        memory.remember_path("/data/test")
        memory.remember_dataset("GSE123")
        memory.set_preference("org", "human")
        
        memory.clear()
        
        assert memory.get_remembered_path() is None
        assert memory.get_remembered_dataset() is None
        assert memory.get_preference("org") is None
    
    def test_singleton(self):
        """Test singleton pattern."""
        mem1 = get_session_memory("test")
        mem2 = get_session_memory("test")
        
        assert mem1 is mem2


# =============================================================================
# Conversation Recovery Tests
# =============================================================================

class TestConversationRecovery:
    """Tests for ConversationRecovery class."""
    
    def test_initialization(self):
        """Test ConversationRecovery initializes correctly."""
        recovery = ConversationRecovery()
        
        assert recovery.max_retries == 2
        assert recovery.retry_with_better_model is True
    
    def test_handle_low_confidence_very_low(self):
        """Test handling very low confidence (< 0.25)."""
        recovery = ConversationRecovery()
        
        response = recovery.handle_low_confidence(
            query="xyz abc",
            confidence=0.15,
            detected_intent="DATA_SEARCH",
        )
        
        assert response.strategy == RecoveryStrategy.CLARIFY
        assert response.needs_user_input is True
        assert "rephrase" in response.message.lower() or "clarify" in response.message.lower() or "sure" in response.message.lower()
    
    def test_handle_low_confidence_low(self):
        """Test handling low confidence (0.25-0.4)."""
        recovery = ConversationRecovery()
        
        response = recovery.handle_low_confidence(
            query="search data",
            confidence=0.30,
            detected_intent="DATA_SEARCH",
            alternative_intents=[
                ("DATA_SEARCH", 0.30),
                ("DATA_SCAN", 0.25),
            ],
        )
        
        assert response.strategy == RecoveryStrategy.CLARIFY
        assert response.needs_user_input is True
    
    def test_handle_low_confidence_medium(self):
        """Test handling medium confidence (> 0.4)."""
        recovery = ConversationRecovery()
        
        response = recovery.handle_low_confidence(
            query="search for human RNA-seq",
            confidence=0.50,
            detected_intent="DATA_SEARCH",
        )
        
        # Should not need clarification
        assert response.needs_user_input is False
    
    def test_handle_error_api(self):
        """Test error handling for API errors."""
        recovery = ConversationRecovery()
        
        response = recovery.handle_error(
            error=Exception("Connection refused: ENCODE API"),
            query="search for human data",
            tool_name="search_databases",
            parameters={"query": "human"},
        )
        
        assert response.strategy == RecoveryStrategy.ACKNOWLEDGE_ERROR
        assert len(response.suggestions) > 0
        assert "try" in response.message.lower() or "issue" in response.message.lower()
    
    def test_handle_error_not_found(self):
        """Test error handling for not found errors."""
        recovery = ConversationRecovery()
        
        response = recovery.handle_error(
            error=Exception("File not found: /data/test.fastq"),
            query="scan /data/test.fastq",
            tool_name="scan_data",
            parameters={"path": "/data/test.fastq"},
        )
        
        assert response.strategy == RecoveryStrategy.ACKNOWLEDGE_ERROR
        assert "not found" in response.message.lower() or "couldn't find" in response.message.lower()
    
    def test_handle_error_timeout(self):
        """Test error handling for timeout errors."""
        recovery = ConversationRecovery()
        
        response = recovery.handle_error(
            error=Exception("Request timed out after 60s"),
            query="download large dataset",
            tool_name="download_dataset",
        )
        
        assert "timeout" in str(response.message).lower() or "taking longer" in response.message.lower()
    
    def test_handle_correction(self):
        """Test handling user corrections."""
        recovery = ConversationRecovery()
        
        response = recovery.handle_correction(
            original_intent="DATA_SCAN",
            corrected_intent="DATA_SEARCH",
            query="find data",
        )
        
        assert response.strategy == RecoveryStrategy.SUGGEST
        assert response.needs_user_input is False
        assert "search" in response.message.lower()
    
    def test_get_fallback_response(self):
        """Test fallback response generation."""
        recovery = ConversationRecovery()
        
        response = recovery.get_fallback_response("random gibberish xyz")
        
        assert response.strategy == RecoveryStrategy.FALLBACK_RESPONSE
        assert len(response.suggestions) > 0
        assert "help" in response.message.lower()
    
    def test_error_categorization(self):
        """Test error categorization logic."""
        recovery = ConversationRecovery()
        
        # Timeout
        cat = recovery._categorize_error(Exception("Request timed out"), None)
        assert cat == ErrorCategory.TIMEOUT
        
        # Not found
        cat = recovery._categorize_error(Exception("404 Not Found"), None)
        assert cat == ErrorCategory.RESOURCE_NOT_FOUND
        
        # Permission
        cat = recovery._categorize_error(Exception("403 Forbidden"), None)
        assert cat == ErrorCategory.PERMISSION_ERROR
        
        # Validation
        cat = recovery._categorize_error(Exception("Invalid parameter: xyz"), None)
        assert cat == ErrorCategory.VALIDATION_ERROR
    
    def test_retry_tracking(self):
        """Test retry count tracking."""
        recovery = ConversationRecovery()
        
        recovery.record_retry("test query")
        recovery.record_retry("test query")
        
        assert recovery._retry_counts.get("test query") == 2
        
        recovery.clear_retry_count("test query")
        assert recovery._retry_counts.get("test query") is None
    
    def test_singleton(self):
        """Test singleton pattern."""
        rec1 = get_conversation_recovery()
        rec2 = get_conversation_recovery()
        
        assert rec1 is rec2


# =============================================================================
# Integration Tests
# =============================================================================

class TestSessionMemoryRecoveryIntegration:
    """Tests for integration between SessionMemory and ConversationRecovery."""
    
    def test_memory_context_in_recovery(self):
        """Test that session memory context can be used in recovery."""
        memory = SessionMemory()
        recovery = ConversationRecovery()
        
        # Set up context
        memory.remember_path("/data/rna_seq")
        memory.set_preference("organism", "human")
        
        # Get fallback - could potentially use context
        response = recovery.get_fallback_response("help with my data")
        
        # Recovery should work regardless
        assert response.message is not None
        assert len(response.suggestions) > 0
    
    def test_error_recovery_with_remembered_path(self):
        """Test error recovery when path is remembered."""
        memory = SessionMemory()
        recovery = ConversationRecovery()
        
        memory.remember_path("/data/test")
        
        response = recovery.handle_error(
            error=Exception("File not found"),
            query="scan the data",
            tool_name="scan_data",
            parameters={"path": "/data/test"},
        )
        
        assert response is not None
        assert response.strategy == RecoveryStrategy.ACKNOWLEDGE_ERROR


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
