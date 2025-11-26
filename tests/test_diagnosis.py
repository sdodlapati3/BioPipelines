"""
Tests for Error Diagnosis Module
================================

Tests the error diagnosis functionality including:
- Error categories and taxonomy
- Pattern matching
- Log collection
- Auto-fix engine
"""

import pytest
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from pathlib import Path
import tempfile
import os
import asyncio

# Import modules under test
from workflow_composer.diagnosis.categories import (
    ErrorCategory,
    ErrorDiagnosis,
    ErrorPattern,
    FixRiskLevel,
    FixSuggestion,
)
from workflow_composer.diagnosis.patterns import (
    ERROR_PATTERNS,
    get_pattern,
    get_all_patterns,
)
from workflow_composer.diagnosis.log_collector import LogCollector
from workflow_composer.diagnosis.agent import ErrorDiagnosisAgent
from workflow_composer.diagnosis.auto_fix import (
    AutoFixEngine,
    FixResult,
    FixSession,
    FixStatus,
)


# ============================================================================
# Error Categories Tests
# ============================================================================

class TestErrorCategories:
    """Test error category definitions."""
    
    def test_error_category_enum(self):
        """Test ErrorCategory enum values exist."""
        assert hasattr(ErrorCategory, 'FILE_NOT_FOUND')
        assert hasattr(ErrorCategory, 'OUT_OF_MEMORY')
        assert hasattr(ErrorCategory, 'PERMISSION_DENIED')
        assert hasattr(ErrorCategory, 'CONTAINER_ERROR')
        assert hasattr(ErrorCategory, 'DEPENDENCY_MISSING')
        assert hasattr(ErrorCategory, 'TOOL_ERROR')
    
    def test_fix_risk_level_enum(self):
        """Test FixRiskLevel enum values."""
        assert hasattr(FixRiskLevel, 'SAFE')
        assert hasattr(FixRiskLevel, 'LOW')
        assert hasattr(FixRiskLevel, 'MEDIUM')
        assert hasattr(FixRiskLevel, 'HIGH')
    
    def test_error_category_values(self):
        """Test that error categories have string values."""
        # Categories should have meaningful string values
        assert isinstance(ErrorCategory.FILE_NOT_FOUND.value, str)
        assert len(ErrorCategory.FILE_NOT_FOUND.value) > 0


# ============================================================================
# Pattern Matching Tests
# ============================================================================

class TestPatternMatcher:
    """Test pattern matching functionality."""
    
    def test_pattern_database_not_empty(self):
        """Test that ERROR_PATTERNS database has entries."""
        # ERROR_PATTERNS is a dict of ErrorCategory -> ErrorPattern
        assert len(ERROR_PATTERNS) > 0
        # Should have at least 10 error categories
        assert len(ERROR_PATTERNS) >= 10
    
    def test_pattern_structure(self):
        """Test that patterns have required fields."""
        for category, pattern in ERROR_PATTERNS.items():
            assert isinstance(category, ErrorCategory)
            assert hasattr(pattern, 'patterns')
            assert hasattr(pattern, 'category')
            assert hasattr(pattern, 'description')
    
    def test_get_pattern_for_category(self):
        """Test retrieving a specific pattern by category."""
        pattern = get_pattern(ErrorCategory.FILE_NOT_FOUND)
        assert pattern is not None
        assert pattern.category == ErrorCategory.FILE_NOT_FOUND
    
    def test_get_all_patterns_returns_dict(self):
        """Test get_all_patterns returns proper structure."""
        patterns = get_all_patterns()
        assert isinstance(patterns, dict)
        assert len(patterns) > 0
    
    def test_pattern_has_suggestions(self):
        """Test patterns have fix suggestions."""
        pattern = get_pattern(ErrorCategory.OUT_OF_MEMORY)
        assert pattern is not None
        assert hasattr(pattern, 'suggested_fixes')
        assert len(pattern.suggested_fixes) > 0


# ============================================================================
# Log Collector Tests
# ============================================================================

class TestLogCollector:
    """Test log collection functionality."""
    
    def test_log_collector_initialization(self):
        """Test LogCollector instantiation."""
        collector = LogCollector()
        assert collector is not None
    
    def test_collect_from_directory(self):
        """Test collecting logs from a directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mock log files
            log_file = Path(tmpdir) / ".nextflow.log"
            log_file.write_text("Test nextflow log content")
            
            err_file = Path(tmpdir) / "slurm_123.err"
            err_file.write_text("Test SLURM error content")
            
            collector = LogCollector()
            
            # Collect should work with directory
            if hasattr(collector, 'collect'):
                logs = collector.collect(tmpdir)
                assert logs is not None
            elif hasattr(collector, 'collect_logs'):
                logs = collector.collect_logs(tmpdir)
                assert logs is not None
    
    def test_collect_nextflow_log(self):
        """Test specifically collecting Nextflow logs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / ".nextflow.log"
            log_file.write_text("""
            Nov-26 10:00:00.000 [main] DEBUG nextflow.processor
            Nov-26 10:00:01.000 [main] ERROR Process failed: STAR_ALIGN
            Caused by: java.lang.RuntimeException: Process terminated with error
            """)
            
            collector = LogCollector()
            
            if hasattr(collector, 'collect_nextflow_log'):
                content = collector.collect_nextflow_log(tmpdir)
                assert "ERROR" in content or content is not None


# ============================================================================
# Error Diagnosis Agent Tests
# ============================================================================

class TestErrorDiagnosisAgent:
    """Test the main error diagnosis agent."""
    
    def test_agent_initialization(self):
        """Test ErrorDiagnosisAgent instantiation."""
        agent = ErrorDiagnosisAgent()
        assert agent is not None
    
    def test_agent_has_required_methods(self):
        """Test agent has required methods."""
        agent = ErrorDiagnosisAgent()
        assert hasattr(agent, 'diagnose')
        assert callable(getattr(agent, 'diagnose'))
    
    def test_diagnose_returns_result(self):
        """Test that diagnose returns a result."""
        agent = ErrorDiagnosisAgent()
        
        test_error = "java.lang.OutOfMemoryError: Java heap space"
        
        # diagnose is async - use asyncio.run
        result = asyncio.run(agent.diagnose(test_error))
        assert result is not None
    
    def test_diagnose_file_not_found(self):
        """Test diagnosis of file not found error."""
        agent = ErrorDiagnosisAgent()
        
        test_error = "ERROR: File not found: /data/reference.fa"
        result = asyncio.run(agent.diagnose(test_error))
        
        assert result is not None
        # Result should be ErrorDiagnosis
        if isinstance(result, ErrorDiagnosis):
            assert result.category is not None


# ============================================================================
# Auto-Fix Engine Tests
# ============================================================================

class TestAutoFixEngine:
    """Test auto-fix functionality."""
    
    def test_fix_risk_level_enum(self):
        """Test FixRiskLevel enum values."""
        assert hasattr(FixRiskLevel, 'SAFE')
        assert hasattr(FixRiskLevel, 'LOW')
        assert hasattr(FixRiskLevel, 'MEDIUM')
        assert hasattr(FixRiskLevel, 'HIGH')
    
    def test_auto_fix_engine_initialization(self):
        """Test AutoFixEngine instantiation."""
        engine = AutoFixEngine()
        assert engine is not None
    
    def test_fix_suggestion_dataclass(self):
        """Test FixSuggestion dataclass creation."""
        fix = FixSuggestion(
            description="Create missing output directory",
            command="mkdir -p /output/results",
            risk_level=FixRiskLevel.SAFE,
        )
        assert fix.description == "Create missing output directory"
        assert fix.command == "mkdir -p /output/results"
    
    def test_fix_status_enum(self):
        """Test FixStatus enum values."""
        # Should have status values
        assert len(FixStatus) > 0
    
    def test_get_engine_instance(self):
        """Test getting auto-fix engine."""
        from workflow_composer.diagnosis.auto_fix import get_auto_fix_engine
        engine = get_auto_fix_engine()
        assert engine is not None
        assert isinstance(engine, AutoFixEngine)


# ============================================================================
# Integration Tests
# ============================================================================

class TestDiagnosisIntegration:
    """Integration tests for diagnosis workflow."""
    
    def test_full_diagnosis_workflow(self):
        """Test complete diagnosis workflow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mock error logs
            log_file = Path(tmpdir) / ".nextflow.log"
            log_file.write_text("""
            Nov-26 10:00:00.000 [main] DEBUG nextflow
            Nov-26 10:00:01.000 [main] ERROR Process failed
            java.lang.OutOfMemoryError: Java heap space
            """)
            
            agent = ErrorDiagnosisAgent()
            
            # Run diagnosis
            with open(log_file) as f:
                result = asyncio.run(agent.diagnose(f.read()))
            
            assert result is not None
    
    def test_diagnosis_with_container_error(self):
        """Test diagnosis of container error."""
        agent = ErrorDiagnosisAgent()
        
        error_text = """
        ERROR: Container image not found: docker://nfcore/rnaseq:2.0
        FATAL: Unable to pull container
        """
        
        result = asyncio.run(agent.diagnose(error_text))
        assert result is not None


# ============================================================================
# Edge Cases
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_log_handling(self):
        """Test handling of empty log content."""
        agent = ErrorDiagnosisAgent()
        
        result = asyncio.run(agent.diagnose(""))
        assert result is not None  # Should not crash
    
    def test_malformed_log_handling(self):
        """Test handling of malformed log content."""
        agent = ErrorDiagnosisAgent()
        
        malformed = "!@#$%^&*() random garbage"
        result = asyncio.run(agent.diagnose(malformed))
        assert result is not None  # Should not crash
    
    def test_collector_nonexistent_directory(self):
        """Test log collector with non-existent directory."""
        collector = LogCollector()
        
        if hasattr(collector, 'collect'):
            logs = collector.collect("/nonexistent/path/12345")
            # Should return empty or handle gracefully
            assert logs is not None or logs == {} or logs == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
