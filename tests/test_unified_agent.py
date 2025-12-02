"""
Tests for the UnifiedAgent
==========================

Tests the unified agent that combines:
- AutonomousAgent-style orchestration (task classification, permissions)
- AgentTools execution (actual tool implementations)
- Executor layer (sandbox, audit, permissions)
"""

import pytest
import asyncio
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from workflow_composer.agents.unified_agent import (
    UnifiedAgent,
    AgentResponse,
    TaskType,
    ResponseType,
    ToolExecution,
    classify_task,
    TASK_KEYWORDS,
    TOOL_PERMISSION_MAPPING,
    LEVEL_PERMISSIONS,
    get_agent,
    reset_agent,
)
from workflow_composer.agents.executor import AutonomyLevel
from workflow_composer.agents.tools import ToolName


# =============================================================================
# Task Classification Tests
# =============================================================================

class TestTaskClassification:
    """Tests for classify_task function."""
    
    def test_workflow_classification(self):
        """Test that workflow-related queries are classified correctly."""
        queries = [
            "generate an RNA-seq workflow",
            "create a ChIP-seq pipeline",
            "run nextflow for ATAC-seq",
            "I need a variant calling pipeline",
        ]
        for query in queries:
            result = classify_task(query)
            assert result == TaskType.WORKFLOW, f"Failed for: {query}"
    
    def test_diagnosis_classification(self):
        """Test that error-related queries are classified correctly."""
        queries = [
            "why did my job fail",
            "diagnose this error",
            "fix this crash",
            "my job is not working",
        ]
        for query in queries:
            result = classify_task(query)
            assert result == TaskType.DIAGNOSIS, f"Failed for: {query}"
    
    def test_data_classification(self):
        """Test that data-related queries are classified correctly."""
        queries = [
            "scan /data/raw for FASTQ files",
            "find all BAM files",
            "search TCGA for lung cancer samples",
            "download the reference genome",
        ]
        for query in queries:
            result = classify_task(query)
            assert result == TaskType.DATA, f"Failed for: {query}"
    
    def test_job_classification(self):
        """Test that job-related queries are classified correctly."""
        queries = [
            "what jobs are running",
            "check SLURM queue",
            "cancel job 12345",
            "show job status",
        ]
        for query in queries:
            result = classify_task(query)
            assert result == TaskType.JOB, f"Failed for: {query}"
    
    def test_education_classification(self):
        """Test that educational queries are classified correctly."""
        queries = [
            "explain what RNA-seq is",
            "what is variant calling",
            "how does ChIP-seq work",
            "help me understand peak calling",
        ]
        for query in queries:
            result = classify_task(query)
            assert result == TaskType.EDUCATION, f"Failed for: {query}"
    
    def test_system_classification(self):
        """Test that system-related queries are classified correctly."""
        queries = [
            "check system health",
            "restart vLLM server",
            "is the GPU available",
        ]
        for query in queries:
            result = classify_task(query)
            assert result == TaskType.SYSTEM, f"Failed for: {query}"
    
    def test_general_classification(self):
        """Test that unknown queries are classified as general."""
        queries = [
            "hello",
            "what time is it",
            "random query with no keywords",
        ]
        for query in queries:
            result = classify_task(query)
            assert result == TaskType.GENERAL, f"Failed for: {query}"


# =============================================================================
# Permission Tests
# =============================================================================

class TestPermissions:
    """Tests for permission checking."""
    
    def test_readonly_permissions(self):
        """Test that READONLY level only allows read operations."""
        agent = UnifiedAgent(autonomy_level=AutonomyLevel.READONLY)
        
        # Should be allowed
        perm = agent.check_tool_permission(ToolName.SCAN_DATA)
        assert perm["allowed"] is True
        assert perm["requires_approval"] is False
        
        # Should not be allowed
        perm = agent.check_tool_permission(ToolName.SUBMIT_JOB)
        assert perm["allowed"] is False
    
    def test_assisted_permissions(self):
        """Test that ASSISTED level allows read/write but needs approval for execute."""
        agent = UnifiedAgent(autonomy_level=AutonomyLevel.ASSISTED)
        
        # Read should be allowed
        perm = agent.check_tool_permission(ToolName.SCAN_DATA)
        assert perm["allowed"] is True
        
        # Write should be allowed
        perm = agent.check_tool_permission(ToolName.DOWNLOAD_DATASET)
        assert perm["allowed"] is True
        
        # Execute needs approval
        perm = agent.check_tool_permission(ToolName.SUBMIT_JOB)
        assert perm["allowed"] is True
        assert perm["requires_approval"] is True
    
    def test_autonomous_permissions(self):
        """Test that AUTONOMOUS level allows everything."""
        agent = UnifiedAgent(autonomy_level=AutonomyLevel.AUTONOMOUS)
        
        # All should be allowed
        for tool_name in ToolName:
            perm = agent.check_tool_permission(tool_name)
            assert perm["allowed"] is True, f"Failed for: {tool_name}"
    
    def test_permission_mapping_coverage(self):
        """Test that all tools have permission mappings."""
        for tool_name in ToolName:
            assert tool_name in TOOL_PERMISSION_MAPPING, \
                f"Tool {tool_name} missing from permission mapping"


# =============================================================================
# Tool Execution Tests
# =============================================================================

class TestToolExecution:
    """Tests for tool execution with permissions."""
    
    def test_execute_allowed_tool(self):
        """Test executing an allowed tool."""
        agent = UnifiedAgent(autonomy_level=AutonomyLevel.AUTONOMOUS)
        
        # Execute a safe tool (SHOW_HELP is the correct enum)
        execution = agent.execute_tool(ToolName.SHOW_HELP)
        
        assert isinstance(execution, ToolExecution)
        assert execution.tool_name == "show_help"
        assert execution.result is not None
    
    def test_execute_blocked_tool(self):
        """Test that blocked tools return error."""
        agent = UnifiedAgent(autonomy_level=AutonomyLevel.READONLY)
        
        # Try to execute a write tool
        execution = agent.execute_tool(ToolName.SUBMIT_JOB, workflow_dir="/test")
        
        assert execution.result.success is False
        assert "not allowed" in execution.result.error.lower()
    
    def test_execute_unknown_tool(self):
        """Test executing an unknown tool returns error."""
        agent = UnifiedAgent(autonomy_level=AutonomyLevel.AUTONOMOUS)
        
        execution = agent.execute_tool("nonexistent_tool")
        
        assert execution.result.success is False
        assert "unknown" in execution.result.error.lower()


# =============================================================================
# Query Processing Tests
# =============================================================================

class TestQueryProcessing:
    """Tests for query processing."""
    
    @pytest.mark.asyncio
    async def test_process_simple_query(self):
        """Test processing a simple query."""
        agent = UnifiedAgent(autonomy_level=AutonomyLevel.AUTONOMOUS)
        
        response = await agent.process_query("get help")
        
        assert isinstance(response, AgentResponse)
        assert response.task_type in [TaskType.EDUCATION, TaskType.GENERAL]
    
    def test_process_sync(self):
        """Test synchronous query processing."""
        agent = UnifiedAgent(autonomy_level=AutonomyLevel.AUTONOMOUS)
        
        response = agent.process_sync("what is RNA-seq")
        
        assert isinstance(response, AgentResponse)
        assert response.task_type == TaskType.EDUCATION
    
    @pytest.mark.asyncio
    async def test_process_with_tool_detection(self):
        """Test query processing with tool detection."""
        agent = UnifiedAgent(autonomy_level=AutonomyLevel.AUTONOMOUS)
        
        # Use explicit command that won't trigger clarification
        response = await agent.process_query("show help")
        
        assert response.task_type == TaskType.EDUCATION
        assert len(response.tool_executions) > 0
    
    @pytest.mark.asyncio
    async def test_process_requires_approval(self):
        """Test query processing that requires approval."""
        agent = UnifiedAgent(autonomy_level=AutonomyLevel.ASSISTED)
        
        response = await agent.process_query("submit job in /test/workflow")
        
        assert response.response_type == ResponseType.NEEDS_APPROVAL
        assert response.requires_approval is True
        assert response.approval_request is not None


# =============================================================================
# Response Building Tests
# =============================================================================

class TestResponseBuilding:
    """Tests for AgentResponse building."""
    
    def test_response_to_dict(self):
        """Test response serialization."""
        response = AgentResponse(
            success=True,
            message="Test message",
            response_type=ResponseType.SUCCESS,
            task_type=TaskType.DATA,
            suggestions=["suggestion 1"],
        )
        
        d = response.to_dict()
        
        assert d["success"] is True
        assert d["message"] == "Test message"
        assert d["response_type"] == "success"
        assert d["task_type"] == "data"
    
    def test_tool_execution_record(self):
        """Test ToolExecution record."""
        from workflow_composer.agents.tools import ToolResult
        
        execution = ToolExecution(
            tool_name="test_tool",
            parameters={"param1": "value1"},
            result=ToolResult(
                success=True, 
                tool_name="test_tool",
                data={"key": "value"}
            ),
            duration_ms=100.5,
        )
        
        assert execution.tool_name == "test_tool"
        assert execution.duration_ms == 100.5
        assert execution.result.success is True


# =============================================================================
# Factory Function Tests
# =============================================================================

class TestFactoryFunctions:
    """Tests for factory functions."""
    
    def test_get_agent_singleton(self):
        """Test that get_agent returns a singleton."""
        reset_agent()  # Reset first
        
        agent1 = get_agent(AutonomyLevel.ASSISTED)
        agent2 = get_agent()  # Should return same instance
        
        assert agent1 is agent2
        
        reset_agent()  # Cleanup
    
    def test_reset_agent(self):
        """Test that reset_agent clears the singleton."""
        agent1 = get_agent()
        reset_agent()
        agent2 = get_agent()
        
        assert agent1 is not agent2
        
        reset_agent()  # Cleanup


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for the unified agent."""
    
    def test_full_workflow_query(self):
        """Test a complete workflow query."""
        agent = UnifiedAgent(autonomy_level=AutonomyLevel.SUPERVISED)
        
        response = agent.process_sync("explain what RNA-seq is")
        
        assert response.success is True
        assert response.task_type == TaskType.EDUCATION
        assert len(response.message) > 0
    
    def test_history_tracking(self):
        """Test that history is tracked correctly."""
        agent = UnifiedAgent(autonomy_level=AutonomyLevel.AUTONOMOUS)
        
        # Use explicit commands that won't trigger clarification
        agent.process_sync("show help")
        agent.process_sync("list jobs")
        
        history = agent.get_history(limit=5)
        
        assert len(history) == 2
        assert all("success" in h for h in history)
    
    def test_history_clearing(self):
        """Test that history can be cleared."""
        agent = UnifiedAgent(autonomy_level=AutonomyLevel.AUTONOMOUS)
        
        agent.process_sync("get help")
        agent.clear_history()
        
        history = agent.get_history()
        assert len(history) == 0
    
    def test_autonomy_level_change(self):
        """Test changing autonomy level."""
        agent = UnifiedAgent(autonomy_level=AutonomyLevel.READONLY)
        
        # Should be blocked initially
        perm = agent.check_tool_permission(ToolName.SUBMIT_JOB)
        assert perm["allowed"] is False
        
        # Change level
        agent.set_autonomy_level(AutonomyLevel.AUTONOMOUS)
        
        # Should be allowed now
        perm = agent.check_tool_permission(ToolName.SUBMIT_JOB)
        assert perm["allowed"] is True


# =============================================================================
# Approval Workflow Tests
# =============================================================================

class TestApprovalWorkflow:
    """Tests for the human-in-the-loop approval workflow."""
    
    def test_get_pending_approvals_empty(self):
        """Test getting pending approvals when empty."""
        agent = UnifiedAgent(autonomy_level=AutonomyLevel.ASSISTED)
        
        approvals = agent.get_pending_approvals()
        
        assert isinstance(approvals, list)
    
    def test_approve_action(self):
        """Test approving an action."""
        agent = UnifiedAgent(autonomy_level=AutonomyLevel.ASSISTED)
        
        # Create a pending approval
        request = agent.permissions.request_approval(
            action="test_action",
            description="Test approval",
        )
        
        # Approve it
        result = agent.approve_action(request.id, approver="test_user")
        
        assert result is True
        assert agent.permissions.is_approved(request.id) is True
    
    def test_deny_action(self):
        """Test denying an action."""
        agent = UnifiedAgent(autonomy_level=AutonomyLevel.ASSISTED)
        
        # Create a pending approval
        request = agent.permissions.request_approval(
            action="test_action",
            description="Test denial",
        )
        
        # Deny it
        result = agent.deny_action(request.id, denier="test_user")
        
        assert result is True
        assert agent.permissions.is_approved(request.id) is False


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
