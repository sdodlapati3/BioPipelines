#!/usr/bin/env python3
"""
Test the Agentic Router System
==============================

Tests the new LLM-based routing and tool integration.

Usage:
    python -m pytest tests/test_agentic_router.py -v
    
Or run directly:
    python tests/test_agentic_router.py
"""

import asyncio
import pytest
from unittest.mock import Mock, patch, AsyncMock
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from workflow_composer.agents.router import (
    AgentRouter,
    RouteResult,
    RoutingStrategy,
    AGENT_TOOLS,
)
from workflow_composer.agents.bridge import AgentBridge
from workflow_composer.agents.tools import AgentTools, ToolName


class TestAgentRouter:
    """Tests for the AgentRouter class."""
    
    def test_tool_definitions_valid(self):
        """Verify tool definitions are valid OpenAI format."""
        assert len(AGENT_TOOLS) > 0
        
        for tool in AGENT_TOOLS:
            assert tool["type"] == "function"
            assert "function" in tool
            func = tool["function"]
            assert "name" in func
            assert "description" in func
            assert "parameters" in func
            assert func["parameters"]["type"] == "object"
    
    def test_router_initialization(self):
        """Test router initializes correctly."""
        router = AgentRouter(
            use_local=False,
            use_cloud=False,
            use_regex_fallback=True
        )
        
        status = router.get_status()
        assert status["fallback_enabled"] is True
        assert status["local_available"] is False
    
    @pytest.mark.asyncio
    async def test_regex_fallback_scan_data(self):
        """Test regex fallback for scan_data."""
        router = AgentRouter(
            use_local=False,
            use_cloud=False,
            use_regex_fallback=True
        )
        
        result = await router.route("scan data in /scratch/mydata")
        
        assert result.tool == "scan_data"
        assert result.arguments.get("path") == "/scratch/mydata"
        assert result.strategy == RoutingStrategy.REGEX_FALLBACK
    
    @pytest.mark.asyncio
    async def test_regex_fallback_search_databases(self):
        """Test regex fallback for search_databases."""
        router = AgentRouter(
            use_local=False,
            use_cloud=False,
            use_regex_fallback=True
        )
        
        result = await router.route("search for human RNA-seq data in ENCODE")
        
        assert result.tool == "search_databases"
        assert result.strategy == RoutingStrategy.REGEX_FALLBACK
    
    @pytest.mark.asyncio
    async def test_regex_fallback_workflow_generation(self):
        """Test regex fallback for workflow generation."""
        router = AgentRouter(
            use_local=False,
            use_cloud=False,
            use_regex_fallback=True
        )
        
        result = await router.route("generate workflow for RNA-seq analysis")
        
        assert result.tool == "generate_workflow"
        assert result.requires_generation is True
        assert result.arguments.get("analysis_type") == "rna_seq"
    
    @pytest.mark.asyncio
    async def test_regex_fallback_job_status(self):
        """Test regex fallback for job status."""
        router = AgentRouter(
            use_local=False,
            use_cloud=False,
            use_regex_fallback=True
        )
        
        result = await router.route("what's the status of my job?")
        
        assert result.tool == "get_job_status"
    
    @pytest.mark.asyncio
    async def test_regex_fallback_no_match(self):
        """Test regex fallback when no pattern matches."""
        router = AgentRouter(
            use_local=False,
            use_cloud=False,
            use_regex_fallback=True
        )
        
        result = await router.route("hello world")
        
        assert result.tool is None
        assert result.confidence == 0.0


class TestAgentBridge:
    """Tests for the AgentBridge integration class."""
    
    def test_bridge_initialization(self):
        """Test bridge initializes correctly."""
        bridge = AgentBridge(use_llm_routing=False)
        
        status = bridge.get_status()
        assert status["llm_routing_enabled"] is False
        assert len(status["tools_available"]) > 0
    
    @pytest.mark.asyncio
    async def test_bridge_scan_data(self):
        """Test bridge processes scan_data correctly."""
        bridge = AgentBridge(use_llm_routing=False)
        
        # scan_data handles missing paths gracefully, no mock needed
        result = await bridge.process_message("scan data in /nonexistent")
        
        assert result is not None
        assert result.get("strategy") == "regex_fallback"
        assert result.get("tool_result") is not None
    
    @pytest.mark.asyncio
    async def test_bridge_search_databases(self):
        """Test bridge processes search_databases correctly."""
        bridge = AgentBridge(use_llm_routing=False)
        
        result = await bridge.process_message("search for human ChIP-seq data")
        
        assert result is not None
        tool_result = result.get("tool_result")
        assert tool_result is not None
        assert tool_result.tool_name == "search_databases"
    
    @pytest.mark.asyncio
    async def test_bridge_no_tool(self):
        """Test bridge returns None for non-tool messages."""
        bridge = AgentBridge(use_llm_routing=False)
        
        result = await bridge.process_message("hello, how are you?")
        
        assert result is None
    
    def test_bridge_sync_wrapper(self):
        """Test synchronous wrapper works."""
        bridge = AgentBridge(use_llm_routing=False)
        
        result = bridge.process_message_sync("scan /tmp/test")
        
        assert result is not None
        assert result.get("tool_result") is not None


class TestAgentTools:
    """Tests for the AgentTools class."""
    
    def test_tool_detection_scan_data(self):
        """Test scan_data pattern matching."""
        tools = AgentTools()
        
        test_cases = [
            ("scan data in /path/to/data", "/path/to/data"),
            ("find files in ~/mydata", "~/mydata"),
            ("check /scratch/samples", "/scratch/samples"),
            ("what data is available in /home/user/data", "/home/user/data"),
        ]
        
        for message, expected_path in test_cases:
            result = tools.detect_tool(message)
            assert result is not None, f"Failed to detect: {message}"
            tool_name, args = result
            assert tool_name == ToolName.SCAN_DATA, f"Wrong tool for: {message}"
            assert expected_path in args[0] or args[0].endswith(expected_path.rstrip('/')), \
                f"Wrong path for '{message}': got {args}"
    
    def test_tool_detection_search(self):
        """Test search pattern matching."""
        tools = AgentTools()
        
        test_cases = [
            "search for human RNA-seq",
            "search ENCODE for ChIP-seq data",
            "query GEO for mouse samples",
        ]
        
        for message in test_cases:
            result = tools.detect_tool(message)
            assert result is not None, f"Failed to detect: {message}"
            tool_name, _ = result
            assert tool_name == ToolName.SEARCH_DATABASES
    
    def test_tool_detection_references(self):
        """Test reference check pattern matching."""
        tools = AgentTools()
        
        result = tools.detect_tool("check reference for human")
        assert result is not None
        tool_name, args = result
        assert tool_name == ToolName.CHECK_REFERENCES
    
    def test_tool_detection_job_commands(self):
        """Test job command pattern matching."""
        tools = AgentTools()
        
        test_cases = [
            ("show status", ToolName.GET_JOB_STATUS),
            ("what's the status?", ToolName.GET_JOB_STATUS),
            ("show logs", ToolName.GET_LOGS),
            ("cancel job 123", ToolName.CANCEL_JOB),
            ("diagnose the error", ToolName.DIAGNOSE_ERROR),
        ]
        
        for message, expected_tool in test_cases:
            result = tools.detect_tool(message)
            assert result is not None, f"Failed to detect: {message}"
            tool_name, _ = result
            assert tool_name == expected_tool, f"Wrong tool for: {message}"
    
    def test_show_help(self):
        """Test help command returns valid content."""
        tools = AgentTools()
        
        result = tools.show_help()
        assert result.success is True
        assert "Data Discovery" in result.message
        assert "Workflow Generation" in result.message


class TestContextIntegration:
    """Tests for context-aware routing."""
    
    @pytest.mark.asyncio
    async def test_router_with_context(self):
        """Test router uses context in prompts."""
        router = AgentRouter(
            use_local=False,
            use_cloud=False,
            use_regex_fallback=True
        )
        
        context = {
            "data_loaded": True,
            "sample_count": 10,
            "data_path": "/scratch/samples",
            "last_workflow": "rna_seq_20240101"
        }
        
        result = await router.route("run it on SLURM", context)
        
        # Should detect submit_job
        assert result.tool == "submit_job"


# Run tests if executed directly
if __name__ == "__main__":
    print("=" * 60)
    print("🧪 Running Agentic Router Tests")
    print("=" * 60)
    
    # Run synchronous tests
    test_router = TestAgentRouter()
    test_router.test_tool_definitions_valid()
    print("✅ test_tool_definitions_valid")
    test_router.test_router_initialization()
    print("✅ test_router_initialization")
    
    test_tools = TestAgentTools()
    test_tools.test_tool_detection_scan_data()
    print("✅ test_tool_detection_scan_data")
    test_tools.test_tool_detection_search()
    print("✅ test_tool_detection_search")
    test_tools.test_tool_detection_references()
    print("✅ test_tool_detection_references")
    test_tools.test_tool_detection_job_commands()
    print("✅ test_tool_detection_job_commands")
    test_tools.test_show_help()
    print("✅ test_show_help")
    
    test_bridge = TestAgentBridge()
    test_bridge.test_bridge_initialization()
    print("✅ test_bridge_initialization")
    test_bridge.test_bridge_sync_wrapper()
    print("✅ test_bridge_sync_wrapper")
    
    # Run async tests
    async def run_async_tests():
        await test_router.test_regex_fallback_scan_data()
        print("✅ test_regex_fallback_scan_data")
        await test_router.test_regex_fallback_search_databases()
        print("✅ test_regex_fallback_search_databases")
        await test_router.test_regex_fallback_workflow_generation()
        print("✅ test_regex_fallback_workflow_generation")
        await test_router.test_regex_fallback_job_status()
        print("✅ test_regex_fallback_job_status")
        await test_router.test_regex_fallback_no_match()
        print("✅ test_regex_fallback_no_match")
        
        await test_bridge.test_bridge_scan_data()
        print("✅ test_bridge_scan_data")
        await test_bridge.test_bridge_search_databases()
        print("✅ test_bridge_search_databases")
        await test_bridge.test_bridge_no_tool()
        print("✅ test_bridge_no_tool")
        
        test_context = TestContextIntegration()
        await test_context.test_router_with_context()
        print("✅ test_router_with_context")
    
    asyncio.run(run_async_tests())
    
    print()
    print("=" * 60)
    print("✅ All tests passed!")
    print("=" * 60)
