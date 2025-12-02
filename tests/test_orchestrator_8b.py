"""
Tests for NVIDIA Orchestrator-8B Integration
=============================================

Tests the orchestrator module including:
- Configuration handling
- Tool catalog management
- Routing decisions (heuristic mode)
- Response parsing
- Integration with SupervisorAgent
"""

import pytest
import asyncio
import json
from unittest.mock import Mock, AsyncMock, patch

from workflow_composer.llm.orchestrator_8b import (
    Orchestrator8B,
    OrchestratorConfig,
    OrchestrationResult,
    RoutingDecision,
    ModelTier,
    ToolDefinition,
    BIOPIPELINE_TOOLS,
    get_orchestrator_8b,
    quick_route
)


class TestOrchestratorConfig:
    """Tests for OrchestratorConfig."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = OrchestratorConfig()
        
        assert config.model_name == "nvidia/Orchestrator-8B"
        assert config.inference_backend == "vllm"
        assert config.prefer_local == True
        assert config.max_cost_per_query == 1.0
        assert config.optimize_for == "balanced"
        assert config.max_turns == 10
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = OrchestratorConfig(
            prefer_local=False,
            max_cost_per_query=0.10,
            optimize_for="cost",
            max_turns=5
        )
        
        assert config.prefer_local == False
        assert config.max_cost_per_query == 0.10
        assert config.optimize_for == "cost"
        assert config.max_turns == 5
    
    def test_config_with_custom_tools(self):
        """Test configuration with custom tool catalog."""
        custom_tools = [
            ToolDefinition(
                name="my_tool",
                description="Custom tool",
                parameters={"x": "int"}
            )
        ]
        
        config = OrchestratorConfig(tool_catalog=custom_tools)
        assert len(config.tool_catalog) == 1
        assert config.tool_catalog[0].name == "my_tool"


class TestToolDefinition:
    """Tests for ToolDefinition."""
    
    def test_default_tool(self):
        """Test default tool definition values."""
        tool = ToolDefinition(
            name="test",
            description="Test tool",
            parameters={}
        )
        
        assert tool.tier == ModelTier.LOCAL_SMALL
        assert tool.cost_per_call == 0.0
        assert tool.latency_estimate_ms == 1000
    
    def test_custom_tool(self):
        """Test custom tool definition."""
        tool = ToolDefinition(
            name="expensive_tool",
            description="Expensive operation",
            parameters={"input": "string"},
            tier=ModelTier.CLOUD_LARGE,
            cost_per_call=0.05,
            latency_estimate_ms=5000
        )
        
        assert tool.tier == ModelTier.CLOUD_LARGE
        assert tool.cost_per_call == 0.05
        assert tool.latency_estimate_ms == 5000


class TestBiopipelineTools:
    """Tests for default BioPipeline tool catalog."""
    
    def test_default_tools_exist(self):
        """Test that default tools are defined."""
        assert len(BIOPIPELINE_TOOLS) >= 5
        
        tool_names = [t.name for t in BIOPIPELINE_TOOLS]
        assert "workflow_planner" in tool_names
        assert "code_generator" in tool_names
        assert "code_validator" in tool_names
        assert "nfcore_reference" in tool_names
    
    def test_tool_tiers(self):
        """Test that tools have appropriate tiers."""
        for tool in BIOPIPELINE_TOOLS:
            if "cloud" in tool.name:
                assert tool.tier in [ModelTier.CLOUD_SMALL, ModelTier.CLOUD_LARGE]
            # Most tools should be local
            assert tool.tier in list(ModelTier)


class TestOrchestrator8B:
    """Tests for Orchestrator8B class."""
    
    def test_initialization(self):
        """Test orchestrator initialization."""
        orch = Orchestrator8B()
        
        assert orch.config is not None
        assert len(orch.config.tool_catalog) > 0
        assert orch._model is None  # Not loaded until initialize()
    
    def test_initialization_with_config(self):
        """Test initialization with custom config."""
        config = OrchestratorConfig(
            prefer_local=False,
            max_cost_per_query=0.50
        )
        
        orch = Orchestrator8B(config)
        
        assert orch.config.prefer_local == False
        assert orch.config.max_cost_per_query == 0.50
    
    def test_build_tool_descriptions(self):
        """Test tool description JSON generation."""
        orch = Orchestrator8B()
        
        desc = orch._tool_descriptions
        assert isinstance(desc, str)
        
        # Should be valid JSON
        tools = json.loads(desc)
        assert isinstance(tools, list)
        assert len(tools) > 0
        assert "name" in tools[0]
        assert "description" in tools[0]
    
    def test_build_preference_string(self):
        """Test preference string generation."""
        config = OrchestratorConfig(
            prefer_local=True,
            max_cost_per_query=0.10,
            optimize_for="cost"
        )
        orch = Orchestrator8B(config)
        
        prefs = orch._build_preference_string()
        
        assert "PREFER local" in prefs
        assert "0.10" in prefs
        assert "MINIMIZE cost" in prefs
    
    def test_build_system_prompt(self):
        """Test system prompt generation."""
        orch = Orchestrator8B()
        
        prompt = orch._build_system_prompt()
        
        assert "orchestrator" in prompt.lower()
        assert "BioPipelines" in prompt
        assert "Available Tools" in prompt
        assert "<thinking>" in prompt
        assert "<tool_call>" in prompt
        assert "<answer>" in prompt


class TestRoutingDecision:
    """Tests for routing decision logic."""
    
    def test_simple_query_routing(self):
        """Test routing for simple queries."""
        orch = Orchestrator8B()
        
        decision = orch.get_routing_decision("What is RNA-seq?")
        
        assert decision.target_tier == ModelTier.LOCAL_SMALL
        assert "ollama" in decision.target_model.lower() or "local" in decision.reasoning.lower()
    
    def test_code_query_routing(self):
        """Test routing for code generation queries."""
        orch = Orchestrator8B()
        
        decision = orch.get_routing_decision(
            "Generate a Nextflow workflow for RNA-seq analysis"
        )
        
        # Should use larger model for code generation
        assert decision.target_tier in [ModelTier.LOCAL_LARGE, ModelTier.LOCAL_SMALL]
        assert "code_generator" in decision.tool_calls_planned
    
    def test_complex_query_routing(self):
        """Test routing for complex queries."""
        config = OrchestratorConfig(prefer_local=False)  # Allow cloud
        orch = Orchestrator8B(config)
        
        decision = orch.get_routing_decision(
            "Design a complex multi-step workflow that integrates "
            "ChIP-seq and RNA-seq with differential binding analysis"
        )
        
        # Complex query should route to larger model
        assert decision.target_tier in [ModelTier.LOCAL_LARGE, ModelTier.CLOUD_LARGE]
    
    def test_prefer_local_override(self):
        """Test that prefer_local overrides cloud routing."""
        config = OrchestratorConfig(prefer_local=True)
        orch = Orchestrator8B(config)
        
        decision = orch.get_routing_decision(
            "Complex workflow with optimization and integration"
        )
        
        # Should stay local even for complex query
        assert decision.target_tier in [ModelTier.LOCAL_SMALL, ModelTier.LOCAL_LARGE]
    
    def test_cost_estimation(self):
        """Test cost estimation in routing decision."""
        orch = Orchestrator8B()
        
        decision = orch.get_routing_decision("Generate RNA-seq workflow")
        
        assert decision.estimated_cost >= 0
        assert decision.confidence > 0


class TestResponseParsing:
    """Tests for response parsing."""
    
    def test_parse_thinking(self):
        """Test parsing thinking block."""
        orch = Orchestrator8B()
        
        response = """
        <thinking>
        The user wants a simple RNA-seq workflow.
        I should use the workflow_planner tool first.
        </thinking>
        
        <tool_call>
        {"name": "workflow_planner", "parameters": {"query": "RNA-seq"}}
        </tool_call>
        """
        
        thinking, tool_call, answer = orch._parse_response(response)
        
        assert thinking is not None
        assert "RNA-seq" in thinking
        assert tool_call is not None
        assert tool_call["name"] == "workflow_planner"
        assert answer is None
    
    def test_parse_answer(self):
        """Test parsing answer block."""
        orch = Orchestrator8B()
        
        response = """
        <thinking>
        Task complete, providing final answer.
        </thinking>
        
        <answer>
        Here is your RNA-seq workflow...
        </answer>
        """
        
        thinking, tool_call, answer = orch._parse_response(response)
        
        assert thinking is not None
        assert tool_call is None
        assert answer is not None
        assert "RNA-seq workflow" in answer
    
    def test_parse_invalid_json(self):
        """Test handling of invalid JSON in tool call."""
        orch = Orchestrator8B()
        
        response = """
        <tool_call>
        {invalid json here}
        </tool_call>
        """
        
        thinking, tool_call, answer = orch._parse_response(response)
        
        # Should gracefully handle invalid JSON
        assert tool_call is None


class TestFactoryFunctions:
    """Tests for factory functions."""
    
    def test_get_orchestrator_8b(self):
        """Test get_orchestrator_8b factory."""
        orch = get_orchestrator_8b(
            prefer_local=False,
            max_cost=0.25,
            backend="vllm"
        )
        
        assert isinstance(orch, Orchestrator8B)
        assert orch.config.prefer_local == False
        assert orch.config.max_cost_per_query == 0.25
        assert orch.config.inference_backend == "vllm"
    
    @pytest.mark.asyncio
    async def test_quick_route(self):
        """Test quick_route helper."""
        result = await quick_route("Generate RNA-seq workflow")
        
        assert isinstance(result, OrchestrationResult)
        assert result.success == True
        assert result.model_used is not None


class TestOrchestratorIntegration:
    """Integration tests for orchestrator."""
    
    @pytest.mark.asyncio
    async def test_orchestrator_without_model(self):
        """Test orchestrator behavior when model not loaded."""
        orch = Orchestrator8B()
        # Don't call initialize()
        
        # Should still be able to get routing decisions
        decision = orch.get_routing_decision("Test query")
        assert decision is not None
        assert decision.confidence < 0.9  # Lower confidence without model
    
    def test_model_tier_ordering(self):
        """Test that model tiers have correct ordering."""
        # This matters for tier comparison logic
        assert ModelTier.LOCAL_SMALL.value == "local_small"
        assert ModelTier.LOCAL_LARGE.value == "local_large"
        assert ModelTier.CLOUD_SMALL.value == "cloud_small"
        assert ModelTier.CLOUD_LARGE.value == "cloud_large"


# === Tests for OrchestratedSupervisor ===

class TestOrchestratedSupervisor:
    """Tests for OrchestratedSupervisor."""
    
    def test_import(self):
        """Test that OrchestratedSupervisor can be imported."""
        from workflow_composer.agents.specialists.orchestrated_supervisor import (
            OrchestratedSupervisor,
            OrchestratedResult,
            get_supervisor
        )
        
        assert OrchestratedSupervisor is not None
        assert OrchestratedResult is not None
        assert get_supervisor is not None
    
    def test_initialization(self):
        """Test OrchestratedSupervisor initialization."""
        from workflow_composer.agents.specialists.orchestrated_supervisor import (
            OrchestratedSupervisor
        )
        
        supervisor = OrchestratedSupervisor(
            use_orchestrator=True,
            prefer_local=True,
            max_cost=0.50
        )
        
        assert supervisor.use_orchestrator == True
        assert supervisor._orch_config.prefer_local == True
        assert supervisor._orch_config.max_cost_per_query == 0.50
    
    def test_initialization_without_orchestrator(self):
        """Test initialization with orchestrator disabled."""
        from workflow_composer.agents.specialists.orchestrated_supervisor import (
            OrchestratedSupervisor
        )
        
        supervisor = OrchestratedSupervisor(use_orchestrator=False)
        
        assert supervisor.use_orchestrator == False
        assert supervisor._orchestrator is None
    
    def test_factory_function(self):
        """Test get_supervisor factory."""
        from workflow_composer.agents.specialists.orchestrated_supervisor import (
            get_supervisor
        )
        
        supervisor = get_supervisor(
            use_orchestrator=True,
            prefer_local=False,
            max_cost=1.0
        )
        
        assert supervisor.use_orchestrator == True
        assert supervisor._orch_config.prefer_local == False


class TestOrchestratedResultMetadata:
    """Tests for OrchestratedResult metadata."""
    
    def test_metadata_fields(self):
        """Test that metadata contains expected fields."""
        from workflow_composer.agents.specialists.orchestrated_supervisor import (
            OrchestratedResult
        )
        
        result = OrchestratedResult(
            success=True,
            metadata={
                "routing": "orchestrator-8b",
                "cost": 0.05,
                "models_used": ["gpt-4", "codellama"],
                "turns": 3
            }
        )
        
        assert result.metadata["routing"] == "orchestrator-8b"
        assert result.metadata["cost"] == 0.05
        assert len(result.metadata["models_used"]) == 2
        assert result.metadata["turns"] == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
