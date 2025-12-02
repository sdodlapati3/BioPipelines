"""
NVIDIA Orchestrator-8B Integration
==================================

Integrates NVIDIA's ToolOrchestra-trained 8B model as an intelligent
orchestration layer for BioPipelines.

The model excels at:
1. Deciding WHICH tools/models to use for a given task
2. Cost-efficient routing (uses smaller models when appropriate)
3. Multi-turn reasoning with tool calls
4. User preference alignment

Architecture:
                     ┌─────────────────────────┐
                     │   User Query            │
                     └───────────┬─────────────┘
                                 │
                     ┌───────────▼─────────────┐
                     │   Orchestrator-8B       │
                     │   (nvidia/Orchestrator) │
                     └───────────┬─────────────┘
                                 │
           ┌─────────────────────┼─────────────────────┐
           │                     │                     │
    ┌──────▼──────┐      ┌───────▼───────┐     ┌──────▼──────┐
    │  Local LLM  │      │  Cloud LLM    │     │  Specialist │
    │  (Ollama)   │      │  (GPT-4/Claude)│    │  (CodeLlama)│
    └─────────────┘      └───────────────┘     └─────────────┘

Reference: https://arxiv.org/abs/2511.21689

Example:
    from workflow_composer.llm import Orchestrator8B, OrchestratorConfig
    
    # Initialize with available tools
    orch = Orchestrator8B(
        config=OrchestratorConfig(
            prefer_local=True,
            max_cost_per_query=0.10,
            tool_catalog=bioinformatics_tools
        )
    )
    
    # Let orchestrator decide routing
    result = await orch.route_and_execute(
        query="Generate a ChIP-seq workflow with MACS2 peak calling",
        available_models=["gpt-4", "claude-3", "codellama", "ollama/llama3"]
    )
    
    # Check what the orchestrator decided
    print(f"Used model: {result.model_used}")
    print(f"Cost: ${result.cost:.4f}")
    print(f"Reasoning: {result.orchestrator_reasoning}")
"""

import os
import json
import logging
import asyncio
from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import time

logger = logging.getLogger(__name__)


class ModelTier(Enum):
    """Model capability tiers for routing decisions."""
    LOCAL_SMALL = "local_small"      # Ollama 7B, local Llama
    LOCAL_LARGE = "local_large"      # Local 70B, vLLM large
    CLOUD_SMALL = "cloud_small"      # GPT-3.5, Claude Haiku
    CLOUD_LARGE = "cloud_large"      # GPT-4, Claude Opus
    SPECIALIST = "specialist"        # CodeLlama, domain-specific


@dataclass
class ToolDefinition:
    """Definition of a tool available to the orchestrator."""
    name: str
    description: str
    parameters: Dict[str, Any]
    tier: ModelTier = ModelTier.LOCAL_SMALL
    cost_per_call: float = 0.0
    latency_estimate_ms: int = 1000


@dataclass
class OrchestratorConfig:
    """Configuration for Orchestrator-8B."""
    # Model settings
    model_name: str = "nvidia/Orchestrator-8B"
    inference_backend: str = "vllm"  # "vllm", "transformers", "api"
    api_endpoint: Optional[str] = None
    
    # User preferences (as in ToolOrchestra paper)
    prefer_local: bool = True
    max_cost_per_query: float = 1.0
    optimize_for: str = "balanced"  # "cost", "speed", "accuracy", "balanced"
    
    # Tool catalog
    tool_catalog: List[ToolDefinition] = field(default_factory=list)
    
    # Model tiers available
    available_tiers: List[ModelTier] = field(default_factory=lambda: [
        ModelTier.LOCAL_SMALL,
        ModelTier.CLOUD_LARGE
    ])
    
    # Inference settings
    max_turns: int = 10
    temperature: float = 0.0
    max_tokens: int = 8000


@dataclass
class OrchestrationResult:
    """Result from orchestrator routing decision."""
    success: bool
    model_used: str
    tier_used: ModelTier
    response: str
    orchestrator_reasoning: str
    tool_calls: List[Dict[str, Any]]
    cost: float
    latency_ms: float
    turns: int


@dataclass
class RoutingDecision:
    """Orchestrator's routing decision."""
    target_model: str
    target_tier: ModelTier
    reasoning: str
    tool_calls_planned: List[str]
    estimated_cost: float
    confidence: float


# BioPipelines-specific tool catalog
BIOPIPELINE_TOOLS = [
    ToolDefinition(
        name="workflow_planner",
        description="Plans bioinformatics workflows by analyzing query intent and selecting appropriate tools. Use for initial workflow design.",
        parameters={"query": "string", "analysis_type": "string"},
        tier=ModelTier.LOCAL_SMALL,
        cost_per_call=0.001,
        latency_estimate_ms=500
    ),
    ToolDefinition(
        name="code_generator",
        description="Generates Nextflow/Snakemake code from workflow plans. Requires structured plan input.",
        parameters={"plan": "WorkflowPlan", "format": "string"},
        tier=ModelTier.LOCAL_LARGE,  # Benefits from larger context
        cost_per_call=0.01,
        latency_estimate_ms=2000
    ),
    ToolDefinition(
        name="code_validator",
        description="Validates generated workflow code for syntax and best practices.",
        parameters={"code": "string", "format": "string"},
        tier=ModelTier.LOCAL_SMALL,
        cost_per_call=0.001,
        latency_estimate_ms=300
    ),
    ToolDefinition(
        name="nfcore_reference",
        description="Searches nf-core modules and pipelines for reference implementations.",
        parameters={"query": "string", "analysis_type": "string"},
        tier=ModelTier.LOCAL_SMALL,
        cost_per_call=0.0,
        latency_estimate_ms=100
    ),
    ToolDefinition(
        name="container_selector",
        description="Selects appropriate containers for bioinformatics tools.",
        parameters={"tools": "list[string]"},
        tier=ModelTier.LOCAL_SMALL,
        cost_per_call=0.0,
        latency_estimate_ms=50
    ),
    ToolDefinition(
        name="cloud_expert",
        description="Consults a powerful cloud model (GPT-4/Claude) for complex reasoning. Use sparingly due to cost.",
        parameters={"query": "string", "context": "string"},
        tier=ModelTier.CLOUD_LARGE,
        cost_per_call=0.05,
        latency_estimate_ms=3000
    ),
]


class Orchestrator8B:
    """
    NVIDIA Orchestrator-8B integration for BioPipelines.
    
    Uses the RL-trained 8B model to make intelligent routing decisions
    about which tools and models to use for each query.
    """
    
    def __init__(self, config: Optional[OrchestratorConfig] = None):
        """
        Initialize Orchestrator-8B.
        
        Args:
            config: Orchestrator configuration
        """
        self.config = config or OrchestratorConfig()
        self._model = None
        self._tokenizer = None
        self._client = None
        
        # Use default BioPipeline tools if none provided
        if not self.config.tool_catalog:
            self.config.tool_catalog = BIOPIPELINE_TOOLS
        
        # Build tool descriptions for prompt
        self._tool_descriptions = self._build_tool_descriptions()
        
        logger.info(f"Orchestrator-8B initialized with {len(self.config.tool_catalog)} tools")
    
    def _build_tool_descriptions(self) -> str:
        """Build tool descriptions JSON for the orchestrator prompt."""
        tools = []
        for tool in self.config.tool_catalog:
            tools.append({
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
                "cost": tool.cost_per_call,
                "latency_ms": tool.latency_estimate_ms
            })
        return json.dumps(tools, indent=2)
    
    async def initialize(self):
        """
        Initialize the model backend.
        
        Supports:
        - vLLM server (recommended for production)
        - Transformers (for development)
        - Remote API endpoint
        """
        if self.config.inference_backend == "vllm":
            await self._init_vllm()
        elif self.config.inference_backend == "transformers":
            await self._init_transformers()
        elif self.config.inference_backend == "api":
            await self._init_api()
        else:
            raise ValueError(f"Unknown backend: {self.config.inference_backend}")
    
    async def _init_vllm(self):
        """Initialize vLLM client."""
        try:
            from openai import AsyncOpenAI
            
            # Default vLLM endpoint
            base_url = self.config.api_endpoint or "http://localhost:8000/v1"
            
            self._client = AsyncOpenAI(
                base_url=base_url,
                api_key="not-needed"  # vLLM doesn't require API key
            )
            
            logger.info(f"vLLM client initialized at {base_url}")
            
        except ImportError:
            raise ImportError("Install openai: pip install openai")
    
    async def _init_transformers(self):
        """Initialize with transformers (for development/testing)."""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            
            logger.info(f"Loading {self.config.model_name} with transformers...")
            
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                trust_remote_code=True
            )
            
            self._model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            )
            
            logger.info("Model loaded successfully")
            
        except ImportError:
            raise ImportError("Install transformers: pip install transformers torch")
    
    async def _init_api(self):
        """Initialize API client for remote endpoint."""
        from openai import AsyncOpenAI
        
        if not self.config.api_endpoint:
            raise ValueError("api_endpoint required for API backend")
        
        self._client = AsyncOpenAI(
            base_url=self.config.api_endpoint,
            api_key=os.environ.get("ORCHESTRATOR_API_KEY", "not-needed")
        )
    
    def _build_system_prompt(self) -> str:
        """Build the system prompt for Orchestrator-8B."""
        preference_str = self._build_preference_string()
        
        return f"""You are an intelligent orchestrator for BioPipelines, a bioinformatics workflow generation system.

Your job is to analyze user queries and decide:
1. WHICH tools to call and in what order
2. WHICH model tier to use for each step (local vs cloud, small vs large)
3. HOW to minimize cost while maximizing quality

## Available Tools
{self._tool_descriptions}

## User Preferences
{preference_str}

## Model Tiers Available
- local_small: Fast, cheap, good for simple tasks (Ollama, small local models)
- local_large: More capable, still local (vLLM with 70B models)
- cloud_small: Fast cloud models (GPT-3.5, Claude Haiku) 
- cloud_large: Most capable, expensive (GPT-4, Claude Opus)
- specialist: Domain-specific models (CodeLlama for code)

## Response Format
For each turn, respond with:
<thinking>
Your reasoning about the task and which tools/models to use
</thinking>

<tool_call>
{{"name": "tool_name", "parameters": {{"param": "value"}}}}
</tool_call>

OR if task is complete:
<answer>
Your final response to the user
</answer>

Remember: Use the simplest/cheapest option that can accomplish the task. Only escalate to cloud_large for genuinely complex reasoning."""

    def _build_preference_string(self) -> str:
        """Build user preference string for prompt."""
        prefs = []
        
        if self.config.prefer_local:
            prefs.append("- PREFER local models over cloud when possible")
        
        if self.config.max_cost_per_query < 0.5:
            prefs.append(f"- STRICT cost limit: ${self.config.max_cost_per_query:.2f} per query")
        else:
            prefs.append(f"- Cost budget: ${self.config.max_cost_per_query:.2f} per query")
        
        opt = self.config.optimize_for
        if opt == "cost":
            prefs.append("- MINIMIZE cost above all else")
        elif opt == "speed":
            prefs.append("- MINIMIZE latency, prefer faster models")
        elif opt == "accuracy":
            prefs.append("- MAXIMIZE quality, cost is secondary")
        else:
            prefs.append("- BALANCE cost, speed, and quality")
        
        return "\n".join(prefs)
    
    async def route_and_execute(
        self,
        query: str,
        context: Optional[str] = None,
        available_models: Optional[List[str]] = None
    ) -> OrchestrationResult:
        """
        Route query through orchestrator and execute with chosen models.
        
        Args:
            query: User query
            context: Optional additional context
            available_models: List of available model identifiers
            
        Returns:
            OrchestrationResult with response and metadata
        """
        start_time = time.time()
        
        # Build messages for orchestrator
        system_prompt = self._build_system_prompt()
        user_message = query
        if context:
            user_message = f"Context:\n{context}\n\nQuery:\n{query}"
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
        
        # Multi-turn orchestration loop
        tool_calls = []
        total_cost = 0.0
        turns = 0
        orchestrator_reasoning = []
        final_response = None
        model_used = "orchestrator-8b"
        tier_used = ModelTier.LOCAL_SMALL
        
        while turns < self.config.max_turns:
            turns += 1
            
            # Get orchestrator response
            response = await self._generate(messages)
            
            # Parse response
            thinking, tool_call, answer = self._parse_response(response)
            
            if thinking:
                orchestrator_reasoning.append(thinking)
            
            if answer:
                # Task complete
                final_response = answer
                break
            
            if tool_call:
                # Execute tool call
                tool_calls.append(tool_call)
                tool_result, tool_cost, tool_tier = await self._execute_tool(tool_call)
                total_cost += tool_cost
                
                if tool_tier.value > tier_used.value:
                    tier_used = tool_tier
                    model_used = tool_call.get("name", "unknown")
                
                # Add to conversation
                messages.append({"role": "assistant", "content": response})
                messages.append({
                    "role": "user", 
                    "content": f"Tool result from {tool_call['name']}:\n{tool_result}"
                })
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        return OrchestrationResult(
            success=final_response is not None,
            model_used=model_used,
            tier_used=tier_used,
            response=final_response or "Max turns reached without completion",
            orchestrator_reasoning="\n".join(orchestrator_reasoning),
            tool_calls=tool_calls,
            cost=total_cost,
            latency_ms=elapsed_ms,
            turns=turns
        )
    
    async def _generate(self, messages: List[Dict[str, str]]) -> str:
        """Generate response from Orchestrator-8B."""
        if self._client:
            # vLLM or API backend
            response = await self._client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            return response.choices[0].message.content
        
        elif self._model:
            # Transformers backend
            import torch
            
            # Format messages for chat
            prompt = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
            
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                    do_sample=self.config.temperature > 0
                )
            
            return self._tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        else:
            raise RuntimeError("Model not initialized. Call initialize() first.")
    
    def _parse_response(self, response: str) -> tuple:
        """Parse orchestrator response into components."""
        import re
        
        thinking = None
        tool_call = None
        answer = None
        
        # Extract thinking
        thinking_match = re.search(r'<thinking>(.*?)</thinking>', response, re.DOTALL)
        if thinking_match:
            thinking = thinking_match.group(1).strip()
        
        # Extract tool call
        tool_match = re.search(r'<tool_call>(.*?)</tool_call>', response, re.DOTALL)
        if tool_match:
            try:
                tool_call = json.loads(tool_match.group(1).strip())
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse tool call: {tool_match.group(1)}")
        
        # Extract answer
        answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
        if answer_match:
            answer = answer_match.group(1).strip()
        
        return thinking, tool_call, answer
    
    async def _execute_tool(self, tool_call: Dict[str, Any]) -> tuple:
        """
        Execute a tool call.
        
        Returns:
            Tuple of (result, cost, tier_used)
        """
        tool_name = tool_call.get("name")
        parameters = tool_call.get("parameters", {})
        
        # Find tool definition
        tool_def = next(
            (t for t in self.config.tool_catalog if t.name == tool_name),
            None
        )
        
        if not tool_def:
            return f"Error: Unknown tool {tool_name}", 0.0, ModelTier.LOCAL_SMALL
        
        # Execute based on tool type
        # This is where we integrate with actual BioPipeline components
        result = await self._dispatch_tool(tool_name, parameters)
        
        return result, tool_def.cost_per_call, tool_def.tier
    
    async def _dispatch_tool(self, tool_name: str, parameters: Dict[str, Any]) -> str:
        """Dispatch to actual tool implementation."""
        # Import here to avoid circular imports
        from ..agents.specialists import PlannerAgent, CodeGenAgent, ValidatorAgent
        from ..agents.specialists import ReferenceDiscoveryAgent
        
        try:
            if tool_name == "workflow_planner":
                planner = PlannerAgent()
                plan = planner.create_plan_sync(parameters.get("query", ""))
                return plan.to_json() if plan else "Planning failed"
            
            elif tool_name == "nfcore_reference":
                ref_agent = ReferenceDiscoveryAgent()
                result = ref_agent.discover_sync(
                    query=parameters.get("query", ""),
                    analysis_type=parameters.get("analysis_type")
                )
                return result.format_for_prompt() if result else "No references found"
            
            elif tool_name == "code_validator":
                validator = ValidatorAgent()
                validation = validator.validate_sync(parameters.get("code", ""))
                return json.dumps({"valid": validation.valid, "issues": validation.issues})
            
            elif tool_name == "cloud_expert":
                # Route to cloud LLM
                from . import get_orchestrator, Strategy
                orch = get_orchestrator(strategy=Strategy.CLOUD_ONLY)
                response = await orch.complete(
                    parameters.get("query", ""),
                    system_prompt=parameters.get("context", "")
                )
                return response.content
            
            else:
                return f"Tool {tool_name} not implemented"
                
        except Exception as e:
            logger.error(f"Tool execution failed: {e}")
            return f"Error executing {tool_name}: {str(e)}"
    
    def get_routing_decision(self, query: str) -> RoutingDecision:
        """
        Get routing decision without executing (for analysis/debugging).
        
        This is a synchronous method that uses heuristics when the model
        isn't loaded, useful for testing.
        """
        # Simple heuristic routing for when model isn't loaded
        query_lower = query.lower()
        
        # Determine complexity
        is_complex = any(word in query_lower for word in [
            "complex", "multi-step", "integrate", "combine", "optimize"
        ])
        
        is_code_heavy = any(word in query_lower for word in [
            "generate", "code", "workflow", "pipeline", "nextflow"
        ])
        
        is_simple = any(word in query_lower for word in [
            "list", "what is", "explain", "describe"
        ])
        
        # Route based on heuristics
        if is_simple:
            target_tier = ModelTier.LOCAL_SMALL
            target_model = "ollama/llama3"
            tools = ["nfcore_reference"]
        elif is_code_heavy and not is_complex:
            target_tier = ModelTier.LOCAL_LARGE
            target_model = "vllm/codellama-34b"
            tools = ["workflow_planner", "code_generator", "code_validator"]
        elif is_complex:
            target_tier = ModelTier.CLOUD_LARGE
            target_model = "gpt-4"
            tools = ["workflow_planner", "code_generator", "cloud_expert", "code_validator"]
        else:
            target_tier = ModelTier.LOCAL_SMALL
            target_model = "ollama/llama3"
            tools = ["workflow_planner", "code_generator"]
        
        # Apply user preferences
        if self.config.prefer_local and target_tier in [ModelTier.CLOUD_SMALL, ModelTier.CLOUD_LARGE]:
            target_tier = ModelTier.LOCAL_LARGE
            target_model = "vllm/llama-70b"
        
        # Estimate cost
        estimated_cost = sum(
            t.cost_per_call for t in self.config.tool_catalog if t.name in tools
        )
        
        return RoutingDecision(
            target_model=target_model,
            target_tier=target_tier,
            reasoning=f"Query complexity: {'complex' if is_complex else 'simple'}, "
                      f"Code generation: {is_code_heavy}",
            tool_calls_planned=tools,
            estimated_cost=estimated_cost,
            confidence=0.7 if not self._model else 0.9
        )


# === Convenience Functions ===

def get_orchestrator_8b(
    prefer_local: bool = True,
    max_cost: float = 1.0,
    backend: str = "vllm"
) -> Orchestrator8B:
    """
    Get a configured Orchestrator-8B instance.
    
    Args:
        prefer_local: Prefer local models when possible
        max_cost: Maximum cost per query in USD
        backend: Inference backend ("vllm", "transformers", "api")
        
    Returns:
        Configured Orchestrator8B instance
    """
    config = OrchestratorConfig(
        inference_backend=backend,
        prefer_local=prefer_local,
        max_cost_per_query=max_cost,
        tool_catalog=BIOPIPELINE_TOOLS
    )
    return Orchestrator8B(config)


async def quick_route(query: str, prefer_local: bool = True) -> OrchestrationResult:
    """
    Quick routing without full initialization.
    
    Uses heuristic routing when model not available.
    """
    orch = get_orchestrator_8b(prefer_local=prefer_local)
    decision = orch.get_routing_decision(query)
    
    return OrchestrationResult(
        success=True,
        model_used=decision.target_model,
        tier_used=decision.target_tier,
        response=f"Routed to {decision.target_model}",
        orchestrator_reasoning=decision.reasoning,
        tool_calls=[],
        cost=decision.estimated_cost,
        latency_ms=0,
        turns=0
    )
