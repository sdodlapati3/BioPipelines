"""
LLM Adapter Layer
=================

Provides a unified interface to multiple LLM providers:
- OpenAI (GPT-4, GPT-4o, GPT-3.5)
- Anthropic (Claude 3.5, Claude 3)
- Ollama (local models: Llama3, Mistral, CodeLlama)
- HuggingFace (any HF model via API, transformers, or vLLM)
- vLLM (high-performance GPU inference for HF models)
- Lightning.ai (30M FREE tokens/month, unified API for all models)
- Custom endpoints

Usage:
    # === NEW Orchestrator API (v2.0) - RECOMMENDED ===
    from workflow_composer.llm import get_orchestrator, Strategy
    
    # Smart routing with automatic fallback
    orch = get_orchestrator(strategy=Strategy.LOCAL_FIRST)
    response = await orch.complete("Generate workflow for RNA-seq")
    print(f"Used: {response.provider}, Cost: ${response.cost:.4f}")
    
    # Ensemble for critical decisions
    response = await orch.ensemble("Is this workflow correct?")
    
    # Use preset configuration
    orch = get_orchestrator(preset="production")
    
    # === Provider API (v2.0) ===
    from workflow_composer.llm import LocalProvider, CloudProvider
    
    # Direct provider access
    local = LocalProvider()
    cloud = CloudProvider()
    
    # === Legacy Adapter API ===
    from workflow_composer.llm import get_llm, VLLMAdapter, LightningAdapter
    
    # Using factory
    llm = get_llm("vllm", model="meta-llama/Llama-3.1-8B-Instruct")
    
    # Direct instantiation
    llm = VLLMAdapter(model="codellama/CodeLlama-34b-Instruct-hf")
"""

# === Orchestrator (v2.0) - RECOMMENDED ===
from .orchestrator import (
    ModelOrchestrator,
    OrchestratorResponse,
    UsageStats,
    get_orchestrator,
    reset_orchestrator,
)

from .strategies import (
    Strategy,
    EnsembleMode,
    ChainRole,
    StrategyConfig,
    PRESETS,
    get_preset,
)

# === Task Router (v2.0) ===
from .task_router import (
    TaskType,
    TaskComplexity,
    TaskAnalysis,
    TaskRouter,
    RouterConfig,
)

# === Cost Tracker (v2.0) ===
from .cost_tracker import (
    CostTracker,
    CostEntry,
    BudgetAlert,
    CostSummary,
)

# === Orchestrator-8B (ToolOrchestra) ===
from .orchestrator_8b import (
    Orchestrator8B,
    OrchestratorConfig,
    OrchestrationResult,
    RoutingDecision,
    ModelTier,
    ToolDefinition,
    BIOPIPELINE_TOOLS,
    get_orchestrator_8b,
    quick_route,
)

# === Provider Layer (v2.0) ===
from .providers import (
    # Types
    ProviderProtocol,
    ProviderType,
    ModelCapability,
    ModelInfo,
    ProviderResponse,
    ProviderHealth,
    
    # Exceptions
    ProviderError,
    ProviderUnavailableError,
    ModelNotFoundError,
    
    # Base
    BaseProvider,
    
    # Unified Providers
    LocalProvider,
    CloudProvider,
    
    # Cloud Model Registry
    CLOUD_MODELS,
    DEFAULT_MODEL,
)

# === Legacy Adapter Layer ===
from .base import LLMAdapter, Message, LLMResponse
from .ollama_adapter import OllamaAdapter
from .openai_adapter import OpenAIAdapter
from .anthropic_adapter import AnthropicAdapter
from .huggingface_adapter import HuggingFaceAdapter
from .vllm_adapter import VLLMAdapter
from .lightning_adapter import LightningAdapter, LightningModelRouter, create_lightning_llm
from .factory import get_llm, LLMFactory, list_providers, check_providers, register_provider

__all__ = [
    # === Orchestrator API (RECOMMENDED) ===
    "ModelOrchestrator",
    "OrchestratorResponse",
    "UsageStats",
    "get_orchestrator",
    "reset_orchestrator",
    "Strategy",
    "EnsembleMode",
    "ChainRole",
    "StrategyConfig",
    "PRESETS",
    "get_preset",
    
    # === Task Router API ===
    "TaskType",
    "TaskComplexity",
    "TaskAnalysis",
    "TaskRouter",
    "RouterConfig",
    
    # === Cost Tracker API ===
    "CostTracker",
    "CostEntry",
    "BudgetAlert",
    "CostSummary",
    
    # === Orchestrator-8B (ToolOrchestra) ===
    "Orchestrator8B",
    "OrchestratorConfig",
    "OrchestrationResult",
    "RoutingDecision",
    "ModelTier",
    "ToolDefinition",
    "BIOPIPELINE_TOOLS",
    "get_orchestrator_8b",
    "quick_route",
    
    # === Provider API ===
    "ProviderProtocol",
    "ProviderType",
    "ModelCapability",
    "ModelInfo",
    "ProviderResponse",
    "ProviderHealth",
    "ProviderError",
    "ProviderUnavailableError",
    "ModelNotFoundError",
    "BaseProvider",
    "LocalProvider",
    "CloudProvider",
    "CLOUD_MODELS",
    "DEFAULT_MODEL",
    
    # === Legacy Adapter API ===
    "LLMAdapter",
    "Message", 
    "LLMResponse",
    "OllamaAdapter",
    "OpenAIAdapter",
    "AnthropicAdapter",
    "HuggingFaceAdapter",
    "VLLMAdapter",
    "LightningAdapter",
    "LightningModelRouter",
    "create_lightning_llm",
    "get_llm",
    "LLMFactory",
    "list_providers",
    "check_providers",
    "register_provider"
]
