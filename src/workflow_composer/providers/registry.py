"""
Provider and Model Registry
===========================

Central registry for all provider configurations and model definitions.
Supports both API providers and local models.
"""

from __future__ import annotations

import os
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Type
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ProviderType(Enum):
    """Type of provider."""
    API = "api"          # Cloud API (OpenAI, Gemini, etc.)
    LOCAL = "local"      # Local server (Ollama, vLLM)


@dataclass
class ProviderConfig:
    """Configuration for a provider."""
    id: str
    name: str
    provider_type: ProviderType
    priority: int  # Lower = higher priority (tried first)
    env_key: Optional[str] = None  # Environment variable for API key
    base_url: Optional[str] = None
    default_model: Optional[str] = None
    models: List[str] = field(default_factory=list)
    free_tier: bool = False
    rate_limit: Optional[str] = None
    enabled: bool = True
    
    def is_configured(self) -> bool:
        """Check if provider has required configuration."""
        if self.provider_type == ProviderType.API and self.env_key:
            return bool(os.environ.get(self.env_key))
        return True
    
    def get_api_key(self) -> Optional[str]:
        """Get API key from environment."""
        if self.env_key:
            return os.environ.get(self.env_key)
        return None


@dataclass
class ModelConfig:
    """Configuration for a local model."""
    id: str
    name: str
    hf_id: str  # HuggingFace model ID
    size_gb: float = 0
    gpus_required: int = 1
    context_length: int = 8192
    capabilities: List[str] = field(default_factory=list)
    enabled: bool = True
    vllm_args: Dict[str, Any] = field(default_factory=dict)


# Default provider configurations
# Priority: Lower = higher priority (tried first)
# Strategy: Maximize free tier usage before falling back to paid
DEFAULT_PROVIDERS: Dict[str, ProviderConfig] = {
    # ==========================================================================
    # TIER 1: Best Free Tiers (High Limits)
    # ==========================================================================
    "gemini": ProviderConfig(
        id="gemini",
        name="Google AI Studio",
        provider_type=ProviderType.API,
        priority=1,  # Best free tier: 250+ req/day, 1M tokens
        env_key="GOOGLE_API_KEY",
        base_url="https://generativelanguage.googleapis.com/v1beta",
        default_model="gemini-2.0-flash",
        models=[
            "gemini-2.0-flash",           # Fast, 200 req/day
            "gemini-2.5-flash",           # 250 req/day
            "gemini-2.5-pro",             # 50 req/day, best quality
        ],
        free_tier=True,
        rate_limit="15/min, 250/day",
    ),
    "cerebras": ProviderConfig(
        id="cerebras",
        name="Cerebras Cloud",
        provider_type=ProviderType.API,
        priority=2,  # Very generous: 14,400 req/day, 1M tokens
        env_key="CEREBRAS_API_KEY",
        base_url="https://api.cerebras.ai/v1",
        default_model="llama-3.3-70b",
        models=[
            "llama-3.3-70b",              # 14,400 req/day
            "qwen3-235b-a22b",            # 235B params, FREE!
            "qwen3-coder-480b",           # Best coding, 100 req/day
            "gpt-oss-120b",               # 14,400 req/day
            "llama-4-scout",              # Latest Llama
        ],
        free_tier=True,
        rate_limit="60k tok/min, 14,400/day",
    ),
    "groq": ProviderConfig(
        id="groq",
        name="Groq Cloud",
        provider_type=ProviderType.API,
        priority=3,  # Fast inference: 1,000+ req/day
        env_key="GROQ_API_KEY",
        base_url="https://api.groq.com/openai/v1",
        default_model="llama-3.3-70b-versatile",
        models=[
            "llama-3.3-70b-versatile",    # 1,000 req/day
            "llama-3.1-8b-instant",       # 14,400 req/day, fastest
            "gpt-oss-120b",               # 1,000 req/day
            "groq/compound",              # Agentic with tools
        ],
        free_tier=True,
        rate_limit="1,000/day (70B), 14,400/day (8B)",
    ),
    
    # ==========================================================================
    # TIER 2: Good Free Tiers (Lower Limits)
    # ==========================================================================
    "openrouter": ProviderConfig(
        id="openrouter",
        name="OpenRouter",
        provider_type=ProviderType.API,
        priority=4,  # Gateway to many free models: 50 req/day
        env_key="OPENROUTER_API_KEY",
        base_url="https://openrouter.ai/api/v1",
        default_model="meta-llama/llama-3.3-70b-instruct:free",
        models=[
            "meta-llama/llama-3.3-70b-instruct:free",
            "qwen/qwen3-235b-a22b:free",
            "deepseek/deepseek-r1t-chimera:free",
            "google/gemma-3-27b-it:free",
            "mistralai/mistral-small-3.1-24b-instruct:free",
        ],
        free_tier=True,
        rate_limit="20/min, 50/day (free models)",
    ),
    "lightning": ProviderConfig(
        id="lightning",
        name="Lightning.ai",
        provider_type=ProviderType.API,
        priority=5,
        env_key="LIGHTNING_API_KEY",
        base_url="https://lightning.ai/api/v1",
        default_model="lightning-ai/DeepSeek-V3.1",
        models=[
            "lightning-ai/DeepSeek-V3.1",
            "openai/gpt-4o",
            "google/gemini-2.5-flash",
            "lightning-ai/gpt-oss-120b",
        ],
        free_tier=True,
        rate_limit="100/min",
        # DISABLED: API returns empty responses (status 200, 0 bytes) as of Dec 2025
        # Re-enable when Lightning.ai fixes their API or account is verified
        enabled=False,
    ),
    "github_models": ProviderConfig(
        id="github_models",
        name="GitHub Models",
        provider_type=ProviderType.API,
        priority=6,  # Free with GitHub Copilot subscription
        env_key="GITHUB_TOKEN",
        base_url="https://models.inference.ai.azure.com",
        default_model="gpt-4o-mini",
        models=["gpt-4o-mini", "gpt-4o", "DeepSeek-R1", "Llama-3.3-70B-Instruct"],
        free_tier=True,
        rate_limit="Tier-dependent",
    ),
    
    # ==========================================================================
    # TIER 3: Paid (No Free Tier)
    # ==========================================================================
    "anthropic": ProviderConfig(
        id="anthropic",
        name="Anthropic",
        provider_type=ProviderType.API,
        priority=10,  # Paid only - use after free tiers exhausted
        env_key="ANTHROPIC_API_KEY",
        base_url="https://api.anthropic.com",
        default_model="claude-3-5-sonnet-20241022",
        models=["claude-3-5-sonnet-20241022", "claude-3-opus-20240229", "claude-3-haiku-20240307"],
        free_tier=False,
    ),
    
    # ==========================================================================
    # TIER 4: Local Models (Always Free)
    # ==========================================================================
    "ollama": ProviderConfig(
        id="ollama",
        name="Ollama (Local)",
        provider_type=ProviderType.LOCAL,
        priority=15,  # Local fallback if cloud unavailable
        base_url="http://localhost:11434",
        default_model="llama3:8b",
        models=["llama3:8b", "mistral:7b", "codellama:13b", "qwen2.5:32b"],
        free_tier=True,
    ),
    "vllm": ProviderConfig(
        id="vllm",
        name="vLLM (Local)",
        provider_type=ProviderType.LOCAL,
        priority=16,  # Local high-performance
        base_url="http://localhost:8000/v1",
        default_model="Qwen/Qwen2.5-Coder-32B-Instruct",
        models=[
            "Qwen/Qwen2.5-Coder-32B-Instruct",
            "deepseek-ai/DeepSeek-Coder-V2-Instruct",
            "meta-llama/Llama-3.3-70B-Instruct",
        ],
        free_tier=True,
    ),
    
    # ==========================================================================
    # TIER 5: Expensive Fallback (Last Resort)
    # ==========================================================================
    "openai": ProviderConfig(
        id="openai",
        name="OpenAI",
        provider_type=ProviderType.API,
        priority=99,  # LAST RESORT - paid, but reliable
        env_key="OPENAI_API_KEY",
        base_url="https://api.openai.com/v1",
        default_model="gpt-4o-mini",
        models=["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"],
        free_tier=False,
        rate_limit="500/min",
    ),
}


# Default local model configurations
DEFAULT_MODELS: Dict[str, ModelConfig] = {
    "qwen-coder-32b": ModelConfig(
        id="qwen-coder-32b",
        name="Qwen2.5-Coder-32B-Instruct",
        hf_id="Qwen/Qwen2.5-Coder-32B-Instruct",
        size_gb=65,
        gpus_required=1,
        context_length=32768,
        capabilities=["code", "chat"],
    ),
    "deepseek-coder-v2": ModelConfig(
        id="deepseek-coder-v2",
        name="DeepSeek-Coder-V2-Instruct",
        hf_id="deepseek-ai/DeepSeek-Coder-V2-Instruct",
        size_gb=120,
        gpus_required=2,
        context_length=128000,
        capabilities=["code", "chat", "reasoning"],
    ),
    "llama-3.3-70b": ModelConfig(
        id="llama-3.3-70b",
        name="Llama-3.3-70B-Instruct",
        hf_id="meta-llama/Llama-3.3-70B-Instruct",
        size_gb=140,
        gpus_required=2,
        context_length=128000,
        capabilities=["code", "chat", "reasoning"],
    ),
    "minimax-m2": ModelConfig(
        id="minimax-m2",
        name="MiniMax-M2",
        hf_id="MiniMaxAI/MiniMax-M2",
        size_gb=230,
        gpus_required=4,
        context_length=128000,
        capabilities=["code", "chat", "reasoning", "agentic"],
    ),
    "codellama-34b": ModelConfig(
        id="codellama-34b",
        name="CodeLlama-34B-Instruct",
        hf_id="codellama/CodeLlama-34b-Instruct-hf",
        size_gb=70,
        gpus_required=1,
        context_length=16384,
        capabilities=["code"],
    ),
}


class ProviderRegistry:
    """
    Central registry for providers and models.
    
    Manages provider configurations and provides lookup methods.
    Loads defaults and optionally reads from YAML configs.
    """
    
    def __init__(self, config_dir: Optional[Path] = None):
        """
        Initialize the registry.
        
        Args:
            config_dir: Optional directory with YAML config files
        """
        self._providers: Dict[str, ProviderConfig] = {}
        self._models: Dict[str, ModelConfig] = {}
        self._provider_classes: Dict[str, Type] = {}
        
        # Load defaults
        self._providers.update(DEFAULT_PROVIDERS)
        self._models.update(DEFAULT_MODELS)
        
        # Load from YAML if available
        if config_dir:
            self._load_yaml_configs(Path(config_dir))
    
    def _load_yaml_configs(self, config_dir: Path):
        """Load configurations from YAML files."""
        providers_file = config_dir / "providers.yaml"
        models_file = config_dir / "models.yaml"
        
        if providers_file.exists():
            try:
                with open(providers_file) as f:
                    data = yaml.safe_load(f) or {}
                    for pid, pdata in data.get("providers", {}).items():
                        self._providers[pid] = ProviderConfig(
                            id=pid,
                            name=pdata.get("name", pid),
                            provider_type=ProviderType(pdata.get("type", "api")),
                            priority=pdata.get("priority", 100),
                            env_key=pdata.get("env_key"),
                            base_url=pdata.get("base_url"),
                            default_model=pdata.get("default_model"),
                            models=pdata.get("models", []),
                            free_tier=pdata.get("free_tier", False),
                            rate_limit=pdata.get("rate_limit"),
                            enabled=pdata.get("enabled", True),
                        )
            except Exception as e:
                logger.warning(f"Failed to load providers.yaml: {e}")
        
        if models_file.exists():
            try:
                with open(models_file) as f:
                    data = yaml.safe_load(f) or {}
                    for mid, mdata in data.get("models", {}).items():
                        self._models[mid] = ModelConfig(
                            id=mid,
                            name=mdata.get("name", mid),
                            hf_id=mdata.get("hf_id", ""),
                            size_gb=mdata.get("size_gb", 0),
                            gpus_required=mdata.get("gpus_required", 1),
                            context_length=mdata.get("context_length", 8192),
                            capabilities=mdata.get("capabilities", []),
                            enabled=mdata.get("enabled", True),
                            vllm_args=mdata.get("vllm_args", {}),
                        )
            except Exception as e:
                logger.warning(f"Failed to load models.yaml: {e}")
    
    def get_provider_config(self, provider_id: str) -> Optional[ProviderConfig]:
        """Get provider configuration by ID."""
        return self._providers.get(provider_id)
    
    def get_model_config(self, model_id: str) -> Optional[ModelConfig]:
        """Get model configuration by ID."""
        return self._models.get(model_id)
    
    def list_providers(
        self,
        configured_only: bool = False,
        enabled_only: bool = True,
    ) -> List[ProviderConfig]:
        """
        List available providers.
        
        Args:
            configured_only: Only return configured providers
            enabled_only: Only return enabled providers
            
        Returns:
            List of provider configurations sorted by priority
        """
        providers = list(self._providers.values())
        
        if enabled_only:
            providers = [p for p in providers if p.enabled]
        
        if configured_only:
            providers = [p for p in providers if p.is_configured()]
        
        return sorted(providers, key=lambda p: p.priority)
    
    def list_models(
        self,
        capability: Optional[str] = None,
        enabled_only: bool = True,
    ) -> List[ModelConfig]:
        """
        List available local models.
        
        Args:
            capability: Filter by capability (e.g., "code", "chat")
            enabled_only: Only return enabled models
            
        Returns:
            List of model configurations
        """
        models = list(self._models.values())
        
        if enabled_only:
            models = [m for m in models if m.enabled]
        
        if capability:
            models = [m for m in models if capability in m.capabilities]
        
        return models
    
    def register_provider_class(self, provider_id: str, cls: Type):
        """Register a provider implementation class."""
        self._provider_classes[provider_id] = cls
    
    def get_provider_class(self, provider_id: str) -> Optional[Type]:
        """Get the implementation class for a provider."""
        return self._provider_classes.get(provider_id)
    
    def get_best_provider(self) -> Optional[ProviderConfig]:
        """Get the highest priority configured provider."""
        providers = self.list_providers(configured_only=True)
        return providers[0] if providers else None


# Global registry instance
_registry: Optional[ProviderRegistry] = None


def get_registry() -> ProviderRegistry:
    """Get the global provider registry."""
    global _registry
    if _registry is None:
        _registry = ProviderRegistry()
    return _registry
