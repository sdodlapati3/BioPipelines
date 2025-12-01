"""
Unified Provider Framework for BioPipelines
============================================

This package provides a single, unified interface to all LLM providers.
It consolidates the previously scattered implementations into one clean layer.

Architecture:
    providers/
    ├── base.py           # BaseProvider abstract class
    ├── registry.py       # Central provider & model registry
    ├── router.py         # Smart routing with fallback
    ├── gemini.py         # Google AI Studio (FREE tier) ⭐
    ├── cerebras.py       # Cerebras Cloud (FREE tier, generous) ⭐
    ├── groq.py           # Groq Cloud (FREE tier, fast) ⭐
    ├── openrouter.py     # OpenRouter gateway (FREE models) ⭐
    ├── lightning.py      # Lightning.ai
    ├── github_models.py  # GitHub Models (with Copilot)
    ├── openai.py         # OpenAI (paid)
    ├── anthropic.py      # Anthropic Claude (paid)
    ├── ollama.py         # Local Ollama (free)
    ├── vllm.py           # Local vLLM server (free)
    └── utils/
        ├── health.py     # Health checking
        └── metrics.py    # Usage tracking

Provider Priority (cascade on rate limit):
    1. Gemini       - 250 req/day, 1M tokens/day FREE
    2. Cerebras     - 14,400 req/day, 1M tokens/day FREE
    3. Groq         - 1,000+ req/day, blazing fast FREE
    4. OpenRouter   - 50 req/day, 20+ free models
    5. Lightning.ai - DeepSeek V3
    6. GitHub Models- With Copilot subscription
    10. Anthropic   - Paid only
    15-16. Local    - Ollama/vLLM
    99. OpenAI      - Paid fallback (always works)

Usage:
    from workflow_composer.providers import (
        get_provider,
        get_best_provider,
        complete,
        chat,
        ProviderRouter,
    )
    
    # Simple completion (auto-selects best free provider)
    response = complete("Explain RNA-seq analysis")
    
    # Chat with messages
    response = chat([
        Message.system("You are a bioinformatics expert."),
        Message.user("What causes OOM errors in STAR?"),
    ])
    
    # Force specific provider
    response = complete("Debug this error", provider="cerebras")
    
    # Get provider instance for advanced use
    provider = get_provider("groq")
    response = provider.complete("...")
    
    # Smart routing with automatic fallback
    router = ProviderRouter()
    response = router.complete("...", fallback=True)
"""

__version__ = "2.0.0"

# Core types
from .base import (
    BaseProvider,
    Message,
    Role,
    ProviderResponse,
    ProviderError,
)

# Registry
from .registry import (
    ProviderRegistry,
    ProviderConfig,
    ModelConfig,
    ProviderType,
    get_registry,
)

# Router
from .router import (
    ProviderRouter,
    get_router,
    get_best_provider,
)

# Individual providers
from .gemini import GeminiProvider
from .cerebras import CerebrasProvider
from .groq import GroqProvider
from .openrouter import OpenRouterProvider
from .lightning import LightningProvider
from .github_models import GitHubModelsProvider
from .openai import OpenAIProvider
from .anthropic import AnthropicProvider
from .ollama import OllamaProvider
from .vllm import VLLMProvider

# Utilities
from .utils import (
    check_provider,
    check_all_providers,
    HealthStatus,
    UsageMetrics,
    get_usage_tracker,
)

# Factory functions
from .factory import (
    get_provider,
    get_router as get_factory_router,
    complete,
    chat,
    stream,
    check_providers,
    get_available_providers,
    list_providers,
    register_provider,
    print_provider_status,
)


__all__ = [
    # Version
    "__version__",
    
    # Core types
    "BaseProvider",
    "Message",
    "Role",
    "ProviderResponse",
    "ProviderError",
    
    # Registry
    "ProviderRegistry",
    "ProviderConfig",
    "ModelConfig",
    "ProviderType",
    "get_registry",
    
    # Router
    "ProviderRouter",
    "get_router",
    "get_best_provider",
    
    # Providers
    "GeminiProvider",
    "CerebrasProvider",
    "GroqProvider",
    "OpenRouterProvider",
    "LightningProvider",
    "GitHubModelsProvider",
    "OpenAIProvider",
    "AnthropicProvider",
    "OllamaProvider",
    "VLLMProvider",
    
    # Factory
    "get_provider",
    "complete",
    "chat",
    "stream",
    "check_providers",
    "get_available_providers",
    "list_providers",
    "register_provider",
    "print_provider_status",
    
    # Utilities
    "check_provider",
    "check_all_providers",
    "HealthStatus",
    "UsageMetrics",
    "get_usage_tracker",
]
