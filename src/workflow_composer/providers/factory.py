"""
Factory functions for provider creation and usage.

Provides convenient access to providers without needing to
understand the registry or router internals.
"""

import os
from typing import Optional, List, Union, AsyncIterator

from .base import BaseProvider, Message, ProviderResponse, Role
from .registry import ProviderType, ProviderRegistry, get_registry
from .router import ProviderRouter
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


# Map of provider IDs to classes (ordered by priority)
PROVIDER_CLASSES = {
    # Tier 1: Best free tiers
    "gemini": GeminiProvider,
    "cerebras": CerebrasProvider,
    "groq": GroqProvider,
    "openrouter": OpenRouterProvider,
    # Tier 2: Good free tiers
    "lightning": LightningProvider,
    "github_models": GitHubModelsProvider,
    "github": GitHubModelsProvider,  # Alias
    # Tier 3: Paid
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    # Tier 4: Local
    "ollama": OllamaProvider,
    "vllm": VLLMProvider,
}


# Global instances (cached)
_providers: dict = {}
_router: Optional[ProviderRouter] = None


def get_provider(
    provider_type: str,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    **kwargs,
) -> BaseProvider:
    """
    Get a provider instance.
    
    Args:
        provider_type: Provider to use ("lightning", "gemini", etc.)
        model: Model to use (defaults to provider's default)
        api_key: API key (uses env var if not provided)
        **kwargs: Additional provider config
        
    Returns:
        Configured provider instance
        
    Example:
        >>> provider = get_provider("gemini")
        >>> response = provider.complete("Hello!")
    """
    # Normalize provider type
    provider_id = provider_type.lower()
    
    # Get the provider class
    provider_class = PROVIDER_CLASSES.get(provider_id)
    if not provider_class:
        raise ValueError(f"Unknown provider: {provider_type}. Available: {list(PROVIDER_CLASSES.keys())}")
    
    # Create and return instance
    return provider_class(model=model, api_key=api_key, **kwargs)


def get_router(skip_providers: Optional[List[str]] = None) -> ProviderRouter:
    """
    Get the global provider router with fallback support.
    
    Args:
        skip_providers: List of provider IDs to skip
        
    Returns:
        ProviderRouter configured with all available providers
    """
    global _router
    
    if _router is None:
        _router = ProviderRouter(skip_providers=skip_providers)
    
    return _router


def complete(
    prompt: str,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 2048,
    system_prompt: Optional[str] = None,
    with_fallback: bool = True,
) -> ProviderResponse:
    """
    Complete a prompt using the best available provider.
    
    This is the main entry point for simple completions.
    Uses automatic fallback to ensure reliability.
    
    Args:
        prompt: The prompt to complete
        provider: Specific provider to use (auto-selects if not specified)
        model: Specific model to use
        temperature: Response creativity (0-1)
        max_tokens: Maximum response length
        system_prompt: System prompt for context
        with_fallback: Whether to try other providers on failure
        
    Returns:
        ProviderResponse with the completion
        
    Example:
        >>> response = complete("Write a haiku about Python")
        >>> print(response.content)
    """
    if provider:
        # Use specific provider
        p = get_provider(provider, model=model)
        return p.complete(
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
        )
    else:
        # Use router for automatic fallback
        router = get_router()
        return router.complete(
            prompt=prompt,
            system_prompt=system_prompt,
            fallback=with_fallback,
            temperature=temperature,
            max_tokens=max_tokens,
        )


def chat(
    messages: List[Message],
    provider: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 2048,
    with_fallback: bool = True,
) -> ProviderResponse:
    """
    Have a multi-turn chat conversation.
    
    Args:
        messages: List of Message objects representing the conversation
        provider: Specific provider to use
        model: Specific model to use
        temperature: Response creativity
        max_tokens: Maximum response length
        with_fallback: Whether to try other providers on failure
        
    Returns:
        ProviderResponse with the assistant's reply
        
    Example:
        >>> messages = [
        ...     Message.system("You are helpful"),
        ...     Message.user("Hello!"),
        ... ]
        >>> response = chat(messages)
        >>> print(response.content)
    """
    if provider:
        p = get_provider(provider, model=model)
        return p.chat(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    else:
        router = get_router()
        return router.chat(
            messages=messages,
            fallback=with_fallback,
            temperature=temperature,
            max_tokens=max_tokens,
        )


async def stream(
    prompt: str,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 2048,
    system_prompt: Optional[str] = None,
) -> AsyncIterator[str]:
    """
    Stream a completion response.
    
    Args:
        prompt: The prompt to complete
        provider: Provider to use (defaults to first available)
        model: Model to use
        temperature: Response creativity
        max_tokens: Maximum response length
        system_prompt: System prompt for context
        
    Yields:
        Chunks of the response as they arrive
        
    Example:
        >>> async for chunk in stream("Tell me a story"):
        ...     print(chunk, end="", flush=True)
    """
    if provider:
        p = get_provider(provider, model=model)
    else:
        router = get_router()
        p = router.primary_provider
    
    async for chunk in p.stream(
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        system_prompt=system_prompt,
    ):
        yield chunk


def check_providers() -> dict:
    """
    Check availability of all providers.
    
    Returns:
        Dict mapping provider names to availability status
        
    Example:
        >>> status = check_providers()
        >>> print(status)
        {'lightning': True, 'gemini': True, 'openai': False, ...}
    """
    from .utils.health import check_all_providers
    
    results = check_all_providers()
    return {
        name: status.available
        for name, status in results.items()
    }


def get_available_providers() -> List[str]:
    """
    Get list of currently available providers.
    
    Returns:
        List of provider names that are ready to use
    """
    status = check_providers()
    return [name for name, available in status.items() if available]


def print_provider_status():
    """Print formatted status of all providers."""
    from .utils.health import check_all_providers
    
    results = check_all_providers()
    
    print("\n" + "=" * 50)
    print("PROVIDER STATUS")
    print("=" * 50 + "\n")
    
    for name, status in sorted(results.items()):
        icon = "✅" if status.available else "❌"
        latency = f" ({status.latency_ms:.0f}ms)" if status.latency_ms else ""
        error = f" - {status.error}" if status.error else ""
        print(f"  {icon} {name.upper()}{latency}{error}")
    
    print()
    available = sum(1 for s in results.values() if s.available)
    print(f"Available: {available}/{len(results)}")
    print("=" * 50 + "\n")


def list_providers() -> List[str]:
    """
    List all registered provider types.
    
    Returns:
        List of provider names
    """
    return list(PROVIDER_CLASSES.keys())


def register_provider(
    name: str,
    provider_class: type,
    **defaults,
):
    """
    Register a new provider type.
    
    Args:
        name: Provider name (lowercase)
        provider_class: Provider class implementing BaseProvider
        **defaults: Default configuration values
        
    Note:
        This is primarily for extensibility. Built-in providers
        are already registered.
    """
    global PROVIDER_CLASSES
    
    name = name.lower()
    if name in PROVIDER_CLASSES:
        raise ValueError(f"Provider {name} is already registered")
    
    PROVIDER_CLASSES[name] = provider_class
