"""
Model Router - Smart provider selection with fallback.

Provides automatic selection of the best available provider
and fallback to alternatives when primary is unavailable.
"""

from __future__ import annotations

import os
import time
import logging
from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass

from .registry import (
    get_registry,
    ModelRegistry,
    ModelConfig,
    ProviderConfig,
    ProviderType,
    ModelCapability,
)

logger = logging.getLogger(__name__)


@dataclass
class CompletionResult:
    """Result from a model completion."""
    content: str
    provider: str
    model: str
    tokens_used: int = 0
    latency_ms: float = 0
    fallback_used: bool = False
    attempts: int = 1


class ProviderError(Exception):
    """Error from a specific provider."""
    def __init__(self, provider: str, message: str, retriable: bool = True):
        self.provider = provider
        self.retriable = retriable
        super().__init__(f"[{provider}] {message}")


class ModelRouter:
    """
    Routes requests to the best available model provider.
    
    Implements automatic fallback when providers fail or are unavailable.
    """
    
    def __init__(
        self,
        registry: Optional[ModelRegistry] = None,
        skip_providers: Optional[List[str]] = None,
    ):
        """
        Initialize the router.
        
        Args:
            registry: Model registry to use (defaults to global)
            skip_providers: List of provider IDs to skip
        """
        self.registry = registry or get_registry()
        self.skip_providers = set(skip_providers or [])
        self._provider_clients: Dict[str, Any] = {}
        self._failures: Dict[str, int] = {}  # Track recent failures
    
    def _get_ordered_providers(
        self,
        capability: Optional[ModelCapability] = None,
    ) -> List[ProviderConfig]:
        """Get providers in priority order."""
        providers = self.registry.list_providers(available_only=True)
        
        # Filter out skipped providers
        providers = [
            p for p in providers
            if p.id not in self.skip_providers
        ]
        
        # Deprioritize providers with recent failures
        def sort_key(p: ProviderConfig) -> tuple:
            failures = self._failures.get(p.id, 0)
            return (failures, p.priority)
        
        return sorted(providers, key=sort_key)
    
    def _get_provider_client(self, provider: ProviderConfig) -> Any:
        """Get or create a client for the provider."""
        if provider.id in self._provider_clients:
            return self._provider_clients[provider.id]
        
        client = self._create_client(provider)
        self._provider_clients[provider.id] = client
        return client
    
    def _create_client(self, provider: ProviderConfig) -> Any:
        """Create a client for the specified provider."""
        # Import provider implementations lazily
        if provider.id == "lightning":
            from .providers.lightning import LightningProvider
            return LightningProvider(provider)
        elif provider.id == "gemini":
            from .providers.gemini import GeminiProvider
            return GeminiProvider(provider)
        elif provider.id == "openai":
            from .providers.openai import OpenAIProvider
            return OpenAIProvider(provider)
        elif provider.id == "github_copilot":
            # GitHub Copilot uses special integration
            from .providers.github_copilot import GitHubCopilotProvider
            return GitHubCopilotProvider(provider)
        elif provider.id == "vllm":
            from .providers.vllm import VLLMProvider
            return VLLMProvider(provider)
        else:
            raise ValueError(f"Unknown provider: {provider.id}")
    
    async def complete_async(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        fallback: bool = True,
        max_retries: int = 3,
        preferred_provider: Optional[str] = None,
        preferred_model: Optional[str] = None,
    ) -> CompletionResult:
        """
        Get a completion from the best available provider (async).
        
        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            fallback: Whether to try other providers on failure
            max_retries: Maximum retries per provider
            preferred_provider: Prefer this provider if available
            preferred_model: Prefer this model if available
            
        Returns:
            CompletionResult with the generated content
        """
        providers = self._get_ordered_providers()
        
        # Move preferred provider to front if specified and available
        if preferred_provider:
            providers = sorted(
                providers,
                key=lambda p: 0 if p.id == preferred_provider else 1
            )
        
        last_error = None
        attempts = 0
        
        for provider in providers:
            for retry in range(max_retries):
                attempts += 1
                try:
                    start_time = time.time()
                    client = self._get_provider_client(provider)
                    
                    # Select model for this provider
                    model = self._select_model(provider, preferred_model)
                    
                    result = await client.complete_async(
                        prompt=prompt,
                        system_prompt=system_prompt,
                        model=model,
                        max_tokens=max_tokens,
                        temperature=temperature,
                    )
                    
                    latency_ms = (time.time() - start_time) * 1000
                    
                    # Clear failure count on success
                    self._failures[provider.id] = 0
                    
                    return CompletionResult(
                        content=result["content"],
                        provider=provider.id,
                        model=model or provider.models[0] if provider.models else "unknown",
                        tokens_used=result.get("tokens_used", 0),
                        latency_ms=latency_ms,
                        fallback_used=(provider != providers[0]),
                        attempts=attempts,
                    )
                    
                except Exception as e:
                    last_error = e
                    logger.warning(
                        f"Provider {provider.id} failed (attempt {retry + 1}): {e}"
                    )
                    
                    # Track failures
                    self._failures[provider.id] = self._failures.get(provider.id, 0) + 1
                    
                    # Don't retry non-retriable errors
                    if isinstance(e, ProviderError) and not e.retriable:
                        break
            
            # Move to next provider if fallback enabled
            if not fallback:
                break
        
        # All providers failed
        raise ProviderError(
            provider="all",
            message=f"All providers failed. Last error: {last_error}",
            retriable=False,
        )
    
    def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> CompletionResult:
        """
        Synchronous completion wrapper.
        
        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt
            **kwargs: Additional arguments passed to complete_async
            
        Returns:
            CompletionResult with the generated content
        """
        import asyncio
        
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(
            self.complete_async(prompt, system_prompt, **kwargs)
        )
    
    def _select_model(
        self,
        provider: ProviderConfig,
        preferred: Optional[str] = None,
    ) -> Optional[str]:
        """Select the best model for a provider."""
        if not provider.models:
            return None
        
        if preferred and preferred in provider.models:
            return preferred
        
        # For local providers, check if model is actually loaded
        if provider.provider_type == ProviderType.LOCAL:
            # Return first available model
            for model_id in provider.models:
                model = self.registry.get_model(model_id)
                if model and model.enabled:
                    return model.hf_id
        
        return provider.models[0]


# Convenience functions
_router: Optional[ModelRouter] = None


def get_router() -> ModelRouter:
    """Get the global model router."""
    global _router
    if _router is None:
        _router = ModelRouter()
    return _router


def get_model_client(
    provider: Optional[str] = None,
    model: Optional[str] = None,
) -> ModelRouter:
    """
    Get a model client for making completions.
    
    Args:
        provider: Specific provider to use (optional)
        model: Specific model to use (optional)
        
    Returns:
        ModelRouter configured for the request
    """
    router = get_router()
    if provider:
        router.skip_providers = {
            p.id for p in router.registry.list_providers()
            if p.id != provider
        }
    return router


def get_best_provider() -> Optional[ProviderConfig]:
    """Get the best available provider."""
    router = get_router()
    providers = router._get_ordered_providers()
    return providers[0] if providers else None
