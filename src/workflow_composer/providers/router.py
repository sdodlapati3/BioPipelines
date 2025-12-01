"""
Provider Router
===============

Smart routing with automatic fallback between providers.
"""

from __future__ import annotations

import time
import logging
from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass

from .base import BaseProvider, Message, ProviderResponse, ProviderError
from .registry import get_registry, ProviderRegistry, ProviderConfig

logger = logging.getLogger(__name__)


@dataclass
class RouteResult:
    """Result of a routed request."""
    response: ProviderResponse
    provider_used: str
    fallback_used: bool = False
    attempts: int = 1
    errors: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []


class ProviderRouter:
    """
    Routes requests to the best available provider with cascading fallback.
    
    Strategy: Cascade (not wait-and-retry)
    -----------------------------------------
    When a provider hits rate limits or fails, we immediately cascade to the
    next provider in priority order. This is faster and simpler than waiting
    for rate limit reset.
    
    Features:
        - Automatic provider selection based on priority
        - Instant cascade to next provider on failure (no waiting)
        - Session-level tracking: rate-limited providers marked inactive
        - Providers reactivate on next session (app restart)
    
    Example:
        router = ProviderRouter()
        
        # Auto-cascades through: GitHub Models → Gemini → OpenAI
        response = router.complete("Explain RNA-seq")
        
        # Check which providers are active this session
        print(router.get_status())
    """
    
    def __init__(
        self,
        registry: Optional[ProviderRegistry] = None,
        skip_providers: Optional[List[str]] = None,
        max_retries: int = 1,  # Reduced: cascade fast, don't retry same provider
    ):
        """
        Initialize the router.
        
        Args:
            registry: Provider registry (uses global if not provided)
            skip_providers: Provider IDs to skip
            max_retries: Max retries per provider (1 = no retries, cascade immediately)
        """
        self.registry = registry or get_registry()
        self.skip_providers = set(skip_providers or [])
        self.max_retries = max_retries
        
        # Session state: track provider status
        self._failure_counts: Dict[str, int] = {}
        self._provider_cache: Dict[str, BaseProvider] = {}
        
        # Rate limit tracking: providers marked inactive for this session
        self._rate_limited: Dict[str, bool] = {}  # provider_id -> is_rate_limited
        self._inactive_reason: Dict[str, str] = {}  # provider_id -> reason
    
    def _get_ordered_providers(self) -> List[ProviderConfig]:
        """Get active providers in priority order, skipping rate-limited ones."""
        providers = self.registry.list_providers(configured_only=True)
        
        # Filter out skipped and rate-limited providers
        active_providers = []
        for p in providers:
            if p.id in self.skip_providers:
                continue
            if self._rate_limited.get(p.id, False):
                logger.debug(f"Skipping rate-limited provider: {p.id}")
                continue
            active_providers.append(p)
        
        # Sort by (failure_count, priority)
        def sort_key(p: ProviderConfig) -> tuple:
            failures = self._failure_counts.get(p.id, 0)
            return (failures > 3, p.priority)
        
        return sorted(active_providers, key=sort_key)
    
    def _is_rate_limit_error(self, error: Exception) -> bool:
        """Check if error is a rate limit (429) error."""
        error_str = str(error).lower()
        if "429" in error_str:
            return True
        if "rate limit" in error_str:
            return True
        if "quota" in error_str:
            return True
        if "too many requests" in error_str:
            return True
        if isinstance(error, ProviderError) and error.status_code == 429:
            return True
        return False
    
    def _mark_rate_limited(self, provider_id: str, reason: str):
        """Mark a provider as rate-limited for this session."""
        self._rate_limited[provider_id] = True
        self._inactive_reason[provider_id] = reason
        logger.warning(f"Provider {provider_id} marked inactive (rate-limited): {reason}")
    
    def reset_provider(self, provider_id: str):
        """Manually reset a provider's rate-limit status."""
        self._rate_limited[provider_id] = False
        self._inactive_reason.pop(provider_id, None)
        self._failure_counts[provider_id] = 0
        logger.info(f"Provider {provider_id} reset to active")
    
    def reset_all_providers(self):
        """Reset all providers to active status."""
        self._rate_limited.clear()
        self._inactive_reason.clear()
        self._failure_counts.clear()
        logger.info("All providers reset to active")
    
    def _get_provider_instance(self, config: ProviderConfig) -> BaseProvider:
        """Get or create provider instance."""
        if config.id in self._provider_cache:
            return self._provider_cache[config.id]
        
        # Import and instantiate
        provider = self._create_provider(config)
        self._provider_cache[config.id] = provider
        return provider
    
    def _create_provider(self, config: ProviderConfig) -> BaseProvider:
        """Create a provider instance."""
        # Lazy imports to avoid circular dependencies
        if config.id == "gemini":
            from .gemini import GeminiProvider
            return GeminiProvider(
                model=config.default_model,
                api_key=config.get_api_key(),
            )
        elif config.id == "cerebras":
            from .cerebras import CerebrasProvider
            return CerebrasProvider(
                model=config.default_model,
                api_key=config.get_api_key(),
            )
        elif config.id == "groq":
            from .groq import GroqProvider
            return GroqProvider(
                model=config.default_model,
                api_key=config.get_api_key(),
            )
        elif config.id == "openrouter":
            from .openrouter import OpenRouterProvider
            return OpenRouterProvider(
                model=config.default_model,
                api_key=config.get_api_key(),
            )
        elif config.id == "lightning":
            from .lightning import LightningProvider
            return LightningProvider(
                model=config.default_model,
                api_key=config.get_api_key(),
            )
        elif config.id == "github_models":
            from .github_models import GitHubModelsProvider
            return GitHubModelsProvider(
                model=config.default_model,
                api_key=config.get_api_key(),
            )
        elif config.id == "openai":
            from .openai import OpenAIProvider
            return OpenAIProvider(
                model=config.default_model,
                api_key=config.get_api_key(),
            )
        elif config.id == "anthropic":
            from .anthropic import AnthropicProvider
            return AnthropicProvider(
                model=config.default_model,
                api_key=config.get_api_key(),
            )
        elif config.id == "ollama":
            from .ollama import OllamaProvider
            return OllamaProvider(
                model=config.default_model,
                base_url=config.base_url,
            )
        elif config.id == "vllm":
            from .vllm import VLLMProvider
            return VLLMProvider(
                model=config.default_model,
                base_url=config.base_url,
            )
        else:
            raise ValueError(f"Unknown provider: {config.id}")
    
    def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        preferred_provider: Optional[str] = None,
        preferred_model: Optional[str] = None,
        fallback: bool = True,
        **kwargs
    ) -> ProviderResponse:
        """
        Generate a completion using the best available provider.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system instructions
            preferred_provider: Prefer this provider if available
            preferred_model: Prefer this model
            fallback: Whether to try other providers on failure
            **kwargs: Additional parameters (temperature, max_tokens)
            
        Returns:
            ProviderResponse from the successful provider
            
        Raises:
            ProviderError: If all providers fail
        """
        providers = self._get_ordered_providers()
        
        # Move preferred provider to front if specified
        if preferred_provider:
            providers = sorted(
                providers,
                key=lambda p: 0 if p.id == preferred_provider else 1
            )
        
        errors = []
        attempts = 0
        
        for config in providers:
            for retry in range(self.max_retries):
                attempts += 1
                
                try:
                    provider = self._get_provider_instance(config)
                    
                    # Override model if preferred
                    if preferred_model:
                        provider.model = preferred_model
                    
                    start = time.time()
                    response = provider.complete(
                        prompt=prompt,
                        system_prompt=system_prompt,
                        **kwargs
                    )
                    response.latency_ms = (time.time() - start) * 1000
                    
                    # Clear failure count on success
                    self._failure_counts[config.id] = 0
                    
                    logger.info(
                        f"Completion successful with {config.id} "
                        f"({response.latency_ms:.0f}ms)"
                    )
                    
                    return response
                    
                except Exception as e:
                    error_msg = f"{config.id}: {str(e)}"
                    errors.append(error_msg)
                    logger.warning(f"Provider failed: {error_msg}")
                    
                    # Check for rate limit - mark inactive and cascade immediately
                    if self._is_rate_limit_error(e):
                        self._mark_rate_limited(config.id, str(e)[:100])
                        break  # Don't retry, cascade to next provider
                    
                    # Track failure
                    self._failure_counts[config.id] = \
                        self._failure_counts.get(config.id, 0) + 1
                    
                    # Non-retriable errors - cascade immediately
                    if isinstance(e, ProviderError) and not e.retriable:
                        break
            
            if not fallback:
                break
        
        # All failed
        raise ProviderError(
            provider="router",
            message=f"All providers failed after {attempts} attempts. "
                    f"Errors: {'; '.join(errors[-3:])}",  # Last 3 errors
            retriable=False,
        )
    
    def chat(
        self,
        messages: List[Union[Message, Dict]],
        preferred_provider: Optional[str] = None,
        preferred_model: Optional[str] = None,
        fallback: bool = True,
        **kwargs
    ) -> ProviderResponse:
        """
        Generate a chat response using the best available provider.
        
        Args:
            messages: List of Message objects or dicts
            preferred_provider: Prefer this provider if available
            preferred_model: Prefer this model
            fallback: Whether to try other providers on failure
            **kwargs: Additional parameters
            
        Returns:
            ProviderResponse from the successful provider
        """
        providers = self._get_ordered_providers()
        
        if preferred_provider:
            providers = sorted(
                providers,
                key=lambda p: 0 if p.id == preferred_provider else 1
            )
        
        # Normalize messages
        normalized = []
        for msg in messages:
            if isinstance(msg, Message):
                normalized.append(msg)
            elif isinstance(msg, dict):
                normalized.append(Message.from_dict(msg))
            else:
                raise ValueError(f"Invalid message type: {type(msg)}")
        
        errors = []
        attempts = 0
        
        for config in providers:
            for retry in range(self.max_retries):
                attempts += 1
                
                try:
                    provider = self._get_provider_instance(config)
                    
                    if preferred_model:
                        provider.model = preferred_model
                    
                    start = time.time()
                    response = provider.chat(normalized, **kwargs)
                    response.latency_ms = (time.time() - start) * 1000
                    
                    self._failure_counts[config.id] = 0
                    
                    return response
                    
                except Exception as e:
                    errors.append(f"{config.id}: {e}")
                    
                    # Check for rate limit - mark inactive and cascade immediately
                    if self._is_rate_limit_error(e):
                        self._mark_rate_limited(config.id, str(e)[:100])
                        break  # Cascade to next provider
                    
                    self._failure_counts[config.id] = \
                        self._failure_counts.get(config.id, 0) + 1
                    
                    if isinstance(e, ProviderError) and not e.retriable:
                        break
            
            if not fallback:
                break
        
        raise ProviderError(
            provider="router",
            message=f"All providers failed. Errors: {'; '.join(errors[-3:])}",
            retriable=False,
        )
    
    async def complete_async(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> ProviderResponse:
        """Async version of complete()."""
        import asyncio
        return await asyncio.to_thread(
            self.complete, prompt, system_prompt, **kwargs
        )
    
    async def chat_async(
        self,
        messages: List[Union[Message, Dict]],
        **kwargs
    ) -> ProviderResponse:
        """Async version of chat()."""
        import asyncio
        return await asyncio.to_thread(self.chat, messages, **kwargs)
    
    def get_status(self) -> Dict[str, Any]:
        """Get current router status including rate-limited providers."""
        all_providers = self.registry.list_providers(configured_only=True)
        
        status = []
        for config in all_providers:
            if config.id in self.skip_providers:
                continue
                
            try:
                provider = self._get_provider_instance(config)
                available = provider.is_available()
            except Exception:
                available = False
            
            is_rate_limited = self._rate_limited.get(config.id, False)
            
            status.append({
                "id": config.id,
                "name": config.name,
                "priority": config.priority,
                "available": available,
                "active": available and not is_rate_limited,
                "rate_limited": is_rate_limited,
                "rate_limit_reason": self._inactive_reason.get(config.id, ""),
                "failures": self._failure_counts.get(config.id, 0),
                "free_tier": config.free_tier,
            })
        
        active_count = sum(1 for s in status if s["active"])
        rate_limited_count = sum(1 for s in status if s["rate_limited"])
        
        return {
            "providers": status,
            "active_providers": active_count,
            "rate_limited_providers": rate_limited_count,
            "total_configured": len(status),
        }
    
    def print_status(self):
        """Print formatted status of all providers."""
        status = self.get_status()
        
        print("\n" + "=" * 60)
        print("PROVIDER CASCADE STATUS")
        print("=" * 60)
        
        for p in status["providers"]:
            if p["active"]:
                icon = "✅"
                state = "ACTIVE"
            elif p["rate_limited"]:
                icon = "⏸️"
                state = "RATE-LIMITED"
            elif p["available"]:
                icon = "⚠️"
                state = "AVAILABLE"
            else:
                icon = "❌"
                state = "UNAVAILABLE"
            
            print(f"  {icon} {p['id']:<15} {state:<15} (priority: {p['priority']})")
            if p["rate_limited"] and p["rate_limit_reason"]:
                print(f"      └─ {p['rate_limit_reason'][:50]}...")
        
        print()
        print(f"Active: {status['active_providers']}/{status['total_configured']}")
        if status['rate_limited_providers'] > 0:
            print(f"Rate-limited: {status['rate_limited_providers']} (will cascade to next)")
        print("=" * 60 + "\n")


# Global router instance
_router: Optional[ProviderRouter] = None


def get_router() -> ProviderRouter:
    """Get the global provider router."""
    global _router
    if _router is None:
        _router = ProviderRouter()
    return _router


def get_best_provider() -> Optional[str]:
    """Get the ID of the best available provider."""
    router = get_router()
    providers = router._get_ordered_providers()
    
    for config in providers:
        try:
            provider = router._get_provider_instance(config)
            if provider.is_available():
                return config.id
        except Exception:
            continue
    
    return None
