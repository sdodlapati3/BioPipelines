"""
LLM Factory
===========

Factory for creating LLM adapters based on provider name.

Usage:
    from workflow_composer.llm import get_llm, LLMFactory
    
    # Simple factory function
    llm = get_llm("ollama", model="llama3:8b")
    
    # Using factory class
    factory = LLMFactory()
    factory.register("custom", CustomAdapter)
    llm = factory.create("custom", model="my-model")
"""

import logging
from typing import Dict, Type, Optional, Any

from .base import LLMAdapter
from .ollama_adapter import OllamaAdapter
from .openai_adapter import OpenAIAdapter
from .anthropic_adapter import AnthropicAdapter
from .huggingface_adapter import HuggingFaceAdapter

logger = logging.getLogger(__name__)


class LLMFactory:
    """
    Factory for creating LLM adapters.
    
    Supports registration of custom adapters for extensibility.
    """
    
    # Default provider registry
    _providers: Dict[str, Type[LLMAdapter]] = {
        "ollama": OllamaAdapter,
        "openai": OpenAIAdapter,
        "anthropic": AnthropicAdapter,
        "huggingface": HuggingFaceAdapter,
        "hf": HuggingFaceAdapter,  # Alias
    }
    
    def __init__(self):
        """Initialize factory with default providers."""
        self.providers = self._providers.copy()
    
    def register(self, name: str, adapter_class: Type[LLMAdapter]) -> None:
        """
        Register a custom LLM adapter.
        
        Args:
            name: Provider name for lookup
            adapter_class: LLMAdapter subclass
        """
        if not issubclass(adapter_class, LLMAdapter):
            raise TypeError(f"{adapter_class} must be a subclass of LLMAdapter")
        
        self.providers[name.lower()] = adapter_class
        logger.info(f"Registered LLM provider: {name}")
    
    def create(
        self,
        provider: str,
        model: Optional[str] = None,
        **kwargs
    ) -> LLMAdapter:
        """
        Create an LLM adapter instance.
        
        Args:
            provider: Provider name (ollama, openai, anthropic, huggingface)
            model: Model name (uses provider default if not specified)
            **kwargs: Additional arguments passed to adapter constructor
            
        Returns:
            Configured LLMAdapter instance
            
        Raises:
            ValueError: If provider is not registered
        """
        provider_lower = provider.lower()
        
        if provider_lower not in self.providers:
            available = ", ".join(self.providers.keys())
            raise ValueError(
                f"Unknown LLM provider: {provider}. "
                f"Available providers: {available}"
            )
        
        adapter_class = self.providers[provider_lower]
        
        # Set default models per provider
        if model is None:
            defaults = {
                "ollama": "llama3:8b",
                "openai": "gpt-4-turbo-preview",
                "anthropic": "claude-3-5-sonnet-20241022",
                "huggingface": "meta-llama/Llama-3-8b-chat-hf",
            }
            model = defaults.get(provider_lower, "")
        
        return adapter_class(model=model, **kwargs)
    
    def list_providers(self) -> Dict[str, str]:
        """
        List available providers with their adapter class names.
        
        Returns:
            Dict mapping provider names to class names
        """
        return {
            name: cls.__name__
            for name, cls in self.providers.items()
        }
    
    def get_available_providers(self) -> Dict[str, bool]:
        """
        Check which providers are currently available.
        
        Returns:
            Dict mapping provider names to availability status
        """
        status = {}
        for name in self.providers:
            try:
                adapter = self.create(name)
                status[name] = adapter.is_available()
            except Exception:
                status[name] = False
        return status


# Global factory instance
_factory = LLMFactory()


def get_llm(
    provider: str = "ollama",
    model: Optional[str] = None,
    **kwargs
) -> LLMAdapter:
    """
    Convenience function to create an LLM adapter.
    
    Args:
        provider: Provider name (ollama, openai, anthropic, huggingface)
        model: Model name (optional, uses provider default)
        **kwargs: Additional arguments for the adapter
        
    Returns:
        Configured LLMAdapter instance
        
    Examples:
        # Local Ollama
        llm = get_llm("ollama", model="llama3:8b")
        
        # OpenAI GPT-4
        llm = get_llm("openai", model="gpt-4-turbo-preview")
        
        # Anthropic Claude
        llm = get_llm("anthropic", model="claude-3-opus-20240229")
        
        # HuggingFace (API)
        llm = get_llm("huggingface", model="meta-llama/Llama-3-8b-chat-hf")
    """
    return _factory.create(provider, model, **kwargs)


def register_provider(name: str, adapter_class: Type[LLMAdapter]) -> None:
    """
    Register a custom LLM provider globally.
    
    Args:
        name: Provider name for lookup
        adapter_class: LLMAdapter subclass
    """
    _factory.register(name, adapter_class)


def list_providers() -> Dict[str, str]:
    """List all registered LLM providers."""
    return _factory.list_providers()


def check_providers() -> Dict[str, bool]:
    """Check availability of all registered providers."""
    return _factory.get_available_providers()
