"""
Base provider interface for all model providers.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any

from ..registry import ProviderConfig


class BaseProvider(ABC):
    """Abstract base class for all model providers."""
    
    def __init__(self, config: ProviderConfig):
        """
        Initialize the provider.
        
        Args:
            config: Provider configuration
        """
        self.config = config
        self._initialized = False
    
    @abstractmethod
    async def complete_async(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Generate a completion asynchronously.
        
        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt
            model: Model to use (provider-specific)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional provider-specific arguments
            
        Returns:
            Dictionary with:
                - content: The generated text
                - tokens_used: Number of tokens consumed
                - model: Model used
                - raw_response: Raw API response (optional)
        """
        pass
    
    def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Synchronous completion wrapper.
        
        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt
            **kwargs: Additional arguments
            
        Returns:
            Completion result dictionary
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
    
    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """
        Check if the provider is healthy.
        
        Returns:
            Dictionary with:
                - available: bool
                - latency_ms: float (optional)
                - error: str (if not available)
        """
        pass
    
    def is_available(self) -> bool:
        """Check if this provider is available."""
        return self.config.is_available()
    
    def _build_messages(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> list:
        """Build message list for chat-style APIs."""
        messages = []
        
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt,
            })
        
        messages.append({
            "role": "user",
            "content": prompt,
        })
        
        return messages
