"""
Abstract base class for LLM adapters.

All LLM providers must implement this interface to ensure
consistent behavior across different backends.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Iterator
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class Role(Enum):
    """Message roles for chat completions."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass
class Message:
    """A single message in a conversation."""
    role: Role
    content: str
    
    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary for API calls."""
        return {
            "role": self.role.value,
            "content": self.content
        }
    
    @classmethod
    def system(cls, content: str) -> "Message":
        """Create a system message."""
        return cls(Role.SYSTEM, content)
    
    @classmethod
    def user(cls, content: str) -> "Message":
        """Create a user message."""
        return cls(Role.USER, content)
    
    @classmethod
    def assistant(cls, content: str) -> "Message":
        """Create an assistant message."""
        return cls(Role.ASSISTANT, content)


@dataclass
class LLMResponse:
    """Response from an LLM."""
    content: str
    model: str
    provider: str
    tokens_used: int = 0
    finish_reason: str = "stop"
    raw_response: Optional[Dict[str, Any]] = None
    
    @property
    def text(self) -> str:
        """Alias for content."""
        return self.content


class LLMAdapter(ABC):
    """
    Abstract base class for LLM adapters.
    
    All LLM providers (OpenAI, Anthropic, Ollama, etc.) must implement
    this interface to be used with the Workflow Composer.
    
    Attributes:
        model: The model identifier
        temperature: Sampling temperature (0.0 - 1.0)
        max_tokens: Maximum tokens in response
        provider_name: Name of the provider (e.g., "openai", "ollama")
    """
    
    def __init__(
        self,
        model: str,
        temperature: float = 0.1,
        max_tokens: int = 4096,
        **kwargs
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._kwargs = kwargs
    
    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider name."""
        pass
    
    @abstractmethod
    def complete(self, prompt: str, **kwargs) -> LLMResponse:
        """
        Generate a completion for a single prompt.
        
        Args:
            prompt: The input prompt
            **kwargs: Additional provider-specific parameters
            
        Returns:
            LLMResponse containing the generated text
        """
        pass
    
    @abstractmethod
    def chat(self, messages: List[Message], **kwargs) -> LLMResponse:
        """
        Generate a response in a multi-turn conversation.
        
        Args:
            messages: List of Message objects representing the conversation
            **kwargs: Additional provider-specific parameters
            
        Returns:
            LLMResponse containing the assistant's response
        """
        pass
    
    def stream(self, prompt: str, **kwargs) -> Iterator[str]:
        """
        Stream a completion token by token.
        
        Default implementation calls complete() and yields the full response.
        Override for true streaming support.
        
        Args:
            prompt: The input prompt
            **kwargs: Additional parameters
            
        Yields:
            String chunks of the response
        """
        response = self.complete(prompt, **kwargs)
        yield response.content
    
    def chat_stream(self, messages: List[Message], **kwargs) -> Iterator[str]:
        """
        Stream a chat response token by token.
        
        Default implementation calls chat() and yields the full response.
        Override for true streaming support.
        
        Args:
            messages: Conversation messages
            **kwargs: Additional parameters
            
        Yields:
            String chunks of the response
        """
        response = self.chat(messages, **kwargs)
        yield response.content
    
    def embed(self, text: str) -> List[float]:
        """
        Generate embeddings for text.
        
        Not all providers support embeddings. Default raises NotImplementedError.
        
        Args:
            text: Text to embed
            
        Returns:
            List of floats representing the embedding vector
        """
        raise NotImplementedError(f"{self.provider_name} does not support embeddings")
    
    def is_available(self) -> bool:
        """
        Check if the LLM provider is available and properly configured.
        
        Returns:
            True if the provider can be used, False otherwise
        """
        try:
            # Try a minimal completion
            response = self.complete("Hi", max_tokens=5)
            return bool(response.content)
        except Exception as e:
            logger.warning(f"LLM availability check failed: {e}")
            return False
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model!r}, provider={self.provider_name!r})"


class MockLLMAdapter(LLMAdapter):
    """
    Mock LLM adapter for testing.
    
    Returns predefined responses based on keywords in the prompt.
    """
    
    def __init__(self, responses: Optional[Dict[str, str]] = None, **kwargs):
        super().__init__(model="mock", **kwargs)
        self.responses = responses or {}
        self.call_history: List[str] = []
    
    @property
    def provider_name(self) -> str:
        return "mock"
    
    def complete(self, prompt: str, **kwargs) -> LLMResponse:
        self.call_history.append(prompt)
        
        # Check for keyword matches
        for keyword, response in self.responses.items():
            if keyword.lower() in prompt.lower():
                return LLMResponse(
                    content=response,
                    model=self.model,
                    provider=self.provider_name
                )
        
        # Default response
        return LLMResponse(
            content="Mock response for testing",
            model=self.model,
            provider=self.provider_name
        )
    
    def chat(self, messages: List[Message], **kwargs) -> LLMResponse:
        # Use the last user message as the prompt
        last_user_msg = next(
            (m.content for m in reversed(messages) if m.role == Role.USER),
            ""
        )
        return self.complete(last_user_msg, **kwargs)
