"""
OpenAI LLM Adapter
==================

Adapter for OpenAI models (GPT-4, GPT-3.5-turbo, etc.)

Requirements:
    - OpenAI API key: Set OPENAI_API_KEY environment variable
    - Optional: openai package for enhanced features

Usage:
    from workflow_composer.llm import OpenAIAdapter
    
    llm = OpenAIAdapter(model="gpt-4-turbo-preview")
    response = llm.complete("Explain RNA-seq")
"""

import json
import os
import logging
from typing import List, Iterator, Optional, Dict, Any
import urllib.request
import urllib.error

from .base import LLMAdapter, LLMResponse, Message

logger = logging.getLogger(__name__)


class OpenAIAdapter(LLMAdapter):
    """
    Adapter for OpenAI API.
    
    Supports:
    - gpt-4-turbo-preview (recommended for complex tasks)
    - gpt-4 (high quality, slower)
    - gpt-3.5-turbo (fast, cost-effective)
    - gpt-3.5-turbo-16k (longer context)
    
    Requires OPENAI_API_KEY environment variable.
    """
    
    API_BASE = "https://api.openai.com/v1"
    
    def __init__(
        self,
        model: str = "gpt-4-turbo-preview",
        api_key: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 4096,
        api_base: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize OpenAI adapter.
        
        Args:
            model: OpenAI model name
            api_key: OpenAI API key (default: from OPENAI_API_KEY env var)
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            api_base: Custom API base URL (for Azure or proxies)
        """
        super().__init__(model=model, temperature=temperature, max_tokens=max_tokens, **kwargs)
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.api_base = (api_base or self.API_BASE).rstrip("/")
        
        if not self.api_key:
            logger.warning("No OpenAI API key provided. Set OPENAI_API_KEY environment variable.")
    
    @property
    def provider_name(self) -> str:
        return "openai"
    
    def _make_request(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Make HTTP request to OpenAI API."""
        url = f"{self.api_base}{endpoint}"
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        request = urllib.request.Request(
            url,
            data=json.dumps(data).encode("utf-8"),
            headers=headers,
            method="POST"
        )
        
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8")
            logger.error(f"OpenAI API error: {e.code} - {error_body}")
            raise
        except urllib.error.URLError as e:
            logger.error(f"OpenAI request failed: {e}")
            raise ConnectionError(f"Failed to connect to OpenAI API") from e
    
    def complete(self, prompt: str, **kwargs) -> LLMResponse:
        """
        Generate a completion using chat endpoint.
        
        Args:
            prompt: The input prompt
            **kwargs: Additional OpenAI parameters
            
        Returns:
            LLMResponse with generated text
        """
        # OpenAI's completion endpoint is deprecated, use chat
        messages = [Message.user(prompt)]
        return self.chat(messages, **kwargs)
    
    def chat(self, messages: List[Message], **kwargs) -> LLMResponse:
        """
        Generate a chat response.
        
        Args:
            messages: List of Message objects
            **kwargs: Additional OpenAI parameters
            
        Returns:
            LLMResponse with assistant's response
        """
        if not self.api_key:
            raise ValueError("OpenAI API key not configured")
        
        data = {
            "model": self.model,
            "messages": [m.to_dict() for m in messages],
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
        }
        
        # Add optional parameters
        for key in ["top_p", "frequency_penalty", "presence_penalty", "stop", "seed"]:
            if key in kwargs:
                data[key] = kwargs[key]
        
        response = self._make_request("/chat/completions", data)
        
        choice = response.get("choices", [{}])[0]
        message = choice.get("message", {})
        usage = response.get("usage", {})
        
        return LLMResponse(
            content=message.get("content", ""),
            model=response.get("model", self.model),
            provider=self.provider_name,
            tokens_used=usage.get("total_tokens", 0),
            finish_reason=choice.get("finish_reason", "stop"),
            raw_response=response
        )
    
    def embed(self, text: str) -> List[float]:
        """
        Generate embeddings using OpenAI embeddings API.
        
        Args:
            text: Text to embed
            
        Returns:
            List of floats (embedding vector)
        """
        if not self.api_key:
            raise ValueError("OpenAI API key not configured")
        
        data = {
            "model": "text-embedding-3-small",
            "input": text
        }
        
        response = self._make_request("/embeddings", data)
        return response.get("data", [{}])[0].get("embedding", [])
    
    def is_available(self) -> bool:
        """Check if OpenAI API is accessible."""
        if not self.api_key:
            return False
        
        try:
            # List models to verify API key
            request = urllib.request.Request(
                f"{self.api_base}/models",
                headers={"Authorization": f"Bearer {self.api_key}"}
            )
            with urllib.request.urlopen(request, timeout=10) as response:
                return response.status == 200
        except Exception as e:
            logger.warning(f"OpenAI not available: {e}")
            return False
