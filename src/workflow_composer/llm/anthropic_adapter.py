"""
Anthropic LLM Adapter
=====================

Adapter for Anthropic Claude models.

Requirements:
    - Anthropic API key: Set ANTHROPIC_API_KEY environment variable

Usage:
    from workflow_composer.llm import AnthropicAdapter
    
    llm = AnthropicAdapter(model="claude-3-opus-20240229")
    response = llm.complete("Explain RNA-seq")
"""

import json
import os
import logging
from typing import List, Iterator, Optional, Dict, Any
import urllib.request
import urllib.error

from .base import LLMAdapter, LLMResponse, Message, Role

logger = logging.getLogger(__name__)


class AnthropicAdapter(LLMAdapter):
    """
    Adapter for Anthropic Claude API.
    
    Supports:
    - claude-3-opus-20240229 (most capable)
    - claude-3-sonnet-20240229 (balanced)
    - claude-3-haiku-20240307 (fast, cost-effective)
    - claude-3-5-sonnet-20241022 (latest)
    
    Requires ANTHROPIC_API_KEY environment variable.
    """
    
    API_BASE = "https://api.anthropic.com/v1"
    API_VERSION = "2023-06-01"
    
    def __init__(
        self,
        model: str = "claude-3-5-sonnet-20241022",
        api_key: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 4096,
        **kwargs
    ):
        """
        Initialize Anthropic adapter.
        
        Args:
            model: Claude model name
            api_key: Anthropic API key (default: from ANTHROPIC_API_KEY env var)
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
        """
        super().__init__(model=model, temperature=temperature, max_tokens=max_tokens, **kwargs)
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        
        if not self.api_key:
            logger.warning("No Anthropic API key provided. Set ANTHROPIC_API_KEY environment variable.")
    
    @property
    def provider_name(self) -> str:
        return "anthropic"
    
    def _make_request(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Make HTTP request to Anthropic API."""
        url = f"{self.API_BASE}{endpoint}"
        
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": self.API_VERSION
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
            logger.error(f"Anthropic API error: {e.code} - {error_body}")
            raise
        except urllib.error.URLError as e:
            logger.error(f"Anthropic request failed: {e}")
            raise ConnectionError(f"Failed to connect to Anthropic API") from e
    
    def complete(self, prompt: str, **kwargs) -> LLMResponse:
        """
        Generate a completion.
        
        Args:
            prompt: The input prompt
            **kwargs: Additional parameters
            
        Returns:
            LLMResponse with generated text
        """
        messages = [Message.user(prompt)]
        return self.chat(messages, **kwargs)
    
    def chat(self, messages: List[Message], **kwargs) -> LLMResponse:
        """
        Generate a chat response.
        
        Args:
            messages: List of Message objects
            **kwargs: Additional parameters
            
        Returns:
            LLMResponse with assistant's response
        """
        if not self.api_key:
            raise ValueError("Anthropic API key not configured")
        
        # Anthropic separates system message from conversation
        system_message = ""
        conversation = []
        
        for msg in messages:
            if msg.role == Role.SYSTEM:
                system_message = msg.content
            else:
                conversation.append(msg.to_dict())
        
        data = {
            "model": self.model,
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "messages": conversation,
        }
        
        if system_message:
            data["system"] = system_message
        
        if "temperature" in kwargs or self.temperature != 0.1:
            data["temperature"] = kwargs.get("temperature", self.temperature)
        
        response = self._make_request("/messages", data)
        
        # Extract content from response
        content_blocks = response.get("content", [])
        content = "".join(
            block.get("text", "") 
            for block in content_blocks 
            if block.get("type") == "text"
        )
        
        usage = response.get("usage", {})
        
        return LLMResponse(
            content=content,
            model=response.get("model", self.model),
            provider=self.provider_name,
            tokens_used=usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
            finish_reason=response.get("stop_reason", "end_turn"),
            raw_response=response
        )
    
    def is_available(self) -> bool:
        """Check if Anthropic API is accessible."""
        if not self.api_key:
            return False
        
        try:
            # Try a minimal request
            response = self.complete("Hi", max_tokens=5)
            return bool(response.content)
        except Exception as e:
            logger.warning(f"Anthropic not available: {e}")
            return False
