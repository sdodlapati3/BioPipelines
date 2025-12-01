"""
Groq Provider
=============

FREE tier with generous limits and FAST inference.
Uses LPU (Language Processing Unit) for blazing-fast responses.

Free Tier Limits (as of Dec 2025):
- Llama 3.3 70B: 1,000 req/day, 12K tokens/min
- Llama 3.1 8B:  14,400 req/day, 6K tokens/min  
- GPT-OSS 120B:  1,000 req/day, 8K tokens/min
- Whisper:       2,000 req/day (audio)

API Key: https://console.groq.com/keys
"""

import os
import time
import logging
from typing import Optional, List, Dict, Any

from .base import BaseProvider, Message, ProviderResponse, ProviderError, Role

logger = logging.getLogger(__name__)


class GroqProvider(BaseProvider):
    """
    Provider for Groq Cloud API.
    
    Groq offers extremely fast inference using custom LPU hardware.
    Free tier is generous enough for most development use cases.
    
    Supported models:
        - llama-3.3-70b-versatile (default, best quality)
        - llama-3.1-8b-instant (fastest, high limit)
        - gpt-oss-120b (largest open-source model)
        - groq/compound (agentic with web search)
    
    Example:
        provider = GroqProvider()
        response = provider.complete("Explain RNA-seq")
        print(f"Response in {response.latency_ms}ms")  # Usually <1 sec!
    """
    
    name = "groq"
    default_model = "llama-3.3-70b-versatile"
    supports_streaming = True
    
    # Model configurations
    MODELS = {
        "llama-3.3-70b-versatile": {"context": 128_000, "daily_limit": 1_000},
        "llama-3.1-8b-instant": {"context": 128_000, "daily_limit": 14_400},
        "llama-3.1-70b-versatile": {"context": 128_000, "daily_limit": 1_000},
        "llama-3.2-3b-preview": {"context": 8_192, "daily_limit": 14_400},
        "gpt-oss-120b": {"context": 128_000, "daily_limit": 1_000},
        "gemma-2-9b-it": {"context": 8_192, "daily_limit": 14_400},
        "groq/compound": {"context": 70_000, "daily_limit": 250},  # Agentic
    }
    
    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs
    ):
        super().__init__(model=model, **kwargs)
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        self.base_url = base_url or "https://api.groq.com/openai/v1"
        self._client = None
    
    @property
    def client(self):
        """Lazy-load OpenAI client for Groq."""
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(
                    base_url=self.base_url,
                    api_key=self.api_key,
                )
            except ImportError:
                raise ProviderError(
                    self.name,
                    "openai package required. Install with: pip install openai",
                    retriable=False,
                )
        return self._client
    
    def is_available(self) -> bool:
        """Check if Groq is configured and reachable."""
        if not self.api_key:
            return False
        
        try:
            # Quick model list check
            models = self.client.models.list()
            return len(models.data) > 0
        except Exception as e:
            logger.debug(f"Groq availability check failed: {e}")
            return False
    
    def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> ProviderResponse:
        """Generate completion using Groq."""
        messages = self._build_messages(prompt, system_prompt)
        return self.chat(messages, **kwargs)
    
    def chat(
        self,
        messages: List[Message],
        **kwargs
    ) -> ProviderResponse:
        """Generate chat response using Groq."""
        if not self.api_key:
            raise ProviderError(
                self.name,
                "GROQ_API_KEY not set. Get one at https://console.groq.com/keys",
                retriable=False,
            )
        
        messages = self._normalize_messages(messages)
        
        # Convert to OpenAI format
        openai_messages = [msg.to_dict() for msg in messages]
        
        model = kwargs.pop("model", self.model)
        
        start = time.time()
        
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=openai_messages,
                temperature=kwargs.get("temperature", self.temperature),
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
                **{k: v for k, v in kwargs.items() 
                   if k not in ["temperature", "max_tokens"]},
            )
            
            content = response.choices[0].message.content or ""
            usage = response.usage
            
            return ProviderResponse(
                content=content,
                provider=self.name,
                model=model,
                tokens_used=usage.total_tokens if usage else 0,
                prompt_tokens=usage.prompt_tokens if usage else 0,
                completion_tokens=usage.completion_tokens if usage else 0,
                latency_ms=(time.time() - start) * 1000,
                finish_reason=response.choices[0].finish_reason or "stop",
                raw_response=response.model_dump() if hasattr(response, 'model_dump') else None,
            )
            
        except Exception as e:
            error_str = str(e).lower()
            is_rate_limit = "429" in str(e) or "rate" in error_str or "quota" in error_str
            
            raise ProviderError(
                self.name,
                f"API error: {e}",
                retriable=is_rate_limit,
                status_code=429 if is_rate_limit else None,
            )
    
    def get_model_info(self, model: Optional[str] = None) -> Dict[str, Any]:
        """Get information about a model."""
        model = model or self.model
        return self.MODELS.get(model, {"context": 8192, "daily_limit": 100})
