"""
Cerebras Provider
=================

FREE tier with VERY generous limits and fast inference.
Custom Wafer Scale Engine hardware for efficient inference.

Free Tier Limits (as of Dec 2025):
- ALL models: 14,400 req/day, 1M tokens/day, 60K tokens/min
- Qwen3-Coder-480B: 100 req/day (more restrictive)

Available Models:
- llama-3.3-70b (default, excellent quality)
- qwen3-235b-a22b (massive, high quality)
- qwen3-coder-480b (best for code, limited)
- gpt-oss-120b (open-source giant)
- llama-4-scout (latest Llama)

API Key: https://cloud.cerebras.ai/
"""

import os
import time
import logging
from typing import Optional, List, Dict, Any

from .base import BaseProvider, Message, ProviderResponse, ProviderError, Role

logger = logging.getLogger(__name__)


class CerebrasProvider(BaseProvider):
    """
    Provider for Cerebras Cloud API.
    
    Cerebras offers the most generous free tier of any provider.
    Uses Wafer Scale Engine for efficient, fast inference.
    
    Supported models:
        - llama-3.3-70b (default, excellent balance)
        - qwen3-235b-a22b (massive model, free!)
        - qwen3-coder-480b (best coding, lower limit)
        - gpt-oss-120b (open-source)
        - llama-4-scout (latest)
    
    Example:
        provider = CerebrasProvider()
        # Access to 235B parameter model for FREE!
        response = provider.complete(
            "Explain RNA-seq", 
            model="qwen3-235b-a22b"
        )
    """
    
    name = "cerebras"
    default_model = "llama-3.3-70b"
    supports_streaming = True
    
    # Model configurations
    MODELS = {
        "llama-3.3-70b": {
            "context": 128_000,
            "daily_requests": 14_400,
            "daily_tokens": 1_000_000,
        },
        "qwen3-235b-a22b": {
            "context": 128_000,
            "daily_requests": 14_400,
            "daily_tokens": 1_000_000,
        },
        "qwen3-coder-480b": {
            "context": 128_000,
            "daily_requests": 100,  # More restrictive
            "daily_tokens": 1_000_000,
        },
        "gpt-oss-120b": {
            "context": 128_000,
            "daily_requests": 14_400,
            "daily_tokens": 1_000_000,
        },
        "llama-4-scout": {
            "context": 128_000,
            "daily_requests": 14_400,
            "daily_tokens": 1_000_000,
        },
    }
    
    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs
    ):
        super().__init__(model=model, **kwargs)
        self.api_key = api_key or os.environ.get("CEREBRAS_API_KEY")
        self.base_url = base_url or "https://api.cerebras.ai/v1"
        self._client = None
    
    @property
    def client(self):
        """Lazy-load OpenAI client for Cerebras."""
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
        """Check if Cerebras is configured and reachable."""
        if not self.api_key:
            return False
        
        try:
            # Quick model list check
            models = self.client.models.list()
            return len(models.data) > 0
        except Exception as e:
            logger.debug(f"Cerebras availability check failed: {e}")
            return False
    
    def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> ProviderResponse:
        """Generate completion using Cerebras."""
        messages = self._build_messages(prompt, system_prompt)
        return self.chat(messages, **kwargs)
    
    def chat(
        self,
        messages: List[Message],
        **kwargs
    ) -> ProviderResponse:
        """Generate chat response using Cerebras."""
        if not self.api_key:
            raise ProviderError(
                self.name,
                "CEREBRAS_API_KEY not set. Get one at https://cloud.cerebras.ai/",
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
        return self.MODELS.get(model, {
            "context": 8192, 
            "daily_requests": 100,
            "daily_tokens": 100_000,
        })
