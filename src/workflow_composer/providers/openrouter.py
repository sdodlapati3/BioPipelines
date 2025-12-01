"""
OpenRouter Provider
===================

Gateway to 400+ models via a single API endpoint.
FREE access to many models with ":free" suffix.

Free Tier Limits (as of Dec 2025):
- Base: 20 req/min, 50 req/day
- After $10 lifetime topup: 1000 req/day (non-free models)
- BYOK (Bring Your Own Key): 1M free req/month, then 5% fee

Free Models (suffix with :free):
- meta-llama/llama-3.3-70b-instruct:free
- qwen/qwen3-235b-a22b:free
- deepseek/deepseek-r1t-chimera:free
- google/gemma-3-27b-it:free
- mistralai/mistral-small-3.1-24b-instruct:free
- moonshotai/kimi-k2:free
- x-ai/grok-4.1-fast:free

API Key: https://openrouter.ai/settings/keys
"""

import os
import time
import logging
from typing import Optional, List, Dict, Any

from .base import BaseProvider, Message, ProviderResponse, ProviderError, Role

logger = logging.getLogger(__name__)


class OpenRouterProvider(BaseProvider):
    """
    Provider for OpenRouter API - Gateway to 400+ models.
    
    OpenRouter provides access to many models through a single API.
    Many models have free tiers (marked with :free suffix).
    
    Supported free models:
        - meta-llama/llama-3.3-70b-instruct:free (default)
        - qwen/qwen3-235b-a22b:free
        - deepseek/deepseek-r1t-chimera:free
        - google/gemma-3-27b-it:free
        - mistralai/mistral-small-3.1-24b-instruct:free
    
    Example:
        provider = OpenRouterProvider()
        
        # Use default free model
        response = provider.complete("Explain RNA-seq")
        
        # Or specify a free model
        response = provider.complete(
            "Explain RNA-seq",
            model="qwen/qwen3-235b-a22b:free"
        )
    """
    
    name = "openrouter"
    default_model = "meta-llama/llama-3.3-70b-instruct:free"
    supports_streaming = True
    
    # Free models available on OpenRouter
    FREE_MODELS = [
        "meta-llama/llama-3.3-70b-instruct:free",
        "qwen/qwen3-235b-a22b:free",
        "qwen/qwen3-30b-a3b:free",
        "qwen/qwen3-coder:free",
        "deepseek/deepseek-r1t-chimera:free",
        "google/gemma-3-27b-it:free",
        "mistralai/mistral-small-3.1-24b-instruct:free",
        "moonshotai/kimi-k2:free",
        "x-ai/grok-4.1-fast:free",
        "openai/gpt-oss-20b:free",
        "nousresearch/deephermes-3-llama-3-8b:free",
    ]
    
    # Model aliases for easier usage
    MODEL_ALIASES = {
        "llama-70b": "meta-llama/llama-3.3-70b-instruct:free",
        "qwen-235b": "qwen/qwen3-235b-a22b:free",
        "deepseek-r1": "deepseek/deepseek-r1t-chimera:free",
        "gemma-27b": "google/gemma-3-27b-it:free",
        "mistral-small": "mistralai/mistral-small-3.1-24b-instruct:free",
        "kimi": "moonshotai/kimi-k2:free",
        "grok": "x-ai/grok-4.1-fast:free",
    }
    
    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        app_name: str = "BioPipelines",
        app_url: str = "https://github.com/sdodlapa/BioPipelines",
        **kwargs
    ):
        super().__init__(model=model, **kwargs)
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        self.base_url = base_url or "https://openrouter.ai/api/v1"
        self.app_name = app_name
        self.app_url = app_url
        self._client = None
    
    @property
    def client(self):
        """Lazy-load OpenAI client for OpenRouter."""
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(
                    base_url=self.base_url,
                    api_key=self.api_key,
                    default_headers={
                        "HTTP-Referer": self.app_url,
                        "X-Title": self.app_name,
                    }
                )
            except ImportError:
                raise ProviderError(
                    self.name,
                    "openai package required. Install with: pip install openai",
                    retriable=False,
                )
        return self._client
    
    def is_available(self) -> bool:
        """Check if OpenRouter is configured and reachable."""
        if not self.api_key:
            return False
        
        try:
            # OpenRouter-specific: check with auth
            import requests
            resp = requests.get(
                f"{self.base_url}/auth/key",
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=5,
            )
            return resp.status_code == 200
        except Exception as e:
            logger.debug(f"OpenRouter availability check failed: {e}")
            return False
    
    def _resolve_model(self, model: Optional[str] = None) -> str:
        """Resolve model name, handling aliases."""
        model = model or self.model
        return self.MODEL_ALIASES.get(model, model)
    
    def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> ProviderResponse:
        """Generate completion using OpenRouter."""
        messages = self._build_messages(prompt, system_prompt)
        return self.chat(messages, **kwargs)
    
    def chat(
        self,
        messages: List[Message],
        **kwargs
    ) -> ProviderResponse:
        """Generate chat response using OpenRouter."""
        if not self.api_key:
            raise ProviderError(
                self.name,
                "OPENROUTER_API_KEY not set. Get one at https://openrouter.ai/settings/keys",
                retriable=False,
            )
        
        messages = self._normalize_messages(messages)
        
        # Convert to OpenAI format
        openai_messages = [msg.to_dict() for msg in messages]
        
        model = self._resolve_model(kwargs.pop("model", None))
        
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
    
    def list_free_models(self) -> List[str]:
        """List available free models."""
        return self.FREE_MODELS.copy()
    
    def get_credits(self) -> Optional[Dict[str, Any]]:
        """Get current credit balance."""
        if not self.api_key:
            return None
        
        try:
            import requests
            resp = requests.get(
                f"{self.base_url}/auth/key",
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=5,
            )
            if resp.status_code == 200:
                data = resp.json().get("data", {})
                return {
                    "credits": data.get("limit", 0) - data.get("usage", 0),
                    "limit": data.get("limit", 0),
                    "usage": data.get("usage", 0),
                }
        except Exception as e:
            logger.debug(f"Failed to get OpenRouter credits: {e}")
        
        return None
