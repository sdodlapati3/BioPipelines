"""
Lightning.ai Provider Implementation.

Lightning.ai provides free access to various models with a generous token limit.
"""

import os
import time
import aiohttp
from typing import Optional, Dict, Any

from .base import BaseProvider
from ..registry import ProviderConfig


class LightningProvider(BaseProvider):
    """Provider for Lightning.ai API."""
    
    DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
    
    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self.api_key = os.environ.get(config.env_key or "LIGHTNING_API_KEY")
        self.base_url = config.base_url or "https://api.lightning.ai/v1"
    
    async def complete_async(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        **kwargs,
    ) -> Dict[str, Any]:
        """Generate completion using Lightning.ai."""
        if not self.api_key:
            raise ValueError("LIGHTNING_API_KEY not set")
        
        model = model or self.DEFAULT_MODEL
        messages = self._build_messages(prompt, system_prompt)
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Lightning.ai API error: {error_text}")
                
                data = await response.json()
        
        # Extract response
        content = data["choices"][0]["message"]["content"]
        tokens_used = data.get("usage", {}).get("total_tokens", 0)
        
        return {
            "content": content,
            "tokens_used": tokens_used,
            "model": model,
            "raw_response": data,
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Check if Lightning.ai is available."""
        if not self.api_key:
            return {
                "available": False,
                "error": "API key not configured",
            }
        
        try:
            start = time.time()
            
            # Simple models list check
            headers = {
                "Authorization": f"Bearer {self.api_key}",
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/models",
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as response:
                    latency = (time.time() - start) * 1000
                    
                    if response.status == 200:
                        return {
                            "available": True,
                            "latency_ms": latency,
                        }
                    else:
                        return {
                            "available": False,
                            "error": f"HTTP {response.status}",
                            "latency_ms": latency,
                        }
        except Exception as e:
            return {
                "available": False,
                "error": str(e),
            }
