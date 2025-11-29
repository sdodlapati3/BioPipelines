"""
Google Gemini Provider Implementation.

Gemini provides free access to powerful models with rate limits.
"""

import os
import time
import aiohttp
from typing import Optional, Dict, Any

from .base import BaseProvider
from ..registry import ProviderConfig


class GeminiProvider(BaseProvider):
    """Provider for Google Gemini API."""
    
    DEFAULT_MODEL = "gemini-2.0-flash"
    
    MODELS = {
        "gemini-2.0-flash": "models/gemini-2.0-flash",
        "gemini-2.5-pro": "models/gemini-2.5-pro-preview-06-05",
        "gemini-2.5-flash": "models/gemini-2.5-flash-preview-05-20",
    }
    
    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self.api_key = os.environ.get(config.env_key or "GOOGLE_API_KEY")
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"
    
    async def complete_async(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        **kwargs,
    ) -> Dict[str, Any]:
        """Generate completion using Gemini."""
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY not set")
        
        model_name = model or self.DEFAULT_MODEL
        model_path = self.MODELS.get(model_name, f"models/{model_name}")
        
        # Build request
        contents = []
        
        if system_prompt:
            contents.append({
                "role": "user",
                "parts": [{"text": f"System instruction: {system_prompt}"}],
            })
            contents.append({
                "role": "model",
                "parts": [{"text": "Understood. I will follow these instructions."}],
            })
        
        contents.append({
            "role": "user",
            "parts": [{"text": prompt}],
        })
        
        payload = {
            "contents": contents,
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "temperature": temperature,
            },
        }
        
        url = f"{self.base_url}/{model_path}:generateContent"
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                params={"key": self.api_key},
                json=payload,
                headers={"Content-Type": "application/json"},
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Gemini API error: {error_text}")
                
                data = await response.json()
        
        # Extract response
        try:
            content = data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError) as e:
            raise Exception(f"Invalid Gemini response: {data}")
        
        # Token usage
        usage = data.get("usageMetadata", {})
        tokens_used = usage.get("totalTokenCount", 0)
        
        return {
            "content": content,
            "tokens_used": tokens_used,
            "model": model_name,
            "raw_response": data,
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Check if Gemini is available."""
        if not self.api_key:
            return {
                "available": False,
                "error": "API key not configured",
            }
        
        try:
            start = time.time()
            
            # List models check
            url = f"{self.base_url}/models"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url,
                    params={"key": self.api_key},
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
