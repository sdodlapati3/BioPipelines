"""
HuggingFace LLM Adapter
=======================

Adapter for HuggingFace models via the Inference API or local transformers.

Supports:
- HuggingFace Inference API (cloud)
- Local transformers (requires torch, transformers packages)

Usage:
    from workflow_composer.llm import HuggingFaceAdapter
    
    # Using Inference API
    llm = HuggingFaceAdapter(model="meta-llama/Llama-3-8b-chat-hf", use_api=True)
    
    # Using local transformers
    llm = HuggingFaceAdapter(model="meta-llama/Llama-3-8b-chat-hf", use_api=False)
"""

import json
import os
import logging
from typing import List, Optional, Dict, Any
import urllib.request
import urllib.error

from .base import LLMAdapter, LLMResponse, Message

logger = logging.getLogger(__name__)


class HuggingFaceAdapter(LLMAdapter):
    """
    Adapter for HuggingFace models.
    
    Can use either:
    1. HuggingFace Inference API (cloud, requires HF_TOKEN)
    2. Local transformers library (requires transformers, torch)
    
    Recommended models:
    - meta-llama/Llama-3-8b-chat-hf
    - mistralai/Mistral-7B-Instruct-v0.2
    - microsoft/Phi-3-mini-4k-instruct
    - google/gemma-7b-it
    """
    
    API_BASE = "https://api-inference.huggingface.co/models"
    
    def __init__(
        self,
        model: str = "meta-llama/Llama-3-8b-chat-hf",
        token: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 4096,
        use_api: bool = True,
        device: str = "auto",
        **kwargs
    ):
        """
        Initialize HuggingFace adapter.
        
        Args:
            model: HuggingFace model ID
            token: HuggingFace API token (default: from HF_TOKEN env var)
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            use_api: Use Inference API (True) or local transformers (False)
            device: Device for local inference ("auto", "cuda", "cpu")
        """
        super().__init__(model=model, temperature=temperature, max_tokens=max_tokens, **kwargs)
        self.token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
        self.use_api = use_api
        self.device = device
        
        self._pipeline = None  # Lazy load for local inference
        
        if not self.token and use_api:
            logger.warning("No HuggingFace token provided. Set HF_TOKEN environment variable.")
    
    @property
    def provider_name(self) -> str:
        return "huggingface"
    
    def _get_pipeline(self):
        """Lazy load transformers pipeline for local inference."""
        if self._pipeline is None:
            try:
                from transformers import pipeline
                import torch
                
                device_map = self.device
                if device_map == "auto":
                    device_map = "cuda" if torch.cuda.is_available() else "cpu"
                
                self._pipeline = pipeline(
                    "text-generation",
                    model=self.model,
                    device_map=device_map,
                    torch_dtype=torch.float16 if device_map == "cuda" else torch.float32,
                    token=self.token
                )
            except ImportError:
                raise ImportError(
                    "Local HuggingFace inference requires transformers and torch. "
                    "Install with: pip install transformers torch"
                )
        return self._pipeline
    
    def _api_request(self, prompt: str, **kwargs) -> str:
        """Make request to HuggingFace Inference API."""
        url = f"{self.API_BASE}/{self.model}"
        
        headers = {
            "Content-Type": "application/json",
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        
        data = {
            "inputs": prompt,
            "parameters": {
                "temperature": kwargs.get("temperature", self.temperature),
                "max_new_tokens": kwargs.get("max_tokens", self.max_tokens),
                "return_full_text": False,
            }
        }
        
        request = urllib.request.Request(
            url,
            data=json.dumps(data).encode("utf-8"),
            headers=headers,
            method="POST"
        )
        
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                result = json.loads(response.read().decode("utf-8"))
                if isinstance(result, list) and len(result) > 0:
                    return result[0].get("generated_text", "")
                return str(result)
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8")
            logger.error(f"HuggingFace API error: {e.code} - {error_body}")
            raise
    
    def _local_generate(self, prompt: str, **kwargs) -> str:
        """Generate using local transformers pipeline."""
        pipe = self._get_pipeline()
        
        result = pipe(
            prompt,
            max_new_tokens=kwargs.get("max_tokens", self.max_tokens),
            temperature=kwargs.get("temperature", self.temperature),
            do_sample=True,
            return_full_text=False
        )
        
        if result and len(result) > 0:
            return result[0].get("generated_text", "")
        return ""
    
    def complete(self, prompt: str, **kwargs) -> LLMResponse:
        """
        Generate a completion.
        
        Args:
            prompt: The input prompt
            **kwargs: Additional parameters
            
        Returns:
            LLMResponse with generated text
        """
        if self.use_api:
            content = self._api_request(prompt, **kwargs)
        else:
            content = self._local_generate(prompt, **kwargs)
        
        return LLMResponse(
            content=content,
            model=self.model,
            provider=self.provider_name,
            tokens_used=0,  # Not easily available
            finish_reason="stop"
        )
    
    def chat(self, messages: List[Message], **kwargs) -> LLMResponse:
        """
        Generate a chat response.
        
        Formats messages into a prompt and uses completion.
        
        Args:
            messages: List of Message objects
            **kwargs: Additional parameters
            
        Returns:
            LLMResponse with assistant's response
        """
        # Format messages into a prompt
        # Use chat template if available, otherwise simple format
        prompt_parts = []
        for msg in messages:
            if msg.role.value == "system":
                prompt_parts.append(f"<|system|>\n{msg.content}</s>")
            elif msg.role.value == "user":
                prompt_parts.append(f"<|user|>\n{msg.content}</s>")
            elif msg.role.value == "assistant":
                prompt_parts.append(f"<|assistant|>\n{msg.content}</s>")
        
        prompt_parts.append("<|assistant|>\n")
        prompt = "\n".join(prompt_parts)
        
        return self.complete(prompt, **kwargs)
    
    def is_available(self) -> bool:
        """Check if HuggingFace is accessible."""
        if self.use_api:
            if not self.token:
                return False
            try:
                # Try a minimal request
                self._api_request("Hi", max_tokens=5)
                return True
            except Exception as e:
                logger.warning(f"HuggingFace API not available: {e}")
                return False
        else:
            try:
                import transformers
                import torch
                return True
            except ImportError:
                return False
