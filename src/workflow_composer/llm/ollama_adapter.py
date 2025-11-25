"""
Ollama LLM Adapter
==================

Adapter for local Ollama models (Llama3, Mistral, CodeLlama, etc.)

Ollama provides a simple way to run LLMs locally without API costs.
This is the recommended default for privacy-sensitive or offline use.

Requirements:
    - Ollama installed and running: https://ollama.ai
    - Model pulled: `ollama pull llama3:8b`

Usage:
    from workflow_composer.llm import OllamaAdapter
    
    llm = OllamaAdapter(model="llama3:8b")
    response = llm.complete("Explain RNA-seq")
"""

import json
import logging
from typing import List, Iterator, Optional, Dict, Any
import urllib.request
import urllib.error

from .base import LLMAdapter, LLMResponse, Message, Role

logger = logging.getLogger(__name__)


class OllamaAdapter(LLMAdapter):
    """
    Adapter for Ollama local LLM server.
    
    Ollama runs LLMs locally, providing:
    - No API costs
    - Privacy (data never leaves your machine)
    - Offline capability
    - Fast inference (especially with GPU)
    
    Supported models:
    - llama3:8b, llama3:70b (Meta Llama 3)
    - mistral:7b (Mistral AI)
    - codellama:13b (Code-specialized Llama)
    - mixtral:8x7b (Mistral MoE)
    - qwen2:7b (Alibaba Qwen)
    - phi3:mini (Microsoft Phi-3)
    
    See https://ollama.ai/library for full list.
    """
    
    DEFAULT_HOST = "http://localhost:11434"
    
    def __init__(
        self,
        model: str = "llama3:8b",
        host: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 4096,
        **kwargs
    ):
        """
        Initialize Ollama adapter.
        
        Args:
            model: Ollama model name (e.g., "llama3:8b", "codellama:13b")
            host: Ollama server URL (default: http://localhost:11434)
            temperature: Sampling temperature (0.0 - 1.0)
            max_tokens: Maximum tokens in response
        """
        super().__init__(model=model, temperature=temperature, max_tokens=max_tokens, **kwargs)
        self.host = (host or self.DEFAULT_HOST).rstrip("/")
    
    @property
    def provider_name(self) -> str:
        return "ollama"
    
    def _make_request(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Make HTTP request to Ollama API."""
        url = f"{self.host}{endpoint}"
        
        request = urllib.request.Request(
            url,
            data=json.dumps(data).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as e:
            logger.error(f"Ollama request failed: {e}")
            raise ConnectionError(f"Failed to connect to Ollama at {self.host}. Is Ollama running?") from e
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Ollama response: {e}")
            raise ValueError("Invalid response from Ollama") from e
    
    def _make_streaming_request(self, endpoint: str, data: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
        """Make streaming HTTP request to Ollama API."""
        url = f"{self.host}{endpoint}"
        data["stream"] = True
        
        request = urllib.request.Request(
            url,
            data=json.dumps(data).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                for line in response:
                    if line:
                        yield json.loads(line.decode("utf-8"))
        except urllib.error.URLError as e:
            logger.error(f"Ollama streaming request failed: {e}")
            raise ConnectionError(f"Failed to connect to Ollama at {self.host}") from e
    
    def complete(self, prompt: str, **kwargs) -> LLMResponse:
        """
        Generate a completion for a prompt.
        
        Args:
            prompt: The input prompt
            **kwargs: Additional Ollama options
            
        Returns:
            LLMResponse with generated text
        """
        data = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": kwargs.get("temperature", self.temperature),
                "num_predict": kwargs.get("max_tokens", self.max_tokens),
            }
        }
        
        # Add any additional options
        for key in ["top_p", "top_k", "repeat_penalty", "seed"]:
            if key in kwargs:
                data["options"][key] = kwargs[key]
        
        response = self._make_request("/api/generate", data)
        
        return LLMResponse(
            content=response.get("response", ""),
            model=self.model,
            provider=self.provider_name,
            tokens_used=response.get("eval_count", 0),
            finish_reason="stop" if response.get("done") else "length",
            raw_response=response
        )
    
    def chat(self, messages: List[Message], **kwargs) -> LLMResponse:
        """
        Generate a chat response.
        
        Args:
            messages: List of Message objects
            **kwargs: Additional Ollama options
            
        Returns:
            LLMResponse with assistant's response
        """
        data = {
            "model": self.model,
            "messages": [m.to_dict() for m in messages],
            "stream": False,
            "options": {
                "temperature": kwargs.get("temperature", self.temperature),
                "num_predict": kwargs.get("max_tokens", self.max_tokens),
            }
        }
        
        response = self._make_request("/api/chat", data)
        
        message = response.get("message", {})
        return LLMResponse(
            content=message.get("content", ""),
            model=self.model,
            provider=self.provider_name,
            tokens_used=response.get("eval_count", 0),
            finish_reason="stop" if response.get("done") else "length",
            raw_response=response
        )
    
    def stream(self, prompt: str, **kwargs) -> Iterator[str]:
        """
        Stream a completion token by token.
        
        Args:
            prompt: The input prompt
            **kwargs: Additional options
            
        Yields:
            String chunks of the response
        """
        data = {
            "model": self.model,
            "prompt": prompt,
            "options": {
                "temperature": kwargs.get("temperature", self.temperature),
                "num_predict": kwargs.get("max_tokens", self.max_tokens),
            }
        }
        
        for chunk in self._make_streaming_request("/api/generate", data):
            if "response" in chunk:
                yield chunk["response"]
    
    def chat_stream(self, messages: List[Message], **kwargs) -> Iterator[str]:
        """
        Stream a chat response token by token.
        
        Args:
            messages: Conversation messages
            **kwargs: Additional options
            
        Yields:
            String chunks of the response
        """
        data = {
            "model": self.model,
            "messages": [m.to_dict() for m in messages],
            "options": {
                "temperature": kwargs.get("temperature", self.temperature),
                "num_predict": kwargs.get("max_tokens", self.max_tokens),
            }
        }
        
        for chunk in self._make_streaming_request("/api/chat", data):
            message = chunk.get("message", {})
            if "content" in message:
                yield message["content"]
    
    def embed(self, text: str) -> List[float]:
        """
        Generate embeddings for text.
        
        Args:
            text: Text to embed
            
        Returns:
            List of floats (embedding vector)
        """
        data = {
            "model": self.model,
            "prompt": text
        }
        
        response = self._make_request("/api/embeddings", data)
        return response.get("embedding", [])
    
    def is_available(self) -> bool:
        """Check if Ollama is running and model is available."""
        try:
            # Check if server is running
            request = urllib.request.Request(f"{self.host}/api/tags")
            with urllib.request.urlopen(request, timeout=5) as response:
                data = json.loads(response.read().decode("utf-8"))
                models = [m["name"] for m in data.get("models", [])]
                
                # Check if our model is available
                model_base = self.model.split(":")[0]
                available = any(model_base in m for m in models)
                
                if not available:
                    logger.warning(f"Model {self.model} not found. Available: {models}")
                    logger.info(f"Pull the model with: ollama pull {self.model}")
                
                return available
        except Exception as e:
            logger.warning(f"Ollama not available: {e}")
            return False
    
    def list_models(self) -> List[str]:
        """List available models on the Ollama server."""
        try:
            request = urllib.request.Request(f"{self.host}/api/tags")
            with urllib.request.urlopen(request, timeout=5) as response:
                data = json.loads(response.read().decode("utf-8"))
                return [m["name"] for m in data.get("models", [])]
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []
    
    def pull_model(self, model: Optional[str] = None) -> bool:
        """
        Pull a model from the Ollama library.
        
        Args:
            model: Model to pull (default: self.model)
            
        Returns:
            True if successful
        """
        model = model or self.model
        logger.info(f"Pulling model {model}...")
        
        try:
            data = {"name": model}
            for chunk in self._make_streaming_request("/api/pull", data):
                status = chunk.get("status", "")
                if status:
                    logger.debug(status)
            return True
        except Exception as e:
            logger.error(f"Failed to pull model: {e}")
            return False
