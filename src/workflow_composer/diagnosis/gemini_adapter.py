"""
Google Gemini adapter for error diagnosis.

Provides access to Gemini models via the free tier API.
Optimized for quick error analysis with minimal token usage.
"""

import os
import logging
from typing import Optional, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class GeminiResponse:
    """Response from Gemini API."""
    content: str
    model: str
    usage: Optional[dict] = None


class GeminiAdapter:
    """
    Adapter for Google Gemini API.
    
    Supports free tier access with gemini-1.5-flash and other models.
    Optimized for error diagnosis with minimal token usage.
    
    Example:
        adapter = GeminiAdapter()
        response = adapter.complete("Analyze this error...")
    """
    
    # Recommended models
    MODELS = {
        "gemini-flash": "gemini-1.5-flash",
        "gemini-pro": "gemini-1.5-pro",
        "gemini-flash-8b": "gemini-1.5-flash-8b",
    }
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-1.5-flash",
    ):
        """
        Initialize Gemini adapter.
        
        Args:
            api_key: Google API key (or from GOOGLE_API_KEY env)
            model: Model name to use
        """
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self.model = self.MODELS.get(model, model)
        self._client = None
        
        if not self.api_key:
            logger.warning("GOOGLE_API_KEY not set - Gemini will be unavailable")
    
    @property
    def client(self):
        """Lazy initialization of Gemini client."""
        if self._client is None:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self._client = genai.GenerativeModel(self.model)
            except ImportError:
                logger.error(
                    "google-generativeai not installed. "
                    "Install with: pip install google-generativeai"
                )
                raise
        return self._client
    
    def is_available(self) -> bool:
        """Check if Gemini API is available."""
        if not self.api_key:
            return False
        
        try:
            import google.generativeai
            return True
        except ImportError:
            return False
    
    def complete(self, prompt: str) -> GeminiResponse:
        """
        Generate completion for a prompt.
        
        Args:
            prompt: The prompt text
            
        Returns:
            GeminiResponse with generated content
        """
        try:
            response = self.client.generate_content(prompt)
            
            return GeminiResponse(
                content=response.text,
                model=self.model,
                usage={
                    "prompt_tokens": response.usage_metadata.prompt_token_count,
                    "completion_tokens": response.usage_metadata.candidates_token_count,
                } if hasattr(response, 'usage_metadata') else None,
            )
        except Exception as e:
            logger.error(f"Gemini completion failed: {e}")
            raise
    
    def chat(self, messages: List) -> GeminiResponse:
        """
        Chat completion (converts to single prompt for Gemini).
        
        Args:
            messages: List of Message objects
            
        Returns:
            GeminiResponse
        """
        # Convert messages to single prompt
        prompt_parts = []
        
        for msg in messages:
            role = getattr(msg, 'role', 'user')
            content = getattr(msg, 'content', str(msg))
            
            if role == 'system':
                prompt_parts.append(f"Instructions: {content}")
            elif role == 'user':
                prompt_parts.append(f"User: {content}")
            elif role == 'assistant':
                prompt_parts.append(f"Assistant: {content}")
        
        prompt = "\n\n".join(prompt_parts)
        prompt += "\n\nAssistant:"
        
        return self.complete(prompt)
    
    def __repr__(self) -> str:
        return f"GeminiAdapter(model={self.model})"


def get_gemini(model: str = "gemini-1.5-flash") -> Optional[GeminiAdapter]:
    """
    Get a Gemini adapter if available.
    
    Args:
        model: Model name
        
    Returns:
        GeminiAdapter or None if unavailable
    """
    adapter = GeminiAdapter(model=model)
    
    if adapter.is_available():
        return adapter
    
    return None


def check_gemini_available() -> bool:
    """Check if Gemini API is available."""
    return bool(os.getenv("GOOGLE_API_KEY"))
