"""
LLM Adapter Layer
=================

Provides a unified interface to multiple LLM providers:
- OpenAI (GPT-4, GPT-3.5)
- Anthropic (Claude 3)
- Ollama (local models: Llama3, Mistral, CodeLlama)
- HuggingFace (any HF model)
- Custom endpoints (vLLM, etc.)

Usage:
    from workflow_composer.llm import get_llm, OllamaAdapter
    
    # Using factory
    llm = get_llm("ollama", model="llama3:8b")
    
    # Direct instantiation
    llm = OllamaAdapter(model="codellama:13b")
    
    # Generate completion
    response = llm.complete("Explain RNA-seq analysis")
    
    # List available providers
    providers = list_providers()
    
    # Check which are available
    available = check_providers()
"""

from .base import LLMAdapter, Message, LLMResponse
from .ollama_adapter import OllamaAdapter
from .openai_adapter import OpenAIAdapter
from .anthropic_adapter import AnthropicAdapter
from .huggingface_adapter import HuggingFaceAdapter
from .factory import get_llm, LLMFactory, list_providers, check_providers, register_provider

__all__ = [
    "LLMAdapter",
    "Message", 
    "LLMResponse",
    "OllamaAdapter",
    "OpenAIAdapter",
    "AnthropicAdapter",
    "HuggingFaceAdapter",
    "get_llm",
    "LLMFactory",
    "list_providers",
    "check_providers",
    "register_provider"
]
