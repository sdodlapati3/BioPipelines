"""
Token Budget Tracker
====================

Tracks token usage across:
- System prompts
- Conversation history
- Tool results
- Current context

Provides:
- Real-time token counting
- Budget warnings
- Compression triggers
- Multi-backend support (tiktoken, transformers, approximate)

Inspired by DeepCode's token management in ConciseMemoryAgent.

References:
    - DeepCode: workflows/agents/memory_agent_concise.py
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class TokenizerBackend(Enum):
    """Available tokenizer backends."""
    TIKTOKEN = "tiktoken"          # OpenAI's tokenizer (most accurate for GPT)
    TRANSFORMERS = "transformers"  # HuggingFace tokenizers
    APPROXIMATE = "approximate"    # Character-based estimation (fallback)


@dataclass
class TokenBudget:
    """
    Token budget configuration.
    
    Attributes:
        max_context: Model's total context window size
        reserved_output: Tokens reserved for model output
        system_prompt_budget: Budget for system prompt
        history_budget: Budget for conversation history
        tool_result_budget: Budget for tool results
        compression_trigger: Trigger compression at this % of available budget
    """
    max_context: int = 8192         # Model's context window
    reserved_output: int = 1024     # Reserve for response
    system_prompt_budget: int = 512
    history_budget: int = 4096
    tool_result_budget: int = 2048
    
    # Compression trigger (% of available budget)
    compression_trigger: float = 0.75
    
    @property
    def available_context(self) -> int:
        """Available context after reserving output space."""
        return self.max_context - self.reserved_output
    
    @property
    def compression_threshold(self) -> int:
        """Token count that triggers compression."""
        return int(self.available_context * self.compression_trigger)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_context": self.max_context,
            "reserved_output": self.reserved_output,
            "available_context": self.available_context,
            "compression_threshold": self.compression_threshold,
        }


@dataclass
class TokenUsage:
    """
    Current token usage breakdown.
    
    Tracks tokens by category for fine-grained management.
    """
    system_prompt: int = 0
    history: int = 0
    tool_results: int = 0
    current_turn: int = 0
    
    @property
    def total(self) -> int:
        """Total tokens across all categories."""
        return self.system_prompt + self.history + self.tool_results + self.current_turn
    
    def to_dict(self) -> Dict[str, int]:
        return {
            "system_prompt": self.system_prompt,
            "history": self.history,
            "tool_results": self.tool_results,
            "current_turn": self.current_turn,
            "total": self.total,
        }
    
    def reset_turn(self):
        """Move current turn to history and reset."""
        self.history += self.current_turn
        self.current_turn = 0
    
    def reset_all(self):
        """Reset all counters except system prompt."""
        self.history = 0
        self.tool_results = 0
        self.current_turn = 0


class TokenTracker:
    """
    Tracks token usage with multiple backend support.
    
    Automatically selects best available tokenizer:
    1. tiktoken (most accurate for OpenAI models)
    2. transformers (for open models)
    3. approximate (fallback, ~4 chars/token)
    
    Examples:
        >>> tracker = TokenTracker(budget=TokenBudget(max_context=32768))
        >>> tracker.add_system_prompt(SYSTEM_PROMPT)
        >>> tracker.add_message("user", query)
        >>> 
        >>> if tracker.should_compress():
        ...     # Trigger memory compression
        ...     pass
        >>> 
        >>> print(f"Remaining budget: {tracker.get_remaining_budget()} tokens")
    """
    
    # Per-message overhead (role, formatting, etc.)
    MESSAGE_OVERHEAD = 4
    
    # Reply priming tokens
    REPLY_PRIMING = 3
    
    def __init__(
        self,
        budget: Optional[TokenBudget] = None,
        model_name: str = "gpt-4",
        backend: Optional[TokenizerBackend] = None,
    ):
        """
        Initialize the token tracker.
        
        Args:
            budget: Token budget configuration
            model_name: Model name (used for tokenizer selection)
            backend: Force specific tokenizer backend
        """
        self.budget = budget or TokenBudget()
        self.model_name = model_name
        self.usage = TokenUsage()
        
        # Auto-select tokenizer backend
        self._backend = backend or self._select_backend()
        self._tokenizer = self._load_tokenizer()
        
        logger.debug(f"TokenTracker using {self._backend.value} backend")
    
    def _select_backend(self) -> TokenizerBackend:
        """Select best available tokenizer backend."""
        try:
            import tiktoken
            return TokenizerBackend.TIKTOKEN
        except ImportError:
            pass
        
        try:
            from transformers import AutoTokenizer
            return TokenizerBackend.TRANSFORMERS
        except ImportError:
            pass
        
        logger.warning(
            "No tokenizer library found, using approximate counting. "
            "Install tiktoken or transformers for accurate counts."
        )
        return TokenizerBackend.APPROXIMATE
    
    def _load_tokenizer(self) -> Any:
        """Load the selected tokenizer."""
        if self._backend == TokenizerBackend.TIKTOKEN:
            import tiktoken
            try:
                # Try model-specific encoding
                return tiktoken.encoding_for_model(self.model_name)
            except KeyError:
                # Fall back to cl100k_base (GPT-4 encoding)
                logger.debug(
                    f"No specific encoding for {self.model_name}, "
                    f"using cl100k_base"
                )
                return tiktoken.get_encoding("cl100k_base")
        
        elif self._backend == TokenizerBackend.TRANSFORMERS:
            from transformers import AutoTokenizer
            
            # Map common model names to HuggingFace paths
            model_map = {
                "llama": "meta-llama/Llama-2-7b-hf",
                "mistral": "mistralai/Mistral-7B-v0.1",
                "qwen": "Qwen/Qwen2-7B",
            }
            
            # Try to find matching model
            model_lower = self.model_name.lower()
            hf_model = self.model_name
            for key, path in model_map.items():
                if key in model_lower:
                    hf_model = path
                    break
            
            try:
                return AutoTokenizer.from_pretrained(hf_model)
            except Exception as e:
                logger.warning(f"Could not load tokenizer for {hf_model}: {e}")
                return None
        
        return None  # Approximate doesn't need tokenizer
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Token count
        """
        if not text:
            return 0
        
        if self._backend == TokenizerBackend.TIKTOKEN:
            return len(self._tokenizer.encode(text))
        
        elif self._backend == TokenizerBackend.TRANSFORMERS and self._tokenizer:
            return len(self._tokenizer.encode(text))
        
        else:
            # Approximate: ~4 characters per token (conservative estimate)
            # This tends to overcount slightly, which is safer
            return len(text) // 4 + 1
    
    def count_messages(self, messages: List[Dict[str, str]]) -> int:
        """
        Count tokens in a list of chat messages.
        
        Accounts for per-message overhead (role, formatting).
        
        Args:
            messages: List of message dicts with "role" and "content"
            
        Returns:
            Total token count including overhead
        """
        total = 0
        for msg in messages:
            # Per-message overhead
            total += self.MESSAGE_OVERHEAD
            total += self.count_tokens(msg.get("content", ""))
        
        # Reply priming
        total += self.REPLY_PRIMING
        
        return total
    
    def add_system_prompt(self, prompt: str) -> int:
        """
        Add system prompt and return token count.
        
        Args:
            prompt: System prompt text
            
        Returns:
            Token count of the prompt
        """
        count = self.count_tokens(prompt)
        self.usage.system_prompt = count
        
        if count > self.budget.system_prompt_budget:
            logger.warning(
                f"System prompt ({count} tokens) exceeds budget "
                f"({self.budget.system_prompt_budget})"
            )
        
        return count
    
    def add_message(self, role: str, content: str) -> int:
        """
        Add a message to history tracking.
        
        Args:
            role: Message role ("user", "assistant", "system")
            content: Message content
            
        Returns:
            Token count of the message
        """
        count = self.count_tokens(content) + self.MESSAGE_OVERHEAD
        self.usage.history += count
        return count
    
    def add_tool_result(self, result: str) -> int:
        """
        Add tool result to tracking.
        
        Args:
            result: Tool result text
            
        Returns:
            Token count of the result
        """
        count = self.count_tokens(result)
        self.usage.tool_results += count
        
        if self.usage.tool_results > self.budget.tool_result_budget:
            logger.warning(
                f"Tool results ({self.usage.tool_results} tokens) exceed budget "
                f"({self.budget.tool_result_budget})"
            )
        
        return count
    
    def set_current_turn(self, content: str) -> int:
        """
        Set current turn content.
        
        Args:
            content: Current turn content
            
        Returns:
            Token count
        """
        count = self.count_tokens(content)
        self.usage.current_turn = count
        return count
    
    def reset_current_turn(self):
        """Reset current turn (moves tokens to history)."""
        self.usage.reset_turn()
    
    def reset_tool_results(self):
        """Reset tool results counter (after compression)."""
        self.usage.tool_results = 0
    
    def reset_history(self):
        """Reset history counter (after compression)."""
        self.usage.history = 0
    
    def should_compress(self) -> bool:
        """
        Check if compression should be triggered.
        
        Returns:
            True if current usage exceeds compression threshold
        """
        return self.usage.total >= self.budget.compression_threshold
    
    def get_remaining_budget(self) -> int:
        """
        Get remaining token budget.
        
        Returns:
            Number of tokens available before hitting limit
        """
        return max(0, self.budget.available_context - self.usage.total)
    
    def get_usage_percentage(self) -> float:
        """
        Get current usage as percentage of available budget.
        
        Returns:
            Usage percentage (0.0 to 100.0+)
        """
        if self.budget.available_context == 0:
            return 100.0
        return (self.usage.total / self.budget.available_context) * 100
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get detailed status for debugging.
        
        Returns:
            Dict with usage, budget, and status information
        """
        return {
            "usage": self.usage.to_dict(),
            "budget": self.budget.to_dict(),
            "remaining": self.get_remaining_budget(),
            "usage_percentage": round(self.get_usage_percentage(), 1),
            "should_compress": self.should_compress(),
            "backend": self._backend.value,
            "model": self.model_name,
        }
    
    def estimate_fit(self, content: str) -> Tuple[bool, int]:
        """
        Check if content fits in remaining budget.
        
        Args:
            content: Content to check
            
        Returns:
            Tuple of (fits: bool, tokens_needed: int)
        """
        tokens = self.count_tokens(content)
        remaining = self.get_remaining_budget()
        return tokens <= remaining, tokens
    
    def estimate_messages_fit(
        self,
        messages: List[Dict[str, str]],
    ) -> Tuple[bool, int]:
        """
        Check if messages fit in remaining budget.
        
        Args:
            messages: List of message dicts
            
        Returns:
            Tuple of (fits: bool, tokens_needed: int)
        """
        tokens = self.count_messages(messages)
        remaining = self.get_remaining_budget()
        return tokens <= remaining, tokens


# =============================================================================
# Factory Functions
# =============================================================================

# Context window sizes for common models
MODEL_CONTEXT_SIZES = {
    # OpenAI
    "gpt-4": 8192,
    "gpt-4-32k": 32768,
    "gpt-4-turbo": 128000,
    "gpt-4o": 128000,
    "gpt-3.5-turbo": 16385,
    
    # Anthropic
    "claude-3": 200000,
    "claude-2": 100000,
    
    # Open source
    "llama-3.3-70b": 128000,
    "llama-3.1": 128000,
    "llama-3": 8192,
    "llama-2": 4096,
    "qwen2.5": 32768,
    "qwen2": 32768,
    "mistral": 32768,
    "mixtral": 32768,
    "deepseek": 32768,
}


def create_tracker_for_model(
    model_name: str,
    compression_trigger: float = 0.75,
) -> TokenTracker:
    """
    Create token tracker with appropriate budget for model.
    
    Automatically determines context window size based on model name.
    
    Args:
        model_name: Model name or identifier
        compression_trigger: When to trigger compression (0.0-1.0)
        
    Returns:
        Configured TokenTracker
        
    Examples:
        >>> tracker = create_tracker_for_model("gpt-4o")
        >>> print(tracker.budget.max_context)
        128000
    """
    # Find matching model
    max_context = 8192  # default
    model_lower = model_name.lower()
    
    for key, size in MODEL_CONTEXT_SIZES.items():
        if key in model_lower:
            max_context = size
            break
    
    budget = TokenBudget(
        max_context=max_context,
        compression_trigger=compression_trigger,
    )
    
    return TokenTracker(budget=budget, model_name=model_name)


def create_budget_for_task(
    task_type: str,
    model_context: int = 8192,
) -> TokenBudget:
    """
    Create token budget optimized for specific task types.
    
    Args:
        task_type: One of "chat", "code_generation", "analysis", "diagnosis"
        model_context: Model's context window size
        
    Returns:
        Configured TokenBudget
    """
    configs = {
        "chat": {
            "reserved_output": 1024,
            "system_prompt_budget": 512,
            "history_budget": model_context // 2,
            "tool_result_budget": 1024,
            "compression_trigger": 0.8,
        },
        "code_generation": {
            "reserved_output": 4096,  # Need more for code output
            "system_prompt_budget": 1024,
            "history_budget": model_context // 4,
            "tool_result_budget": 2048,
            "compression_trigger": 0.7,
        },
        "analysis": {
            "reserved_output": 2048,
            "system_prompt_budget": 512,
            "history_budget": model_context // 3,
            "tool_result_budget": model_context // 3,  # Large tool results
            "compression_trigger": 0.75,
        },
        "diagnosis": {
            "reserved_output": 2048,
            "system_prompt_budget": 768,
            "history_budget": 1024,
            "tool_result_budget": model_context // 2,  # Error logs can be large
            "compression_trigger": 0.7,
        },
    }
    
    config = configs.get(task_type, configs["chat"])
    
    return TokenBudget(
        max_context=model_context,
        **config,
    )
