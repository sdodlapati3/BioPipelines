# DeepCode-Inspired Enhancements Implementation Plan

**Date:** December 2, 2025  
**Author:** BioPipelines Enhancement Initiative  
**Status:** Planning → Implementation  
**Reference:** [HKUDS/DeepCode](https://github.com/HKUDS/DeepCode)

---

## Executive Summary

This document outlines a comprehensive plan to enhance BioPipelines' agentic system by borrowing battle-tested patterns from DeepCode. The focus is on **coding agent improvements**, **error diagnosis capabilities**, and **error resolving/recovery patterns** - NOT paper parsing functionality.

### Key Features to Implement

| # | Feature | Priority | Complexity | Impact | Files Affected |
|---|---------|----------|------------|--------|----------------|
| 1 | JSON Repair Utility | P0 | Low | High | New: `agents/utils/json_repair.py` |
| 2 | Adaptive Retry with Parameter Reduction | P0 | Low | High | `coding_agent.py`, `react_agent.py` |
| 3 | Token Budget Tracking | P1 | Medium | High | New: `agents/memory/token_tracker.py` |
| 4 | Enhanced Error Guidance Generation | P1 | Medium | Medium | `coding_agent.py` |
| 5 | Concise Memory Agent Pattern | P2 | Medium | High | New: `agents/memory/concise_memory.py` |
| 6 | Multi-File Clean Slate Pattern | P2 | Medium | Medium | Integrate with workflow generator |

### Additional BioPipelines-Specific Enhancements

| # | Feature | Priority | Rationale |
|---|---------|----------|-----------|
| 7 | LLM Response Validator | P1 | Validate structure before parsing |
| 8 | Graceful Degradation Chain | P1 | Fallback hierarchy for failures |
| 9 | Bioinformatics Error Patterns | P1 | Expand error pattern database |

---

## Phase 1: Core Utilities (P0 - Immediate)

### 1.1 JSON Repair Utility

**File:** `src/workflow_composer/agents/utils/json_repair.py`

**Purpose:** Fix common JSON truncation/malformation issues from LLM outputs.

**DeepCode Pattern:**
```python
# From DeepCode: workflows/code_implementation_workflow.py
def _repair_truncated_json(self, content: str) -> Optional[str]:
    if content.count('{') > content.count('}'):
        content += '}' * (content.count('{') - content.count('}'))
    if content.count('[') > content.count(']'):
        content += ']' * (content.count('[') - content.count(']'))
```

**BioPipelines Enhanced Implementation:**
```python
"""
JSON Repair Utility
===================

Repairs common JSON issues from LLM outputs:
- Truncated JSON (unclosed braces/brackets)
- Trailing commas (not valid JSON)
- Unclosed strings
- Invalid escape sequences
- Control characters
"""

import re
import json
import logging
from typing import Optional, Tuple, Any

logger = logging.getLogger(__name__)


class JSONRepairError(Exception):
    """Raised when JSON cannot be repaired."""
    pass


def repair_json(content: str, strict: bool = False) -> Tuple[str, bool]:
    """
    Attempt to repair malformed JSON.
    
    Args:
        content: Potentially malformed JSON string
        strict: If True, raise on failure; if False, return original
        
    Returns:
        Tuple of (repaired_json, was_modified)
        
    Raises:
        JSONRepairError: If strict=True and repair fails
    """
    if not content or not content.strip():
        if strict:
            raise JSONRepairError("Empty content")
        return content, False
    
    original = content
    modified = False
    
    # Step 1: Strip markdown code blocks
    content, stripped = _strip_code_blocks(content)
    modified = modified or stripped
    
    # Step 2: Fix trailing commas before ] or }
    content, fixed = _fix_trailing_commas(content)
    modified = modified or fixed
    
    # Step 3: Balance braces and brackets
    content, balanced = _balance_brackets(content)
    modified = modified or balanced
    
    # Step 4: Fix unclosed strings
    content, fixed_strings = _fix_unclosed_strings(content)
    modified = modified or fixed_strings
    
    # Step 5: Remove control characters
    content, cleaned = _remove_control_chars(content)
    modified = modified or cleaned
    
    # Validate result
    try:
        json.loads(content)
        if modified:
            logger.info("JSON successfully repaired")
        return content, modified
    except json.JSONDecodeError as e:
        if strict:
            raise JSONRepairError(f"Could not repair JSON: {e}")
        logger.warning(f"JSON repair failed: {e}, returning original")
        return original, False


def _strip_code_blocks(content: str) -> Tuple[str, bool]:
    """Remove markdown code block markers."""
    # Match ```json ... ``` or ``` ... ```
    pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
    match = re.search(pattern, content)
    if match:
        return match.group(1).strip(), True
    return content, False


def _fix_trailing_commas(content: str) -> Tuple[str, bool]:
    """Remove trailing commas before } or ]."""
    # Match: , followed by whitespace then } or ]
    pattern = r',(\s*[}\]])'
    fixed = re.sub(pattern, r'\1', content)
    return fixed, fixed != content


def _balance_brackets(content: str) -> Tuple[str, bool]:
    """Balance unclosed braces and brackets."""
    modified = False
    
    # Count brackets
    open_braces = content.count('{') - content.count('}')
    open_brackets = content.count('[') - content.count(']')
    
    # Add missing closing brackets
    if open_braces > 0:
        content = content.rstrip() + '}' * open_braces
        modified = True
    if open_brackets > 0:
        content = content.rstrip() + ']' * open_brackets
        modified = True
    
    return content, modified


def _fix_unclosed_strings(content: str) -> Tuple[str, bool]:
    """Attempt to close unclosed string literals."""
    # This is tricky - only handle obvious cases
    # Count quotes outside of escaped sequences
    quote_count = 0
    in_string = False
    i = 0
    while i < len(content):
        if content[i] == '"' and (i == 0 or content[i-1] != '\\'):
            in_string = not in_string
            quote_count += 1
        i += 1
    
    if quote_count % 2 != 0:
        # Odd number of quotes - try to close
        content = content.rstrip()
        if not content.endswith('"'):
            content += '"'
            return content, True
    
    return content, False


def _remove_control_chars(content: str) -> Tuple[str, bool]:
    """Remove control characters that break JSON."""
    # Remove characters 0x00-0x1F except \t, \n, \r
    cleaned = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', content)
    return cleaned, cleaned != content


def safe_json_loads(content: str, default: Any = None) -> Any:
    """
    Safely parse JSON with automatic repair.
    
    Args:
        content: JSON string to parse
        default: Default value if parsing fails
        
    Returns:
        Parsed JSON or default value
    """
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        try:
            repaired, _ = repair_json(content, strict=False)
            return json.loads(repaired)
        except (json.JSONDecodeError, JSONRepairError):
            logger.warning("JSON parsing failed even after repair")
            return default


def extract_json_from_text(text: str) -> Optional[str]:
    """
    Extract JSON object or array from mixed text.
    
    Useful when LLM outputs JSON embedded in explanation text.
    """
    # Try to find JSON object
    brace_match = re.search(r'\{[\s\S]*\}', text)
    if brace_match:
        candidate = brace_match.group()
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            repaired, _ = repair_json(candidate)
            try:
                json.loads(repaired)
                return repaired
            except:
                pass
    
    # Try to find JSON array
    bracket_match = re.search(r'\[[\s\S]*\]', text)
    if bracket_match:
        candidate = bracket_match.group()
        try:
            json.loads(candidate)
            return candidate
        except:
            pass
    
    return None
```

**Integration Points:**
- `react_agent.py`: `_parse_response()` - Use `safe_json_loads()` for action_input
- `coding_agent.py`: `validate_workflow()` - Use `extract_json_from_text()` for LLM output
- `mcp/server.py`: All tool handlers that parse JSON

**Tests Required:** (in `tests/test_json_repair.py`)
- Test empty content handling
- Test trailing comma removal
- Test bracket balancing
- Test unclosed string fix
- Test markdown code block stripping
- Test extraction from mixed text
- Test real-world LLM truncation scenarios

---

### 1.2 Adaptive Retry with Parameter Reduction

**Files to Modify:**
- `src/workflow_composer/agents/coding_agent.py`
- `src/workflow_composer/agents/react_agent.py`

**New File:** `src/workflow_composer/agents/utils/retry_strategy.py`

**DeepCode Pattern:**
```python
# From DeepCode: workflows/agent_orchestration_engine.py
def _adjust_params_for_retry(self, params, attempt, context_size=32768):
    """On retry: REDUCE tokens and lower temperature for stability"""
    if attempt > 0:
        if context_size >= 32768:
            params["max_tokens"] = 15000  # Was 20000
        else:
            params["max_tokens"] = 6000   # Was 8000
        params["temperature"] = max(0.1, params.get("temperature", 0.7) - 0.3)
    return params
```

**BioPipelines Enhanced Implementation:**
```python
"""
Retry Strategy Module
=====================

Implements intelligent retry with parameter adjustment.

Key insight from DeepCode: On retry, REDUCE output size and temperature.
This is counterintuitive but effective because:
- Smaller output = less chance of truncation/malformation
- Lower temperature = more deterministic/stable output
- Reduced complexity = more likely to succeed
"""

import time
import logging
from typing import TypeVar, Callable, Any, Dict, Optional
from dataclasses import dataclass, field
from functools import wraps

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 30.0
    exponential_base: float = 2.0
    
    # Parameter reduction on retry
    token_reduction_factor: float = 0.75  # Reduce by 25% each retry
    temperature_reduction: float = 0.2     # Reduce by 0.2 each retry
    min_temperature: float = 0.1
    min_tokens: int = 512


@dataclass
class RetryState:
    """Tracks retry state across attempts."""
    attempt: int = 0
    last_error: Optional[str] = None
    adjusted_params: Dict[str, Any] = field(default_factory=dict)
    total_time: float = 0.0


def adjust_llm_params_for_retry(
    params: Dict[str, Any],
    attempt: int,
    config: Optional[RetryConfig] = None,
    context_size: int = 32768,
) -> Dict[str, Any]:
    """
    Adjust LLM parameters for retry attempt.
    
    Strategy: REDUCE complexity on retry for higher success rate.
    
    Args:
        params: Original LLM parameters
        attempt: Current attempt number (0 = first try)
        config: Retry configuration
        context_size: Model's context window size
        
    Returns:
        Adjusted parameters dict
    """
    if config is None:
        config = RetryConfig()
    
    adjusted = params.copy()
    
    if attempt == 0:
        return adjusted
    
    # Reduce max_tokens (multiplicative reduction)
    if "max_tokens" in adjusted:
        original_tokens = adjusted["max_tokens"]
        factor = config.token_reduction_factor ** attempt
        adjusted["max_tokens"] = max(
            config.min_tokens,
            int(original_tokens * factor)
        )
        logger.debug(f"Retry {attempt}: Reduced max_tokens {original_tokens} -> {adjusted['max_tokens']}")
    
    # Reduce temperature (additive reduction)
    if "temperature" in adjusted:
        original_temp = adjusted["temperature"]
        adjusted["temperature"] = max(
            config.min_temperature,
            original_temp - (config.temperature_reduction * attempt)
        )
        logger.debug(f"Retry {attempt}: Reduced temperature {original_temp} -> {adjusted['temperature']}")
    
    # Add stop sequences if not present (helps prevent runaway generation)
    if "stop" not in adjusted and attempt >= 2:
        adjusted["stop"] = ["\n\n\n", "```\n\n"]
    
    return adjusted


def get_retry_delay(attempt: int, config: Optional[RetryConfig] = None) -> float:
    """Calculate delay before next retry with exponential backoff."""
    if config is None:
        config = RetryConfig()
    
    delay = config.base_delay * (config.exponential_base ** attempt)
    return min(delay, config.max_delay)


def with_retry(
    config: Optional[RetryConfig] = None,
    retry_on: tuple = (Exception,),
    on_retry: Optional[Callable[[int, Exception], None]] = None,
):
    """
    Decorator for retrying functions with exponential backoff.
    
    Usage:
        @with_retry(config=RetryConfig(max_attempts=3))
        def call_llm(prompt: str) -> str:
            ...
    """
    if config is None:
        config = RetryConfig()
    
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_error = None
            
            for attempt in range(config.max_attempts):
                try:
                    return func(*args, **kwargs)
                except retry_on as e:
                    last_error = e
                    if attempt < config.max_attempts - 1:
                        delay = get_retry_delay(attempt, config)
                        logger.warning(
                            f"Attempt {attempt + 1} failed: {e}. "
                            f"Retrying in {delay:.1f}s..."
                        )
                        if on_retry:
                            on_retry(attempt, e)
                        time.sleep(delay)
                    else:
                        logger.error(f"All {config.max_attempts} attempts failed")
            
            raise last_error
        
        return wrapper
    return decorator


class AdaptiveLLMCaller:
    """
    Wrapper for LLM calls with adaptive retry.
    
    Example:
        caller = AdaptiveLLMCaller(client, model="gpt-4")
        response = caller.call(messages=messages, max_tokens=1024)
    """
    
    def __init__(
        self,
        client: Any,
        model: str,
        config: Optional[RetryConfig] = None,
    ):
        self.client = client
        self.model = model
        self.config = config or RetryConfig()
        self._state = RetryState()
    
    def call(
        self,
        messages: list,
        **kwargs,
    ) -> Any:
        """
        Make LLM call with adaptive retry.
        
        Args:
            messages: Chat messages
            **kwargs: Additional parameters (max_tokens, temperature, etc.)
            
        Returns:
            LLM response
        """
        self._state = RetryState()
        last_error = None
        
        for attempt in range(self.config.max_attempts):
            self._state.attempt = attempt
            
            # Adjust parameters for retry
            adjusted_params = adjust_llm_params_for_retry(
                kwargs, attempt, self.config
            )
            self._state.adjusted_params = adjusted_params
            
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    **adjusted_params,
                )
                return response
                
            except Exception as e:
                last_error = e
                self._state.last_error = str(e)
                
                if attempt < self.config.max_attempts - 1:
                    delay = get_retry_delay(attempt, self.config)
                    logger.warning(
                        f"LLM call attempt {attempt + 1} failed: {e}. "
                        f"Adjusting params and retrying in {delay:.1f}s"
                    )
                    time.sleep(delay)
        
        raise last_error
    
    @property
    def last_state(self) -> RetryState:
        """Get state from last call for debugging."""
        return self._state
```

**Integration into CodingAgent:**
```python
# In coding_agent.py diagnose_error() method:
# Before:
response = client.chat.completions.create(
    model=model,
    messages=[...],
    temperature=0.1,
    max_tokens=1024,
)

# After:
from .utils.retry_strategy import AdaptiveLLMCaller, RetryConfig

caller = AdaptiveLLMCaller(
    client, model,
    config=RetryConfig(max_attempts=3, token_reduction_factor=0.75)
)
response = caller.call(
    messages=[...],
    temperature=0.1,
    max_tokens=1024,
)
```

---

## Phase 2: Memory & Token Management (P1)

### 2.1 Token Budget Tracking

**File:** `src/workflow_composer/agents/memory/token_tracker.py`

**Purpose:** Track token usage to prevent context overflow and trigger proactive compression.

**DeepCode Pattern:**
```python
# From DeepCode: workflows/agents/memory_agent_concise.py
def calculate_messages_token_count(self, messages: list) -> int:
    """Track exactly how many tokens we're using"""
    # Triggers compression when exceeding summary_trigger_tokens
```

**BioPipelines Implementation:**
```python
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
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class TokenizerBackend(Enum):
    """Available tokenizer backends."""
    TIKTOKEN = "tiktoken"      # OpenAI's tokenizer
    TRANSFORMERS = "transformers"  # HuggingFace tokenizers
    APPROXIMATE = "approximate"    # Character-based estimation


@dataclass
class TokenBudget:
    """Token budget configuration."""
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


@dataclass
class TokenUsage:
    """Current token usage breakdown."""
    system_prompt: int = 0
    history: int = 0
    tool_results: int = 0
    current_turn: int = 0
    
    @property
    def total(self) -> int:
        return self.system_prompt + self.history + self.tool_results + self.current_turn
    
    def to_dict(self) -> Dict[str, int]:
        return {
            "system_prompt": self.system_prompt,
            "history": self.history,
            "tool_results": self.tool_results,
            "current_turn": self.current_turn,
            "total": self.total,
        }


class TokenTracker:
    """
    Tracks token usage with multiple backend support.
    
    Automatically selects best available tokenizer:
    1. tiktoken (most accurate for OpenAI models)
    2. transformers (for open models)
    3. approximate (fallback)
    
    Example:
        tracker = TokenTracker(budget=TokenBudget(max_context=32768))
        tracker.add_system_prompt(SYSTEM_PROMPT)
        tracker.add_message("user", query)
        
        if tracker.should_compress():
            # Trigger memory compression
            pass
    """
    
    def __init__(
        self,
        budget: Optional[TokenBudget] = None,
        model_name: str = "gpt-4",
        backend: Optional[TokenizerBackend] = None,
    ):
        self.budget = budget or TokenBudget()
        self.model_name = model_name
        self.usage = TokenUsage()
        
        # Auto-select tokenizer backend
        self._backend = backend or self._select_backend()
        self._tokenizer = self._load_tokenizer()
        
        logger.info(f"TokenTracker using {self._backend.value} backend")
    
    def _select_backend(self) -> TokenizerBackend:
        """Select best available tokenizer."""
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
                return tiktoken.get_encoding("cl100k_base")
        
        elif self._backend == TokenizerBackend.TRANSFORMERS:
            from transformers import AutoTokenizer
            # Map common model names to HuggingFace paths
            model_map = {
                "llama": "meta-llama/Llama-2-7b-hf",
                "mistral": "mistralai/Mistral-7B-v0.1",
            }
            hf_model = model_map.get(self.model_name.lower().split("/")[0], self.model_name)
            try:
                return AutoTokenizer.from_pretrained(hf_model)
            except:
                return None
        
        return None  # Approximate doesn't need tokenizer
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        if not text:
            return 0
        
        if self._backend == TokenizerBackend.TIKTOKEN:
            return len(self._tokenizer.encode(text))
        
        elif self._backend == TokenizerBackend.TRANSFORMERS and self._tokenizer:
            return len(self._tokenizer.encode(text))
        
        else:
            # Approximate: ~4 characters per token (conservative)
            return len(text) // 4
    
    def count_messages(self, messages: List[Dict[str, str]]) -> int:
        """
        Count tokens in a list of messages.
        
        Accounts for message overhead (role, formatting).
        """
        total = 0
        for msg in messages:
            # Per-message overhead (~4 tokens for role/formatting)
            total += 4
            total += self.count_tokens(msg.get("content", ""))
        # Reply priming
        total += 2
        return total
    
    def add_system_prompt(self, prompt: str) -> int:
        """Add system prompt and return token count."""
        count = self.count_tokens(prompt)
        self.usage.system_prompt = count
        return count
    
    def add_message(self, role: str, content: str) -> int:
        """Add a message to history tracking."""
        count = self.count_tokens(content) + 4  # overhead
        self.usage.history += count
        return count
    
    def add_tool_result(self, result: str) -> int:
        """Add tool result to tracking."""
        count = self.count_tokens(result)
        self.usage.tool_results += count
        return count
    
    def set_current_turn(self, content: str) -> int:
        """Set current turn content."""
        count = self.count_tokens(content)
        self.usage.current_turn = count
        return count
    
    def reset_current_turn(self):
        """Reset current turn (after it becomes history)."""
        self.usage.history += self.usage.current_turn
        self.usage.current_turn = 0
    
    def reset_tool_results(self):
        """Reset tool results (after compression)."""
        self.usage.tool_results = 0
    
    def should_compress(self) -> bool:
        """Check if compression should be triggered."""
        return self.usage.total >= self.budget.compression_threshold
    
    def get_remaining_budget(self) -> int:
        """Get remaining token budget."""
        return max(0, self.budget.available_context - self.usage.total)
    
    def get_status(self) -> Dict[str, Any]:
        """Get detailed status for debugging."""
        return {
            "usage": self.usage.to_dict(),
            "budget": {
                "max_context": self.budget.max_context,
                "available": self.budget.available_context,
                "remaining": self.get_remaining_budget(),
                "compression_threshold": self.budget.compression_threshold,
            },
            "should_compress": self.should_compress(),
            "backend": self._backend.value,
        }
    
    def estimate_fit(self, content: str) -> Tuple[bool, int]:
        """
        Check if content fits in remaining budget.
        
        Returns:
            Tuple of (fits: bool, tokens_needed: int)
        """
        tokens = self.count_tokens(content)
        remaining = self.get_remaining_budget()
        return tokens <= remaining, tokens


# Convenience functions
def create_tracker_for_model(model_name: str) -> TokenTracker:
    """Create token tracker with appropriate budget for model."""
    # Context window sizes for common models
    context_sizes = {
        "gpt-4": 8192,
        "gpt-4-32k": 32768,
        "gpt-4-turbo": 128000,
        "gpt-4o": 128000,
        "claude-3": 200000,
        "llama-3.3-70b": 128000,
        "qwen2.5": 32768,
        "mistral": 32768,
    }
    
    # Find matching model
    max_context = 8192  # default
    model_lower = model_name.lower()
    for key, size in context_sizes.items():
        if key in model_lower:
            max_context = size
            break
    
    budget = TokenBudget(max_context=max_context)
    return TokenTracker(budget=budget, model_name=model_name)
```

---

### 2.2 Enhanced Error Guidance Generation

**File to Modify:** `src/workflow_composer/agents/coding_agent.py`

**Purpose:** Convert passive diagnosis results into actionable recovery guidance.

**DeepCode Pattern:**
```python
# From DeepCode: workflows/code_implementation_workflow.py
def _generate_error_guidance(self, error_info: dict) -> str:
    """Generate context-aware recovery instructions"""
    return f"""
## Error Encountered
**Type:** {error_info.get('type', 'Unknown')}
...
## What NOT to do
- Don't {anti_pattern_1}
"""
```

**BioPipelines Implementation:**

Add new dataclass and method to `coding_agent.py`:

```python
@dataclass
class ErrorGuidance:
    """Actionable error recovery guidance."""
    error_summary: str
    recovery_steps: List[str]
    anti_patterns: List[str]  # What NOT to do
    example_fix: Optional[str] = None
    related_docs: List[str] = field(default_factory=list)


# Add to ERROR_PATTERNS - expanded bioinformatics patterns
BIOINFORMATICS_ERROR_PATTERNS = {
    ErrorType.TOOL: [
        # STAR-specific
        (r"EXITING because of FATAL ERROR in reads input: quality string length is not equal to sequence length",
         "FASTQ file is corrupted or has mismatched quality scores. Re-download or validate with FastQC."),
        (r"STAR.*Genome version.*is INCOMPATIBLE",
         "STAR index was built with different version. Rebuild index or use matching STAR version."),
        
        # BWA-specific  
        (r"bwa.*fail to open file",
         "BWA cannot open input file. Check file path, permissions, and that index files exist."),
        
        # Samtools-specific
        (r"samtools.*truncated file",
         "BAM/SAM file is truncated. Re-run alignment or download complete file."),
        (r"samtools.*invalid BAM binary header",
         "BAM file has corrupted header. File may be truncated or wrong format."),
        
        # GATK-specific
        (r"MalformedRead.*CIGAR.*extends past end",
         "Read has invalid CIGAR string. May need to reprocess BAM with CleanSam."),
        (r"htsjdk.*Contig.*not in sequence dictionary",
         "Reference contigs don't match. Ensure BAM and reference have same chromosome naming."),
        
        # General I/O
        (r"gzip: stdin: unexpected end of file",
         "Compressed file is truncated. Re-download or check disk space during creation."),
    ],
    ErrorType.MEMORY: [
        (r"java\.lang\.OutOfMemoryError: GC overhead limit exceeded",
         "Java spending too much time in garbage collection. Increase heap (-Xmx) or process smaller batches."),
        (r"std::bad_alloc",
         "C++ memory allocation failed. Increase memory or use streaming/chunked processing."),
    ],
}


def generate_error_guidance(diagnosis: DiagnosisResult) -> ErrorGuidance:
    """
    Generate actionable error recovery guidance.
    
    Converts passive diagnosis into active steps with anti-patterns.
    """
    recovery_steps = []
    anti_patterns = []
    example_fix = None
    related_docs = []
    
    error_type = diagnosis.error_type
    
    if error_type == ErrorType.MEMORY:
        recovery_steps = [
            "1. Check current memory allocation in workflow config",
            "2. Increase memory: `process.memory = '32 GB'` or `--mem=32G`",
            "3. If still failing, reduce parallelism: `process.maxForks = 2`",
            "4. Consider processing samples in smaller batches",
        ]
        anti_patterns = [
            "Don't increase memory beyond node limits (wastes resources)",
            "Don't just retry without changing parameters",
            "Avoid running multiple memory-heavy processes in parallel",
        ]
        example_fix = """
// nextflow.config
process {
    withName: 'STAR_ALIGN' {
        memory = { 32.GB * task.attempt }
        maxRetries = 2
        errorStrategy = 'retry'
    }
}
"""
        related_docs = [
            "https://www.nextflow.io/docs/latest/process.html#memory",
            "https://nf-co.re/docs/usage/configuration#max-resources",
        ]
    
    elif error_type == ErrorType.DISK:
        recovery_steps = [
            "1. Check available disk space: `df -h /path/to/workDir`",
            "2. Clean up intermediate files: `nextflow clean -f -before <run_name>`",
            "3. Set work directory to larger filesystem: `-work-dir /scratch/work`",
            "4. Enable automatic cleanup: `cleanup = true` in config",
        ]
        anti_patterns = [
            "Don't delete active work directories during a run",
            "Don't ignore disk warnings - they cause data corruption",
            "Avoid storing large intermediates in home directory",
        ]
        
    elif error_type == ErrorType.SLURM:
        recovery_steps = [
            "1. Check job status: `sacct -j <jobid> --format=JobID,State,ExitCode,MaxRSS`",
            "2. Review SLURM logs: `cat slurm-<jobid>.out`",
            "3. Adjust resource requests in config or process directive",
            "4. Consider using `errorStrategy 'ignore'` for non-critical steps",
        ]
        anti_patterns = [
            "Don't request more resources than available on any node",
            "Don't set walltime too short for large datasets",
            "Avoid submitting many jobs simultaneously without throttling",
        ]
        
    elif error_type == ErrorType.NEXTFLOW:
        recovery_steps = [
            "1. Check process definition for syntax errors",
            "2. Validate input channels with `.view()` operator",
            "3. Ensure output declarations match actual file patterns",
            "4. Run with `-with-trace` to identify bottlenecks",
        ]
        anti_patterns = [
            "Don't mix DSL1 and DSL2 syntax",
            "Don't hardcode paths - use params or input channels",
            "Avoid using `.collect()` on large datasets (creates bottleneck)",
        ]
    
    elif error_type == ErrorType.TOOL:
        recovery_steps = [
            "1. Verify input file integrity: `gzip -t file.fastq.gz`",
            "2. Check tool version compatibility with reference/index",
            "3. Review tool-specific documentation for error code",
            "4. Test with a small subset to isolate the issue",
        ]
        anti_patterns = [
            "Don't assume all FASTQ files are valid without checking",
            "Don't mix reference genome versions within a pipeline",
            "Avoid using tool defaults for production - explicitly set parameters",
        ]
    
    else:  # UNKNOWN or other
        recovery_steps = [
            "1. Review full error log for additional context",
            "2. Search error message in tool documentation",
            "3. Check BioPipelines troubleshooting guide",
            "4. Submit issue with full logs if unresolved",
        ]
        anti_patterns = [
            "Don't ignore warnings that precede errors",
            "Don't modify multiple parameters at once when debugging",
        ]
    
    return ErrorGuidance(
        error_summary=f"{error_type.value.upper()}: {diagnosis.root_cause}",
        recovery_steps=recovery_steps,
        anti_patterns=anti_patterns,
        example_fix=example_fix,
        related_docs=related_docs,
    )


def format_guidance_for_agent(guidance: ErrorGuidance) -> str:
    """Format error guidance for injection into agent context."""
    parts = [
        f"## Error Encountered",
        f"**Summary:** {guidance.error_summary}",
        "",
        "## Recovery Strategy",
    ]
    parts.extend(guidance.recovery_steps)
    
    parts.append("")
    parts.append("## What NOT to Do")
    for ap in guidance.anti_patterns:
        parts.append(f"- {ap}")
    
    if guidance.example_fix:
        parts.append("")
        parts.append("## Example Fix")
        parts.append("```")
        parts.append(guidance.example_fix.strip())
        parts.append("```")
    
    return "\n".join(parts)
```

---

## Phase 3: Concise Memory (P2)

### 3.1 Concise Memory Agent Pattern

**File:** `src/workflow_composer/agents/memory/concise_memory.py`

**Purpose:** Implement "clean slate" memory pattern for multi-step workflows.

**DeepCode Pattern:**
```python
# From DeepCode: workflows/agents/memory_agent_concise.py
class ConciseMemoryAgent:
    """Clean slate approach - after each file write, start fresh but preserve:
    - system_prompt (never changes)
    - initial_plan (the task description)  
    - Compressed summaries instead of full conversation
    """
```

**BioPipelines Implementation:**
```python
"""
Concise Memory Agent
====================

Implements the "clean slate" memory pattern from DeepCode.

Key principles:
1. Preserve: System prompt + initial query + completed summaries
2. Discard: Full conversation history, file contents, intermediate tool outputs
3. Compress: Convert completed steps to one-line summaries
4. Result: 70-80% token reduction in long-running workflows
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

from .token_tracker import TokenTracker, TokenBudget

logger = logging.getLogger(__name__)


@dataclass
class CompletedStep:
    """Summary of a completed workflow step."""
    step_num: int
    action: str
    summary: str  # One-line summary
    timestamp: datetime = field(default_factory=datetime.now)
    
    def __str__(self) -> str:
        return f"✓ Step {self.step_num}: {self.summary}"


@dataclass 
class ConciseState:
    """Minimal state preserved across clean slate resets."""
    system_prompt: str
    initial_query: str
    completed_steps: List[CompletedStep] = field(default_factory=list)
    current_context: Dict[str, Any] = field(default_factory=dict)
    
    def get_context_summary(self) -> str:
        """Get formatted summary for prompt injection."""
        parts = ["## Completed Steps"]
        for step in self.completed_steps:
            parts.append(str(step))
        
        if self.current_context:
            parts.append("")
            parts.append("## Current Context")
            for key, value in self.current_context.items():
                parts.append(f"- {key}: {value}")
        
        return "\n".join(parts)


class ConciseMemory:
    """
    Memory manager with aggressive compression.
    
    After each major action (file write, tool completion), triggers
    compression to prevent context overflow.
    
    Example:
        memory = ConciseMemory(system_prompt=SYSTEM_PROMPT)
        memory.set_initial_query(user_query)
        
        # During workflow...
        result = execute_action(action)
        memory.complete_step(
            step_num=1,
            action="STAR_ALIGN",
            summary="Aligned 48 samples with STAR (96% mapped)"
        )
        
        # Get compressed context for next LLM call
        context = memory.get_prompt_context()
    """
    
    def __init__(
        self,
        system_prompt: str,
        token_budget: Optional[TokenBudget] = None,
        auto_compress: bool = True,
    ):
        self.state = ConciseState(
            system_prompt=system_prompt,
            initial_query="",
        )
        self.token_tracker = TokenTracker(budget=token_budget)
        self.auto_compress = auto_compress
        
        # Track full history for debugging (not sent to LLM)
        self._full_history: List[Dict[str, Any]] = []
        
        # Track current working context (will be compressed)
        self._working_context: List[Dict[str, str]] = []
    
    def set_initial_query(self, query: str):
        """Set the initial user query (preserved across resets)."""
        self.state.initial_query = query
        self.token_tracker.add_system_prompt(self.state.system_prompt)
    
    def add_working_message(self, role: str, content: str):
        """Add a message to working context (may be compressed)."""
        self._working_context.append({"role": role, "content": content})
        self._full_history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
        })
        
        self.token_tracker.add_message(role, content)
        
        # Check if compression needed
        if self.auto_compress and self.token_tracker.should_compress():
            logger.info("Token budget exceeded, triggering compression")
            self._compress()
    
    def complete_step(
        self,
        step_num: int,
        action: str,
        summary: str,
        context_updates: Optional[Dict[str, Any]] = None,
    ):
        """
        Mark a step as complete and compress.
        
        This is the key "clean slate" trigger - after completing a step,
        we discard the detailed working context and keep only the summary.
        """
        completed = CompletedStep(
            step_num=step_num,
            action=action,
            summary=summary,
        )
        self.state.completed_steps.append(completed)
        
        if context_updates:
            self.state.current_context.update(context_updates)
        
        # Clean slate: discard working context
        self._working_context = []
        
        # Reset token tracker for fresh start
        self.token_tracker = TokenTracker(budget=self.token_tracker.budget)
        self.token_tracker.add_system_prompt(self.state.system_prompt)
        
        logger.info(f"Completed step {step_num}, clean slate applied")
    
    def _compress(self):
        """Compress working context to summaries."""
        if not self._working_context:
            return
        
        # Create summary of working context
        # In production, this could use an LLM to summarize
        summary_parts = []
        for msg in self._working_context[-5:]:  # Keep last 5 messages
            if msg["role"] == "assistant":
                # Truncate long responses
                content = msg["content"][:200]
                if len(msg["content"]) > 200:
                    content += "..."
                summary_parts.append(f"Assistant: {content}")
        
        # Store compression event
        self._full_history.append({
            "type": "compression",
            "messages_compressed": len(self._working_context),
            "timestamp": datetime.now().isoformat(),
        })
        
        # Clear working context
        self._working_context = []
        
        # Reset token tracker
        self.token_tracker = TokenTracker(budget=self.token_tracker.budget)
        self.token_tracker.add_system_prompt(self.state.system_prompt)
    
    def get_prompt_context(self) -> List[Dict[str, str]]:
        """
        Get messages for LLM prompt.
        
        Returns minimal context:
        1. System prompt
        2. Completed steps summary
        3. Initial query
        4. Recent working context (if any)
        """
        messages = [
            {"role": "system", "content": self.state.system_prompt}
        ]
        
        # Add completed steps if any
        if self.state.completed_steps:
            context_summary = self.state.get_context_summary()
            messages.append({
                "role": "system",
                "content": f"Previous progress:\n{context_summary}"
            })
        
        # Add initial query
        messages.append({
            "role": "user",
            "content": self.state.initial_query
        })
        
        # Add recent working context
        messages.extend(self._working_context)
        
        return messages
    
    def get_token_status(self) -> Dict[str, Any]:
        """Get current token usage status."""
        return self.token_tracker.get_status()
    
    def get_full_history(self) -> List[Dict[str, Any]]:
        """Get full history for debugging (not for LLM)."""
        return self._full_history.copy()
    
    def save_checkpoint(self) -> Dict[str, Any]:
        """Save state for recovery."""
        return {
            "system_prompt": self.state.system_prompt,
            "initial_query": self.state.initial_query,
            "completed_steps": [
                {
                    "step_num": s.step_num,
                    "action": s.action,
                    "summary": s.summary,
                    "timestamp": s.timestamp.isoformat(),
                }
                for s in self.state.completed_steps
            ],
            "current_context": self.state.current_context,
        }
    
    def restore_checkpoint(self, checkpoint: Dict[str, Any]):
        """Restore state from checkpoint."""
        self.state.system_prompt = checkpoint["system_prompt"]
        self.state.initial_query = checkpoint["initial_query"]
        self.state.completed_steps = [
            CompletedStep(
                step_num=s["step_num"],
                action=s["action"],
                summary=s["summary"],
                timestamp=datetime.fromisoformat(s["timestamp"]),
            )
            for s in checkpoint.get("completed_steps", [])
        ]
        self.state.current_context = checkpoint.get("current_context", {})
```

---

## Phase 4: Additional BioPipelines-Specific Enhancements

### 4.1 LLM Response Validator

**File:** `src/workflow_composer/agents/utils/response_validator.py`

**Purpose:** Validate LLM response structure before parsing.

```python
"""
LLM Response Validator
======================

Validates LLM responses before parsing to catch issues early.
"""

import re
from typing import Optional, List, Tuple, Any
from dataclasses import dataclass


@dataclass
class ValidationResult:
    """Result of response validation."""
    valid: bool
    errors: List[str]
    warnings: List[str]
    repaired_content: Optional[str] = None


class ResponseValidator:
    """Validates LLM responses for expected structure."""
    
    @staticmethod
    def validate_react_response(response: str) -> ValidationResult:
        """
        Validate ReAct agent response format.
        
        Expected:
        Thought: ...
        Action: ...
        Action Input: {...}
        """
        errors = []
        warnings = []
        
        has_thought = bool(re.search(r'^Thought:', response, re.MULTILINE))
        has_action = bool(re.search(r'^Action:', response, re.MULTILINE))
        has_input = bool(re.search(r'^Action Input:', response, re.MULTILINE))
        
        if not has_thought:
            warnings.append("Missing 'Thought:' prefix")
        
        if has_input and not has_action:
            errors.append("Has 'Action Input' without 'Action'")
        
        # Check for valid JSON in action input
        if has_input:
            input_match = re.search(r'Action Input:\s*(.+)', response, re.DOTALL)
            if input_match:
                input_content = input_match.group(1).strip()
                if input_content and not (input_content.startswith('{') or input_content.startswith('[')):
                    warnings.append("Action Input doesn't appear to be JSON")
        
        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )
    
    @staticmethod
    def validate_diagnosis_response(response: str) -> ValidationResult:
        """Validate coding agent diagnosis response."""
        errors = []
        warnings = []
        
        required_fields = ["ERROR TYPE:", "ROOT CAUSE:", "EXPLANATION:"]
        for field in required_fields:
            if field.lower() not in response.lower():
                warnings.append(f"Missing field: {field}")
        
        return ValidationResult(
            valid=True,  # Diagnosis can work with partial info
            errors=errors,
            warnings=warnings,
        )
    
    @staticmethod
    def validate_json_response(
        response: str,
        required_keys: Optional[List[str]] = None,
    ) -> ValidationResult:
        """Validate JSON response structure."""
        from .json_repair import extract_json_from_text, safe_json_loads
        
        errors = []
        warnings = []
        repaired = None
        
        json_str = extract_json_from_text(response)
        if not json_str:
            errors.append("No JSON found in response")
            return ValidationResult(valid=False, errors=errors, warnings=warnings)
        
        data = safe_json_loads(json_str)
        if data is None:
            errors.append("Failed to parse JSON even after repair")
            return ValidationResult(valid=False, errors=errors, warnings=warnings)
        
        # Check required keys
        if required_keys and isinstance(data, dict):
            for key in required_keys:
                if key not in data:
                    warnings.append(f"Missing expected key: {key}")
        
        return ValidationResult(
            valid=True,
            errors=errors,
            warnings=warnings,
            repaired_content=json_str,
        )
```

### 4.2 Graceful Degradation Chain

**File:** `src/workflow_composer/agents/utils/degradation.py`

**Purpose:** Provide fallback hierarchy when primary methods fail.

```python
"""
Graceful Degradation Chain
==========================

Provides fallback hierarchy for agent operations.

Pattern:
1. Try: Full LLM-powered response
2. Fallback: Simpler LLM prompt
3. Fallback: Pattern matching
4. Fallback: Default/cached response
"""

import logging
from typing import Callable, TypeVar, Optional, List, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class FallbackResult:
    """Result from degradation chain."""
    value: T
    method_used: str
    fallback_level: int  # 0 = primary, 1+ = fallback
    error_chain: List[str]


class DegradationChain:
    """
    Executes a chain of methods with automatic fallback.
    
    Example:
        chain = DegradationChain()
        chain.add("llm_diagnosis", lambda: llm_diagnose(error))
        chain.add("pattern_match", lambda: pattern_diagnose(error))
        chain.add("default", lambda: default_diagnosis())
        
        result = chain.execute()
        print(f"Used method: {result.method_used}")
    """
    
    def __init__(self):
        self._methods: List[tuple] = []
    
    def add(
        self,
        name: str,
        method: Callable[[], T],
        condition: Optional[Callable[[], bool]] = None,
    ):
        """
        Add a method to the chain.
        
        Args:
            name: Method identifier for logging
            method: Callable that returns result or raises exception
            condition: Optional check before attempting method
        """
        self._methods.append((name, method, condition))
        return self  # Allow chaining
    
    def execute(self, default: T = None) -> FallbackResult:
        """
        Execute the chain, returning first successful result.
        """
        error_chain = []
        
        for i, (name, method, condition) in enumerate(self._methods):
            # Check condition if present
            if condition is not None:
                try:
                    if not condition():
                        logger.debug(f"Skipping {name}: condition not met")
                        continue
                except Exception as e:
                    logger.debug(f"Condition check failed for {name}: {e}")
                    continue
            
            # Try method
            try:
                result = method()
                if result is not None:
                    logger.info(f"Degradation chain: {name} succeeded at level {i}")
                    return FallbackResult(
                        value=result,
                        method_used=name,
                        fallback_level=i,
                        error_chain=error_chain,
                    )
            except Exception as e:
                error_msg = f"{name}: {type(e).__name__}: {str(e)}"
                error_chain.append(error_msg)
                logger.warning(f"Degradation chain: {error_msg}")
        
        # All methods failed
        logger.error("All methods in degradation chain failed")
        return FallbackResult(
            value=default,
            method_used="default",
            fallback_level=len(self._methods),
            error_chain=error_chain,
        )
```

---

## Implementation Order

### Week 1: Core Utilities ✅ COMPLETED
1. [x] Create `src/workflow_composer/agents/utils/__init__.py`
2. [x] Implement `json_repair.py` with tests
3. [x] Implement `retry_strategy.py` with tests
4. [x] Implement `response_validator.py`
5. [x] Implement `degradation.py`
6. [x] Implement `error_guidance.py`

### Week 2: Token & Memory Management ✅ COMPLETED
7. [x] Implement `token_tracker.py` with tests
8. [x] Implement `concise_memory.py` with tests  
9. [x] Create `tests/test_deepcode_utilities.py` (45 tests)
10. [x] Create `tests/test_deepcode_memory.py` (38 tests)

**Test Results:**
- 83 tests passing
- All utility modules functional
- All memory modules functional

### Week 3: Agent Integration ✅ COMPLETED
11. [x] Integrate JSON repair into `react_agent.py`
12. [x] Integrate adaptive retry into `coding_agent.py`
13. [x] Add ConciseMemory to long-running agent workflows
14. [x] Add error guidance generation to CodingAgent
15. [x] Add token tracking to ReactAgent

**Integration Summary:**
- CodingAgent enhanced with:
  - Adaptive retry with parameter reduction
  - JSON repair for LLM outputs
  - Graceful degradation chain
  - Error guidance generation
- ReactAgent enhanced with:
  - JSON repair for action input parsing
  - Response validation
  - Token tracking
  - Optional concise memory for long workflows

### Week 4: Reference Discovery & Codebase Indexing ✅ COMPLETED
16. [x] Implement `ReferenceDiscoveryAgent` for nf-core/modules discovery
17. [x] Implement `CodebaseIndexer` for existing codebase indexing
18. [x] Create `tests/test_reference_discovery.py` (18 tests)
19. [x] Create `tests/test_codebase_indexer.py` (21 tests)
20. [x] Update specialists `__init__.py` exports

**New Components:**
- **ReferenceDiscoveryAgent**: Discovers relevant code references from:
  - nf-core/modules (17+ tools indexed)
  - nf-core/pipelines (10+ analysis types)
  - Local knowledge base
  - GitHub repositories (optional, requires token)
  - Includes module snippets for common tools

- **CodebaseIndexer**: Indexes existing Nextflow codebases:
  - Parses processes, workflows, channels, parameters
  - Detects containers and tools automatically
  - Builds relationship graphs between components
  - Supports save/load for persistent indices
  - Provides search and similarity matching

**Test Results:**
- 39 additional tests passing
- 122+ total tests across DeepCode enhancements

### Week 5: ToolOrchestra Integration ✅ COMPLETED
21. [x] Implement `Orchestrator8B` for intelligent model routing
22. [x] Implement `OrchestratedSupervisor` (SupervisorAgent + Orchestrator-8B)
23. [x] Create `tests/test_orchestrator_8b.py` (29 tests)
24. [x] Create `docs/ORCHESTRATOR_8B_INTEGRATION.md`
25. [x] Update module exports in `llm/__init__.py` and `specialists/__init__.py`

**New Components:**
- **Orchestrator8B**: NVIDIA's RL-trained orchestrator from ToolOrchestra paper:
  - Routes queries to optimal model tier (local_small, local_large, cloud_small, cloud_large)
  - User preference alignment (cost, speed, accuracy, balanced)
  - Multi-turn tool orchestration
  - Supports vLLM, transformers, or API backends
  - Heuristic fallback when model not loaded

- **OrchestratedSupervisor**: SupervisorAgent enhanced with Orchestrator-8B:
  - Cost-aware routing decisions
  - Automatic model tier selection
  - Detailed metadata (cost, models used, reasoning)
  - Graceful fallback to standard execution

- **BioPipeline Tool Catalog**: Pre-configured tools for orchestrator:
  - workflow_planner, code_generator, code_validator
  - nfcore_reference, container_selector, cloud_expert

**Test Results:**
- 29 additional tests passing
- 151+ total tests across DeepCode enhancements

**Reference:** [ToolOrchestra Paper (arXiv:2511.21689)](https://arxiv.org/abs/2511.21689)

### Week 6: Integration & Documentation (NEXT)
26. [ ] Integration tests for full workflow with orchestrator
27. [ ] vLLM deployment guide for Orchestrator-8B
28. [ ] Fine-tuning data collection for BioPipelines-specific routing
29. [ ] Performance benchmarks comparing with/without orchestrator
30. [ ] Create cost savings report

---

## Success Metrics

| Metric | Current | Target |
|--------|---------|--------|
| JSON parse failures | Unknown | < 5% |
| LLM retry success rate | Unknown | > 80% |
| Token usage efficiency | N/A | 70%+ reduction with ConciseMemory |
| Error diagnosis accuracy | Pattern-only | Pattern + LLM + Guidance |
| Reference discovery coverage | N/A | 90%+ common tools/pipelines |
| Codebase indexing accuracy | N/A | 95%+ process detection |
| **Cost efficiency (with Orchestrator)** | **N/A** | **80%+ reduction** |
| **Routing accuracy** | **N/A** | **>90% optimal tier** |

---

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Token counting accuracy varies by model | Support multiple backends, conservative estimates |
| JSON repair may produce invalid semantics | Validate repaired JSON structure |
| Aggressive compression loses context | Keep full history for debugging, allow manual override |
| Retry storms on persistent failures | Exponential backoff, max attempts |

---

## References

- DeepCode Repository: https://github.com/HKUDS/DeepCode
- ToolOrchestra Paper: https://arxiv.org/abs/2511.21689
- Orchestrator-8B Model: https://huggingface.co/nvidia/Orchestrator-8B
- Key files analyzed:
  - `workflows/agents/memory_agent_concise.py`
  - `workflows/agents/code_implementation_agent.py`
  - `workflows/agent_orchestration_engine.py`
  - `workflows/code_implementation_workflow.py`
