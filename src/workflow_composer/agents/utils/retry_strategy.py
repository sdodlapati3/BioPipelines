"""
Retry Strategy Module
=====================

Implements intelligent retry with adaptive parameter adjustment.

Key insight from DeepCode: On retry, REDUCE output size and temperature.
This is counterintuitive but effective because:
- Smaller output = less chance of truncation/malformation
- Lower temperature = more deterministic/stable output
- Reduced complexity = more likely to succeed

References:
    - DeepCode: workflows/agent_orchestration_engine.py (_adjust_params_for_retry)
"""

import time
import asyncio
import logging
from typing import TypeVar, Callable, Any, Dict, Optional, List, Union
from dataclasses import dataclass, field
from functools import wraps

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class RetryConfig:
    """
    Configuration for retry behavior.
    
    Attributes:
        max_attempts: Maximum number of attempts (including first try)
        base_delay: Initial delay in seconds before first retry
        max_delay: Maximum delay cap for exponential backoff
        exponential_base: Base for exponential backoff calculation
        token_reduction_factor: Factor to multiply max_tokens by on each retry
        temperature_reduction: Amount to subtract from temperature on each retry
        min_temperature: Minimum temperature to use
        min_tokens: Minimum max_tokens to use
        jitter: Add randomness to delay (helps prevent thundering herd)
    """
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 30.0
    exponential_base: float = 2.0
    
    # Parameter reduction on retry (key insight from DeepCode)
    token_reduction_factor: float = 0.75   # Reduce by 25% each retry
    temperature_reduction: float = 0.2      # Reduce by 0.2 each retry
    min_temperature: float = 0.1
    min_tokens: int = 512
    
    # Jitter
    jitter: float = 0.1  # ±10% randomness


@dataclass
class RetryState:
    """
    Tracks retry state across attempts.
    
    Useful for debugging and monitoring retry behavior.
    """
    attempt: int = 0
    last_error: Optional[str] = None
    last_error_type: Optional[str] = None
    adjusted_params: Dict[str, Any] = field(default_factory=dict)
    total_delay: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "attempt": self.attempt,
            "last_error": self.last_error,
            "last_error_type": self.last_error_type,
            "adjusted_params": self.adjusted_params,
            "total_delay": self.total_delay,
        }


def adjust_llm_params_for_retry(
    params: Dict[str, Any],
    attempt: int,
    config: Optional[RetryConfig] = None,
    context_size: int = 32768,
) -> Dict[str, Any]:
    """
    Adjust LLM parameters for retry attempt.
    
    Strategy: REDUCE complexity on retry for higher success rate.
    
    This is the key pattern from DeepCode: when an LLM call fails,
    reducing the output size and temperature often helps because:
    1. Shorter outputs are less likely to be truncated
    2. Lower temperature produces more consistent outputs
    3. Simpler responses are easier to parse
    
    Args:
        params: Original LLM parameters dict
        attempt: Current attempt number (0 = first try, 1 = first retry)
        config: Retry configuration (uses default if None)
        context_size: Model's context window size (affects token limits)
        
    Returns:
        New parameters dict with adjusted values
        
    Examples:
        >>> params = {"max_tokens": 1024, "temperature": 0.7}
        >>> adjust_llm_params_for_retry(params, attempt=1)
        {'max_tokens': 768, 'temperature': 0.5}
    """
    if config is None:
        config = RetryConfig()
    
    adjusted = params.copy()
    
    # First attempt - no changes
    if attempt == 0:
        return adjusted
    
    # Reduce max_tokens (multiplicative reduction)
    if "max_tokens" in adjusted:
        original_tokens = adjusted["max_tokens"]
        factor = config.token_reduction_factor ** attempt
        new_tokens = max(
            config.min_tokens,
            int(original_tokens * factor)
        )
        adjusted["max_tokens"] = new_tokens
        logger.debug(
            f"Retry {attempt}: Reduced max_tokens {original_tokens} -> {new_tokens}"
        )
    
    # Reduce temperature (additive reduction)
    if "temperature" in adjusted:
        original_temp = adjusted["temperature"]
        new_temp = max(
            config.min_temperature,
            original_temp - (config.temperature_reduction * attempt)
        )
        adjusted["temperature"] = new_temp
        logger.debug(
            f"Retry {attempt}: Reduced temperature {original_temp:.2f} -> {new_temp:.2f}"
        )
    
    # Add stop sequences on later retries (helps prevent runaway generation)
    if attempt >= 2 and "stop" not in adjusted:
        adjusted["stop"] = ["\n\n\n", "```\n\n", "---\n"]
        logger.debug(f"Retry {attempt}: Added stop sequences")
    
    return adjusted


def get_retry_delay(
    attempt: int,
    config: Optional[RetryConfig] = None,
) -> float:
    """
    Calculate delay before next retry with exponential backoff.
    
    Uses exponential backoff with optional jitter to prevent
    thundering herd problems.
    
    Args:
        attempt: Current attempt number (0 = first try)
        config: Retry configuration
        
    Returns:
        Delay in seconds before next attempt
        
    Examples:
        >>> get_retry_delay(0)  # No delay for first attempt
        0.0
        >>> get_retry_delay(1)  # First retry: ~1 second
        1.0
        >>> get_retry_delay(2)  # Second retry: ~2 seconds
        2.0
    """
    if config is None:
        config = RetryConfig()
    
    if attempt == 0:
        return 0.0
    
    # Exponential backoff
    delay = config.base_delay * (config.exponential_base ** (attempt - 1))
    delay = min(delay, config.max_delay)
    
    # Add jitter
    if config.jitter > 0:
        import random
        jitter_range = delay * config.jitter
        delay += random.uniform(-jitter_range, jitter_range)
        delay = max(0.1, delay)  # Ensure positive delay
    
    return delay


def with_retry(
    config: Optional[RetryConfig] = None,
    retry_on: tuple = (Exception,),
    on_retry: Optional[Callable[[int, Exception], None]] = None,
    on_success: Optional[Callable[[int], None]] = None,
):
    """
    Decorator for retrying functions with exponential backoff.
    
    Args:
        config: Retry configuration
        retry_on: Tuple of exception types to retry on
        on_retry: Callback called before each retry (attempt, exception)
        on_success: Callback called on success (attempt)
        
    Returns:
        Decorated function
        
    Examples:
        >>> @with_retry(config=RetryConfig(max_attempts=3))
        ... def unreliable_function():
        ...     # May fail sometimes
        ...     pass
    """
    if config is None:
        config = RetryConfig()
    
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_error = None
            
            for attempt in range(config.max_attempts):
                try:
                    result = func(*args, **kwargs)
                    if on_success:
                        on_success(attempt)
                    return result
                    
                except retry_on as e:
                    last_error = e
                    
                    if attempt < config.max_attempts - 1:
                        delay = get_retry_delay(attempt + 1, config)
                        logger.warning(
                            f"Attempt {attempt + 1}/{config.max_attempts} failed: "
                            f"{type(e).__name__}: {str(e)[:100]}. "
                            f"Retrying in {delay:.1f}s..."
                        )
                        
                        if on_retry:
                            on_retry(attempt, e)
                        
                        time.sleep(delay)
                    else:
                        logger.error(
                            f"All {config.max_attempts} attempts failed. "
                            f"Last error: {type(e).__name__}: {str(e)[:200]}"
                        )
            
            raise last_error
        
        return wrapper
    
    return decorator


def with_async_retry(
    config: Optional[RetryConfig] = None,
    retry_on: tuple = (Exception,),
    on_retry: Optional[Callable[[int, Exception], None]] = None,
):
    """
    Async version of with_retry decorator.
    
    Usage:
        @with_async_retry(config=RetryConfig(max_attempts=3))
        async def async_llm_call():
            ...
    """
    if config is None:
        config = RetryConfig()
    
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            last_error = None
            
            for attempt in range(config.max_attempts):
                try:
                    return await func(*args, **kwargs)
                    
                except retry_on as e:
                    last_error = e
                    
                    if attempt < config.max_attempts - 1:
                        delay = get_retry_delay(attempt + 1, config)
                        logger.warning(
                            f"Async attempt {attempt + 1}/{config.max_attempts} failed: "
                            f"{type(e).__name__}: {str(e)[:100]}. "
                            f"Retrying in {delay:.1f}s..."
                        )
                        
                        if on_retry:
                            on_retry(attempt, e)
                        
                        await asyncio.sleep(delay)
                    else:
                        logger.error(f"All {config.max_attempts} async attempts failed")
            
            raise last_error
        
        return wrapper
    
    return decorator


class AdaptiveLLMCaller:
    """
    Wrapper for LLM calls with adaptive retry.
    
    Combines retry logic with parameter adjustment for maximum
    reliability. On each retry, automatically reduces max_tokens
    and temperature.
    
    Attributes:
        client: OpenAI-compatible LLM client
        model: Model name/identifier
        config: Retry configuration
        last_state: State from most recent call (for debugging)
        
    Examples:
        >>> from openai import OpenAI
        >>> client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
        >>> caller = AdaptiveLLMCaller(client, model="llama-3.3-70b")
        >>> response = caller.call(
        ...     messages=[{"role": "user", "content": "Hello"}],
        ...     max_tokens=1024,
        ...     temperature=0.7
        ... )
    """
    
    def __init__(
        self,
        client: Any,
        model: str,
        config: Optional[RetryConfig] = None,
    ):
        """
        Initialize the adaptive LLM caller.
        
        Args:
            client: OpenAI-compatible client instance
            model: Model name to use for completions
            config: Retry configuration (uses default if None)
        """
        self.client = client
        self.model = model
        self.config = config or RetryConfig()
        self._state = RetryState()
    
    def call(
        self,
        messages: List[Dict[str, str]],
        **kwargs,
    ) -> Any:
        """
        Make LLM call with adaptive retry.
        
        On failure, retries with reduced max_tokens and temperature.
        
        Args:
            messages: List of chat messages
            **kwargs: Additional parameters (max_tokens, temperature, etc.)
            
        Returns:
            LLM response object
            
        Raises:
            Last exception if all retries fail
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
                
                logger.debug(f"LLM call succeeded on attempt {attempt + 1}")
                return response
                
            except Exception as e:
                last_error = e
                self._state.last_error = str(e)
                self._state.last_error_type = type(e).__name__
                
                if attempt < self.config.max_attempts - 1:
                    delay = get_retry_delay(attempt + 1, self.config)
                    self._state.total_delay += delay
                    
                    logger.warning(
                        f"LLM call attempt {attempt + 1}/{self.config.max_attempts} failed: "
                        f"{type(e).__name__}: {str(e)[:100]}. "
                        f"Adjusting params and retrying in {delay:.1f}s"
                    )
                    time.sleep(delay)
                else:
                    logger.error(
                        f"LLM call failed after {self.config.max_attempts} attempts. "
                        f"Last error: {str(e)[:200]}"
                    )
        
        raise last_error
    
    async def call_async(
        self,
        messages: List[Dict[str, str]],
        **kwargs,
    ) -> Any:
        """
        Async version of call().
        
        Same behavior as call() but uses asyncio for delays.
        """
        self._state = RetryState()
        last_error = None
        
        for attempt in range(self.config.max_attempts):
            self._state.attempt = attempt
            
            adjusted_params = adjust_llm_params_for_retry(
                kwargs, attempt, self.config
            )
            self._state.adjusted_params = adjusted_params
            
            try:
                # Handle both sync and async create methods
                create_method = self.client.chat.completions.create
                if asyncio.iscoroutinefunction(create_method):
                    response = await create_method(
                        model=self.model,
                        messages=messages,
                        **adjusted_params,
                    )
                else:
                    response = create_method(
                        model=self.model,
                        messages=messages,
                        **adjusted_params,
                    )
                
                return response
                
            except Exception as e:
                last_error = e
                self._state.last_error = str(e)
                self._state.last_error_type = type(e).__name__
                
                if attempt < self.config.max_attempts - 1:
                    delay = get_retry_delay(attempt + 1, self.config)
                    self._state.total_delay += delay
                    
                    logger.warning(
                        f"Async LLM call attempt {attempt + 1} failed: "
                        f"{type(e).__name__}. Retrying in {delay:.1f}s"
                    )
                    await asyncio.sleep(delay)
        
        raise last_error
    
    @property
    def last_state(self) -> RetryState:
        """Get state from last call for debugging."""
        return self._state
    
    def reset_state(self):
        """Reset retry state."""
        self._state = RetryState()


# Specialized retry configurations for different scenarios
DIAGNOSTIC_RETRY_CONFIG = RetryConfig(
    max_attempts=3,
    token_reduction_factor=0.75,
    temperature_reduction=0.15,
    min_temperature=0.05,  # Very low for diagnostic precision
)

CODE_GENERATION_RETRY_CONFIG = RetryConfig(
    max_attempts=4,
    token_reduction_factor=0.8,
    temperature_reduction=0.1,
    min_temperature=0.1,
    min_tokens=1024,  # Need more tokens for code
)

VALIDATION_RETRY_CONFIG = RetryConfig(
    max_attempts=2,
    token_reduction_factor=0.5,  # Aggressive reduction
    temperature_reduction=0.3,
    min_temperature=0.0,  # Zero temperature for deterministic validation
)
