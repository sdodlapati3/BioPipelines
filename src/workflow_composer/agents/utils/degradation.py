"""
Graceful Degradation Chain
==========================

Provides fallback hierarchy for agent operations.

Pattern:
1. Try: Full LLM-powered response
2. Fallback: Simpler LLM prompt
3. Fallback: Pattern matching
4. Fallback: Default/cached response

This ensures operations have multiple paths to success,
improving reliability in production environments.

Example use case:
    Error diagnosis tries:
    1. Full LLM analysis with context
    2. LLM with simplified prompt
    3. Regex pattern matching
    4. Generic error message
"""

import logging
from typing import TypeVar, Callable, Optional, List, Any, Generic
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class FallbackResult(Generic[T]):
    """
    Result from degradation chain execution.
    
    Attributes:
        value: The result value (from whichever method succeeded)
        method_used: Name of the method that produced the result
        fallback_level: 0 = primary, 1+ = fallback levels
        error_chain: List of error messages from failed attempts
        execution_time: Time taken for successful execution (seconds)
        metadata: Additional metadata from the successful method
    """
    value: T
    method_used: str
    fallback_level: int
    error_chain: List[str] = field(default_factory=list)
    execution_time: float = 0.0
    metadata: dict = field(default_factory=dict)
    
    @property
    def used_fallback(self) -> bool:
        """Check if a fallback method was used."""
        return self.fallback_level > 0
    
    @property
    def all_errors(self) -> str:
        """Get all errors as a single string."""
        return "; ".join(self.error_chain)
    
    def to_dict(self) -> dict:
        return {
            "method_used": self.method_used,
            "fallback_level": self.fallback_level,
            "used_fallback": self.used_fallback,
            "error_chain": self.error_chain,
            "execution_time": self.execution_time,
            "metadata": self.metadata,
        }


@dataclass
class ChainMethod:
    """A single method in the degradation chain."""
    name: str
    method: Callable[[], T]
    condition: Optional[Callable[[], bool]] = None
    timeout: Optional[float] = None
    catch_exceptions: tuple = (Exception,)
    
    def __str__(self) -> str:
        return f"ChainMethod({self.name})"


class DegradationChain(Generic[T]):
    """
    Executes a chain of methods with automatic fallback.
    
    Methods are tried in order until one succeeds. Failed methods
    are logged and the chain continues. This provides graceful
    degradation for unreliable operations.
    
    Examples:
        >>> chain = DegradationChain()
        >>> chain.add("llm_full", lambda: call_llm(detailed_prompt))
        >>> chain.add("llm_simple", lambda: call_llm(simple_prompt))
        >>> chain.add("pattern", lambda: pattern_match(data))
        >>> chain.add("default", lambda: get_cached_response())
        >>> 
        >>> result = chain.execute()
        >>> print(f"Used: {result.method_used}, fallback: {result.used_fallback}")
        
    With conditions:
        >>> chain = DegradationChain()
        >>> chain.add(
        ...     "gpu_method",
        ...     lambda: gpu_inference(data),
        ...     condition=lambda: gpu_available()
        ... )
        >>> chain.add("cpu_method", lambda: cpu_inference(data))
    """
    
    def __init__(self, name: str = "degradation_chain"):
        """
        Initialize the degradation chain.
        
        Args:
            name: Name for logging and identification
        """
        self.name = name
        self._methods: List[ChainMethod] = []
        self._last_result: Optional[FallbackResult] = None
    
    def add(
        self,
        name: str,
        method: Callable[[], T],
        condition: Optional[Callable[[], bool]] = None,
        timeout: Optional[float] = None,
        catch: tuple = (Exception,),
    ) -> "DegradationChain[T]":
        """
        Add a method to the chain.
        
        Args:
            name: Method identifier for logging
            method: Callable that returns result or raises exception
            condition: Optional check before attempting method
            timeout: Optional timeout in seconds (not implemented)
            catch: Exception types to catch and fallback from
            
        Returns:
            Self for method chaining
            
        Examples:
            >>> chain.add("primary", primary_method)
            >>> chain.add("fallback", fallback_method, condition=lambda: is_available)
        """
        self._methods.append(ChainMethod(
            name=name,
            method=method,
            condition=condition,
            timeout=timeout,
            catch_exceptions=catch,
        ))
        return self  # Allow chaining
    
    def insert(
        self,
        index: int,
        name: str,
        method: Callable[[], T],
        condition: Optional[Callable[[], bool]] = None,
    ) -> "DegradationChain[T]":
        """
        Insert a method at specific position in chain.
        
        Args:
            index: Position to insert (0 = first)
            name: Method identifier
            method: Callable to execute
            condition: Optional condition check
            
        Returns:
            Self for method chaining
        """
        self._methods.insert(index, ChainMethod(
            name=name,
            method=method,
            condition=condition,
        ))
        return self
    
    def execute(
        self,
        default: T = None,
        raise_on_total_failure: bool = False,
    ) -> FallbackResult[T]:
        """
        Execute the chain, returning first successful result.
        
        Args:
            default: Value to return if all methods fail
            raise_on_total_failure: If True, raise exception on total failure
            
        Returns:
            FallbackResult with value and metadata
            
        Raises:
            RuntimeError: If raise_on_total_failure=True and all methods fail
        """
        error_chain = []
        start_time = datetime.now()
        
        for i, chain_method in enumerate(self._methods):
            method_start = datetime.now()
            
            # Check condition if present
            if chain_method.condition is not None:
                try:
                    if not chain_method.condition():
                        logger.debug(
                            f"{self.name}: Skipping {chain_method.name} "
                            f"(condition not met)"
                        )
                        continue
                except Exception as e:
                    logger.debug(
                        f"{self.name}: Condition check failed for "
                        f"{chain_method.name}: {e}"
                    )
                    error_chain.append(
                        f"{chain_method.name} (condition): {type(e).__name__}: {str(e)}"
                    )
                    continue
            
            # Try method
            try:
                result = chain_method.method()
                
                # Check for None results (may indicate soft failure)
                if result is None:
                    logger.debug(
                        f"{self.name}: {chain_method.name} returned None, "
                        f"treating as failure"
                    )
                    error_chain.append(f"{chain_method.name}: returned None")
                    continue
                
                execution_time = (datetime.now() - method_start).total_seconds()
                
                if i > 0:
                    logger.info(
                        f"{self.name}: {chain_method.name} succeeded "
                        f"at fallback level {i}"
                    )
                else:
                    logger.debug(f"{self.name}: {chain_method.name} succeeded")
                
                self._last_result = FallbackResult(
                    value=result,
                    method_used=chain_method.name,
                    fallback_level=i,
                    error_chain=error_chain,
                    execution_time=execution_time,
                )
                return self._last_result
                
            except chain_method.catch_exceptions as e:
                error_msg = f"{chain_method.name}: {type(e).__name__}: {str(e)[:100]}"
                error_chain.append(error_msg)
                logger.warning(f"{self.name}: {error_msg}")
        
        # All methods failed
        total_time = (datetime.now() - start_time).total_seconds()
        
        if raise_on_total_failure:
            raise RuntimeError(
                f"{self.name}: All {len(self._methods)} methods failed. "
                f"Errors: {'; '.join(error_chain)}"
            )
        
        logger.error(
            f"{self.name}: All {len(self._methods)} methods failed, "
            f"returning default"
        )
        
        self._last_result = FallbackResult(
            value=default,
            method_used="default",
            fallback_level=len(self._methods),
            error_chain=error_chain,
            execution_time=total_time,
        )
        return self._last_result
    
    @property
    def last_result(self) -> Optional[FallbackResult[T]]:
        """Get result from last execution."""
        return self._last_result
    
    @property
    def method_count(self) -> int:
        """Get number of methods in chain."""
        return len(self._methods)
    
    def clear(self):
        """Clear all methods from chain."""
        self._methods = []
        self._last_result = None
    
    def __len__(self) -> int:
        return len(self._methods)
    
    def __repr__(self) -> str:
        methods = [m.name for m in self._methods]
        return f"DegradationChain({self.name}, methods={methods})"


def create_llm_degradation_chain(
    full_llm_call: Callable[[], T],
    simple_llm_call: Optional[Callable[[], T]] = None,
    pattern_fallback: Optional[Callable[[], T]] = None,
    default_value: T = None,
    llm_available: Optional[Callable[[], bool]] = None,
) -> DegradationChain[T]:
    """
    Create a standard LLM degradation chain.
    
    Provides a common pattern for LLM-based operations with fallbacks.
    
    Args:
        full_llm_call: Primary LLM call with full context
        simple_llm_call: Simplified LLM call (optional)
        pattern_fallback: Pattern matching fallback (optional)
        default_value: Final fallback value
        llm_available: Check if LLM is available
        
    Returns:
        Configured DegradationChain
        
    Examples:
        >>> chain = create_llm_degradation_chain(
        ...     full_llm_call=lambda: diagnose_with_llm(error, context),
        ...     pattern_fallback=lambda: pattern_diagnose(error),
        ...     default_value=default_diagnosis(),
        ... )
        >>> result = chain.execute()
    """
    chain = DegradationChain[T](name="llm_chain")
    
    # Primary: Full LLM call
    chain.add(
        "llm_full",
        full_llm_call,
        condition=llm_available,
    )
    
    # Secondary: Simplified LLM call
    if simple_llm_call:
        chain.add(
            "llm_simple",
            simple_llm_call,
            condition=llm_available,
        )
    
    # Tertiary: Pattern matching
    if pattern_fallback:
        chain.add(
            "pattern_match",
            pattern_fallback,
        )
    
    # Final: Default value
    chain.add(
        "default",
        lambda: default_value,
    )
    
    return chain


class ConditionalMethod:
    """
    Helper for creating conditional methods.
    
    Wraps a method with a condition check, useful for
    methods that depend on external state.
    
    Examples:
        >>> method = ConditionalMethod(
        ...     gpu_inference,
        ...     condition=lambda: torch.cuda.is_available()
        ... )
        >>> if method.can_execute():
        ...     result = method()
    """
    
    def __init__(
        self,
        method: Callable[..., T],
        condition: Callable[[], bool],
        fallback: Optional[Callable[..., T]] = None,
    ):
        self.method = method
        self.condition = condition
        self.fallback = fallback
    
    def can_execute(self) -> bool:
        """Check if method can execute."""
        try:
            return self.condition()
        except Exception:
            return False
    
    def __call__(self, *args, **kwargs) -> T:
        """Execute method or fallback."""
        if self.can_execute():
            return self.method(*args, **kwargs)
        elif self.fallback:
            return self.fallback(*args, **kwargs)
        else:
            raise RuntimeError("Condition not met and no fallback provided")
