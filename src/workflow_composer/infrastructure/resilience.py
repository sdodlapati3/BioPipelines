"""
Resilience Patterns
===================

Production-grade resilience patterns for external API calls:

- **CircuitBreaker**: Prevent cascade failures when services are down
- **RetryWithBackoff**: Exponential backoff with jitter for transient failures
- **Timeout**: Configurable timeouts for all external calls

Usage:
    from workflow_composer.infrastructure.resilience import (
        CircuitBreaker,
        CircuitBreakerConfig,
        retry_with_backoff,
        circuit_protected,
    )
    
    # Create a circuit breaker for ENCODE API
    encode_breaker = CircuitBreaker("encode", CircuitBreakerConfig(
        failure_threshold=5,
        timeout=30.0,
    ))
    
    # Use with decorator
    @circuit_protected(encode_breaker)
    @retry_with_backoff(max_retries=3)
    async def search_encode(query: str):
        ...
    
    # Or manually
    if encode_breaker.can_execute():
        try:
            result = await search_encode(query)
            encode_breaker.record_success()
        except Exception as e:
            encode_breaker.record_failure()
            raise

Circuit Breaker States:
    CLOSED  → Normal operation, requests pass through
    OPEN    → Service failing, requests rejected immediately  
    HALF_OPEN → Testing recovery, limited requests allowed

Transitions:
    CLOSED → OPEN: When failure_threshold reached
    OPEN → HALF_OPEN: After timeout expires
    HALF_OPEN → CLOSED: When success_threshold reached
    HALF_OPEN → OPEN: On any failure
"""

import asyncio
import functools
import logging
import random
import time
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])


# =============================================================================
# Circuit Breaker
# =============================================================================

class CircuitState(Enum):
    """State of the circuit breaker."""
    CLOSED = "closed"        # Normal operation
    OPEN = "open"            # Failing, reject calls
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreakerConfig:
    """Configuration for a circuit breaker."""
    
    failure_threshold: int = 5
    """Number of failures before opening the circuit."""
    
    success_threshold: int = 2
    """Successes needed to close from half-open state."""
    
    timeout: float = 30.0
    """Seconds to stay in open state before testing recovery."""
    
    half_open_max_calls: int = 1
    """Maximum test calls allowed in half-open state."""
    
    failure_window: float = 60.0
    """Window in seconds for counting failures (rolling window)."""
    
    excluded_exceptions: Tuple[Type[Exception], ...] = ()
    """Exceptions that should NOT count as failures."""


class CircuitBreakerError(Exception):
    """Raised when circuit breaker is open."""
    
    def __init__(self, name: str, state: CircuitState, retry_after: float = None):
        self.name = name
        self.state = state
        self.retry_after = retry_after
        message = f"Circuit breaker '{name}' is {state.value}"
        if retry_after:
            message += f", retry after {retry_after:.1f}s"
        super().__init__(message)


class CircuitBreaker:
    """
    Circuit breaker pattern implementation.
    
    Prevents cascade failures by fast-failing requests when a service is down.
    
    Example:
        breaker = CircuitBreaker("encode_api")
        
        for attempt in range(10):
            if breaker.can_execute():
                try:
                    result = call_encode_api()
                    breaker.record_success()
                    return result
                except Exception as e:
                    breaker.record_failure(e)
                    if not breaker.can_execute():
                        raise CircuitBreakerError(...)
            else:
                raise CircuitBreakerError(...)
    
    Thread-safe implementation using locks.
    """
    
    def __init__(
        self,
        name: str,
        config: CircuitBreakerConfig = None,
    ):
        """
        Initialize circuit breaker.
        
        Args:
            name: Identifier for this circuit (e.g., "encode", "geo")
            config: Configuration options
        """
        self.name = name
        self.config = config or CircuitBreakerConfig()
        
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._failure_times: list = []
        self._last_failure_time: Optional[datetime] = None
        self._last_state_change: datetime = datetime.now()
        self._half_open_calls = 0
        
        self._lock = threading.RLock()
        
        # Metrics
        self._total_calls = 0
        self._total_failures = 0
        self._total_successes = 0
        self._total_rejections = 0
    
    @property
    def state(self) -> CircuitState:
        """Current state of the circuit."""
        with self._lock:
            self._check_state_transition()
            return self._state
    
    @property
    def is_closed(self) -> bool:
        """Check if circuit is closed (normal operation)."""
        return self.state == CircuitState.CLOSED
    
    @property
    def is_open(self) -> bool:
        """Check if circuit is open (failing)."""
        return self.state == CircuitState.OPEN
    
    @property
    def is_half_open(self) -> bool:
        """Check if circuit is half-open (testing recovery)."""
        return self.state == CircuitState.HALF_OPEN
    
    def can_execute(self) -> bool:
        """
        Check if a request can be executed.
        
        Returns:
            True if request should proceed, False if it should be rejected.
        """
        with self._lock:
            self._check_state_transition()
            
            if self._state == CircuitState.CLOSED:
                return True
            
            elif self._state == CircuitState.OPEN:
                return False
            
            else:  # HALF_OPEN
                if self._half_open_calls < self.config.half_open_max_calls:
                    self._half_open_calls += 1
                    return True
                return False
    
    def record_success(self):
        """Record a successful call."""
        with self._lock:
            self._total_calls += 1
            self._total_successes += 1
            
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                logger.debug(
                    f"Circuit '{self.name}' half-open success "
                    f"({self._success_count}/{self.config.success_threshold})"
                )
                
                if self._success_count >= self.config.success_threshold:
                    self._transition_to_closed()
            else:
                # Reset failure count on success in closed state
                self._failure_count = 0
                self._failure_times.clear()
    
    def record_failure(self, exception: Exception = None):
        """
        Record a failed call.
        
        Args:
            exception: The exception that caused the failure (optional).
                       Used to check against excluded_exceptions.
        """
        # Check if this exception should be excluded
        if exception and isinstance(exception, self.config.excluded_exceptions):
            logger.debug(
                f"Circuit '{self.name}' ignoring excluded exception: "
                f"{type(exception).__name__}"
            )
            return
        
        with self._lock:
            self._total_calls += 1
            self._total_failures += 1
            
            now = datetime.now()
            self._last_failure_time = now
            
            if self._state == CircuitState.HALF_OPEN:
                logger.warning(
                    f"Circuit '{self.name}' half-open test failed, opening"
                )
                self._transition_to_open()
            
            else:  # CLOSED
                # Add to rolling window
                self._failure_times.append(now)
                
                # Remove old failures outside window
                window_start = now - timedelta(seconds=self.config.failure_window)
                self._failure_times = [
                    t for t in self._failure_times if t > window_start
                ]
                
                self._failure_count = len(self._failure_times)
                
                logger.debug(
                    f"Circuit '{self.name}' failure count: "
                    f"{self._failure_count}/{self.config.failure_threshold}"
                )
                
                if self._failure_count >= self.config.failure_threshold:
                    self._transition_to_open()
    
    def reset(self):
        """Manually reset the circuit to closed state."""
        with self._lock:
            self._transition_to_closed()
            logger.info(f"Circuit '{self.name}' manually reset")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get circuit breaker metrics."""
        with self._lock:
            return {
                "name": self.name,
                "state": self._state.value,
                "total_calls": self._total_calls,
                "total_successes": self._total_successes,
                "total_failures": self._total_failures,
                "total_rejections": self._total_rejections,
                "current_failure_count": self._failure_count,
                "last_failure": self._last_failure_time.isoformat() if self._last_failure_time else None,
                "last_state_change": self._last_state_change.isoformat(),
            }
    
    def _check_state_transition(self):
        """Check if state should transition (called under lock)."""
        if self._state == CircuitState.OPEN:
            elapsed = (datetime.now() - self._last_state_change).total_seconds()
            if elapsed >= self.config.timeout:
                self._transition_to_half_open()
    
    def _transition_to_open(self):
        """Transition to OPEN state."""
        self._state = CircuitState.OPEN
        self._last_state_change = datetime.now()
        self._half_open_calls = 0
        self._success_count = 0
        
        logger.warning(
            f"Circuit '{self.name}' OPENED after {self._failure_count} failures"
        )
    
    def _transition_to_half_open(self):
        """Transition to HALF_OPEN state."""
        self._state = CircuitState.HALF_OPEN
        self._last_state_change = datetime.now()
        self._half_open_calls = 0
        self._success_count = 0
        
        logger.info(
            f"Circuit '{self.name}' entering HALF_OPEN state (testing recovery)"
        )
    
    def _transition_to_closed(self):
        """Transition to CLOSED state."""
        self._state = CircuitState.CLOSED
        self._last_state_change = datetime.now()
        self._failure_count = 0
        self._failure_times.clear()
        self._half_open_calls = 0
        self._success_count = 0
        
        logger.info(f"Circuit '{self.name}' CLOSED (recovered)")
    
    def __repr__(self) -> str:
        return f"CircuitBreaker(name='{self.name}', state={self._state.value})"


# =============================================================================
# Global Circuit Breaker Registry
# =============================================================================

_circuit_breakers: Dict[str, CircuitBreaker] = {}
_registry_lock = threading.Lock()


def get_circuit_breaker(
    name: str,
    config: CircuitBreakerConfig = None,
) -> CircuitBreaker:
    """
    Get or create a circuit breaker by name.
    
    Args:
        name: Unique identifier for the circuit
        config: Configuration (only used on first creation)
    
    Returns:
        The circuit breaker instance
    """
    with _registry_lock:
        if name not in _circuit_breakers:
            _circuit_breakers[name] = CircuitBreaker(name, config)
        return _circuit_breakers[name]


def reset_all_circuits():
    """Reset all circuit breakers to closed state."""
    with _registry_lock:
        for breaker in _circuit_breakers.values():
            breaker.reset()


def get_all_circuit_metrics() -> Dict[str, Dict[str, Any]]:
    """Get metrics for all circuit breakers."""
    with _registry_lock:
        return {
            name: breaker.get_metrics()
            for name, breaker in _circuit_breakers.items()
        }


# =============================================================================
# Retry with Backoff
# =============================================================================

@dataclass
class BackoffConfig:
    """Configuration for retry with backoff."""
    
    max_retries: int = 3
    """Maximum number of retry attempts."""
    
    base_delay: float = 1.0
    """Initial delay in seconds."""
    
    max_delay: float = 60.0
    """Maximum delay in seconds."""
    
    exponential_base: float = 2.0
    """Base for exponential calculation."""
    
    jitter: bool = True
    """Whether to add random jitter to delay."""
    
    jitter_range: Tuple[float, float] = (0.0, 1.0)
    """Range for jitter in seconds."""
    
    retryable_exceptions: Tuple[Type[Exception], ...] = (Exception,)
    """Exceptions that should trigger a retry."""
    
    non_retryable_exceptions: Tuple[Type[Exception], ...] = ()
    """Exceptions that should NOT trigger a retry."""


def calculate_backoff_delay(
    attempt: int,
    config: BackoffConfig,
) -> float:
    """
    Calculate delay for a given retry attempt.
    
    Args:
        attempt: Zero-based attempt number
        config: Backoff configuration
    
    Returns:
        Delay in seconds
    """
    delay = config.base_delay * (config.exponential_base ** attempt)
    delay = min(delay, config.max_delay)
    
    if config.jitter:
        jitter = random.uniform(*config.jitter_range)
        delay += jitter
    
    return delay


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    retryable_exceptions: Tuple[Type[Exception], ...] = (Exception,),
    non_retryable_exceptions: Tuple[Type[Exception], ...] = (),
    on_retry: Callable[[int, Exception, float], None] = None,
) -> Callable[[F], F]:
    """
    Decorator for retry with exponential backoff.
    
    Works with both sync and async functions.
    
    Args:
        max_retries: Maximum retry attempts
        base_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
        exponential_base: Base for exponential calculation
        jitter: Whether to add random jitter
        retryable_exceptions: Exceptions that trigger retry
        non_retryable_exceptions: Exceptions that should NOT retry
        on_retry: Callback called before each retry (attempt, exception, delay)
    
    Example:
        @retry_with_backoff(max_retries=3, base_delay=1.0)
        async def fetch_data():
            ...
        
        @retry_with_backoff(
            max_retries=5,
            retryable_exceptions=(ConnectionError, TimeoutError),
        )
        def sync_fetch():
            ...
    """
    config = BackoffConfig(
        max_retries=max_retries,
        base_delay=base_delay,
        max_delay=max_delay,
        exponential_base=exponential_base,
        jitter=jitter,
        retryable_exceptions=retryable_exceptions,
        non_retryable_exceptions=non_retryable_exceptions,
    )
    
    def decorator(func: F) -> F:
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                last_exception = None
                
                for attempt in range(max_retries + 1):
                    try:
                        return await func(*args, **kwargs)
                    except non_retryable_exceptions as e:
                        # Don't retry these
                        raise
                    except retryable_exceptions as e:
                        last_exception = e
                        
                        if attempt == max_retries:
                            logger.warning(
                                f"Retry exhausted for {func.__name__} after "
                                f"{max_retries + 1} attempts: {e}"
                            )
                            raise
                        
                        delay = calculate_backoff_delay(attempt, config)
                        
                        logger.debug(
                            f"Retry {attempt + 1}/{max_retries} for "
                            f"{func.__name__} after {delay:.2f}s: {e}"
                        )
                        
                        if on_retry:
                            on_retry(attempt, e, delay)
                        
                        await asyncio.sleep(delay)
                
                raise last_exception
            
            return async_wrapper  # type: ignore
        
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                last_exception = None
                
                for attempt in range(max_retries + 1):
                    try:
                        return func(*args, **kwargs)
                    except non_retryable_exceptions as e:
                        raise
                    except retryable_exceptions as e:
                        last_exception = e
                        
                        if attempt == max_retries:
                            logger.warning(
                                f"Retry exhausted for {func.__name__} after "
                                f"{max_retries + 1} attempts: {e}"
                            )
                            raise
                        
                        delay = calculate_backoff_delay(attempt, config)
                        
                        logger.debug(
                            f"Retry {attempt + 1}/{max_retries} for "
                            f"{func.__name__} after {delay:.2f}s: {e}"
                        )
                        
                        if on_retry:
                            on_retry(attempt, e, delay)
                        
                        time.sleep(delay)
                
                raise last_exception
            
            return sync_wrapper  # type: ignore
    
    return decorator


# =============================================================================
# Circuit Breaker Decorator
# =============================================================================

def circuit_protected(
    circuit: Union[str, CircuitBreaker],
    config: CircuitBreakerConfig = None,
) -> Callable[[F], F]:
    """
    Decorator to protect a function with a circuit breaker.
    
    Args:
        circuit: Circuit breaker instance or name
        config: Configuration (if passing name)
    
    Example:
        @circuit_protected("encode_api")
        async def search_encode(query: str):
            ...
        
        # Or with explicit breaker
        breaker = CircuitBreaker("geo_api")
        
        @circuit_protected(breaker)
        def search_geo(query: str):
            ...
    """
    def decorator(func: F) -> F:
        # Get or create the circuit breaker
        nonlocal circuit
        if isinstance(circuit, str):
            circuit = get_circuit_breaker(circuit, config)
        
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                if not circuit.can_execute():
                    raise CircuitBreakerError(
                        circuit.name,
                        circuit.state,
                        retry_after=circuit.config.timeout,
                    )
                
                try:
                    result = await func(*args, **kwargs)
                    circuit.record_success()
                    return result
                except Exception as e:
                    circuit.record_failure(e)
                    raise
            
            return async_wrapper  # type: ignore
        
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                if not circuit.can_execute():
                    raise CircuitBreakerError(
                        circuit.name,
                        circuit.state,
                        retry_after=circuit.config.timeout,
                    )
                
                try:
                    result = func(*args, **kwargs)
                    circuit.record_success()
                    return result
                except Exception as e:
                    circuit.record_failure(e)
                    raise
            
            return sync_wrapper  # type: ignore
    
    return decorator


# =============================================================================
# Combined Resilient Call
# =============================================================================

def resilient_call(
    circuit_name: str,
    circuit_config: CircuitBreakerConfig = None,
    max_retries: int = 3,
    base_delay: float = 1.0,
    retryable_exceptions: Tuple[Type[Exception], ...] = (Exception,),
) -> Callable[[F], F]:
    """
    Combined decorator: circuit breaker + retry with backoff.
    
    This is the recommended decorator for external API calls.
    
    Args:
        circuit_name: Name for the circuit breaker
        circuit_config: Circuit breaker configuration
        max_retries: Maximum retry attempts
        base_delay: Initial retry delay
        retryable_exceptions: Exceptions to retry
    
    Example:
        @resilient_call(
            "encode_api",
            max_retries=3,
            retryable_exceptions=(requests.RequestException,),
        )
        async def search_encode(query: str):
            response = await session.get(f"{BASE_URL}/search/?q={query}")
            return response.json()
    """
    def decorator(func: F) -> F:
        # Apply retry first (inner), then circuit breaker (outer)
        wrapped = retry_with_backoff(
            max_retries=max_retries,
            base_delay=base_delay,
            retryable_exceptions=retryable_exceptions,
            # Don't retry circuit breaker errors
            non_retryable_exceptions=(CircuitBreakerError,),
        )(func)
        
        wrapped = circuit_protected(
            circuit_name,
            circuit_config,
        )(wrapped)
        
        return wrapped  # type: ignore
    
    return decorator


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Circuit Breaker
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitBreakerError",
    "CircuitState",
    "get_circuit_breaker",
    "reset_all_circuits",
    "get_all_circuit_metrics",
    # Retry
    "BackoffConfig",
    "retry_with_backoff",
    "calculate_backoff_delay",
    # Decorators
    "circuit_protected",
    "resilient_call",
]
