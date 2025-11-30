"""
Tests for Resilience Patterns
=============================

Tests for circuit breaker, retry with backoff, and related patterns.
"""

import asyncio
import pytest
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock

from workflow_composer.infrastructure.resilience import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerError,
    CircuitState,
    BackoffConfig,
    calculate_backoff_delay,
    retry_with_backoff,
    circuit_protected,
    resilient_call,
    get_circuit_breaker,
    reset_all_circuits,
    get_all_circuit_metrics,
)


# =============================================================================
# Circuit Breaker Tests
# =============================================================================

class TestCircuitBreaker:
    """Tests for CircuitBreaker class."""
    
    def test_initial_state_is_closed(self):
        """Circuit should start in closed state."""
        breaker = CircuitBreaker("test")
        assert breaker.state == CircuitState.CLOSED
        assert breaker.is_closed
        assert not breaker.is_open
    
    def test_can_execute_when_closed(self):
        """Should allow execution when closed."""
        breaker = CircuitBreaker("test")
        assert breaker.can_execute()
    
    def test_opens_after_failure_threshold(self):
        """Circuit should open after reaching failure threshold."""
        config = CircuitBreakerConfig(failure_threshold=3)
        breaker = CircuitBreaker("test", config)
        
        # Record failures
        breaker.record_failure()
        breaker.record_failure()
        assert breaker.is_closed
        
        breaker.record_failure()  # Third failure
        assert breaker.is_open
    
    def test_rejects_when_open(self):
        """Should reject calls when open."""
        config = CircuitBreakerConfig(failure_threshold=1, timeout=60.0)
        breaker = CircuitBreaker("test", config)
        
        breaker.record_failure()
        assert breaker.is_open
        assert not breaker.can_execute()
    
    def test_transitions_to_half_open_after_timeout(self):
        """Should transition to half-open after timeout."""
        config = CircuitBreakerConfig(
            failure_threshold=1,
            timeout=0.1,  # 100ms timeout
        )
        breaker = CircuitBreaker("test", config)
        
        breaker.record_failure()
        assert breaker.is_open
        
        # Wait for timeout
        time.sleep(0.15)
        
        # Check state - should be half-open now
        assert breaker.state == CircuitState.HALF_OPEN
        assert breaker.is_half_open
    
    def test_half_open_allows_limited_calls(self):
        """Half-open should allow limited test calls."""
        config = CircuitBreakerConfig(
            failure_threshold=1,
            timeout=0.01,
            half_open_max_calls=1,
        )
        breaker = CircuitBreaker("test", config)
        
        breaker.record_failure()
        time.sleep(0.02)  # Wait for transition
        
        # First call should be allowed
        assert breaker.can_execute()
        
        # Second call should be rejected
        assert not breaker.can_execute()
    
    def test_closes_after_success_threshold(self):
        """Should close after reaching success threshold in half-open."""
        config = CircuitBreakerConfig(
            failure_threshold=1,
            timeout=0.01,
            success_threshold=2,
            half_open_max_calls=3,
        )
        breaker = CircuitBreaker("test", config)
        
        breaker.record_failure()
        time.sleep(0.02)
        
        # Record successes
        breaker.can_execute()  # Consume a half-open call
        breaker.record_success()
        assert breaker.is_half_open
        
        breaker.can_execute()
        breaker.record_success()
        assert breaker.is_closed
    
    def test_reopens_on_half_open_failure(self):
        """Should reopen if failure occurs in half-open state."""
        config = CircuitBreakerConfig(
            failure_threshold=1,
            timeout=0.01,
        )
        breaker = CircuitBreaker("test", config)
        
        breaker.record_failure()
        time.sleep(0.02)
        assert breaker.is_half_open
        
        breaker.record_failure()
        assert breaker.is_open
    
    def test_success_resets_failure_count(self):
        """Success should reset the failure count."""
        config = CircuitBreakerConfig(failure_threshold=3)
        breaker = CircuitBreaker("test", config)
        
        breaker.record_failure()
        breaker.record_failure()
        breaker.record_success()  # Reset
        
        breaker.record_failure()
        breaker.record_failure()
        assert breaker.is_closed  # Should still be closed
    
    def test_manual_reset(self):
        """Manual reset should close the circuit."""
        config = CircuitBreakerConfig(failure_threshold=1)
        breaker = CircuitBreaker("test", config)
        
        breaker.record_failure()
        assert breaker.is_open
        
        breaker.reset()
        assert breaker.is_closed
    
    def test_excluded_exceptions_not_counted(self):
        """Excluded exceptions should not count as failures."""
        config = CircuitBreakerConfig(
            failure_threshold=2,
            excluded_exceptions=(ValueError,),
        )
        breaker = CircuitBreaker("test", config)
        
        breaker.record_failure(ValueError("ignored"))
        breaker.record_failure(ValueError("also ignored"))
        breaker.record_failure(ValueError("still ignored"))
        
        assert breaker.is_closed
        
        # But other exceptions count
        breaker.record_failure(RuntimeError("counts"))
        breaker.record_failure(RuntimeError("also counts"))
        assert breaker.is_open
    
    def test_rolling_window_failures(self):
        """Failures outside the window should not count."""
        config = CircuitBreakerConfig(
            failure_threshold=3,
            failure_window=0.1,  # 100ms window
        )
        breaker = CircuitBreaker("test", config)
        
        breaker.record_failure()
        breaker.record_failure()
        time.sleep(0.15)  # Wait for window to expire
        
        breaker.record_failure()  # Only this should count
        assert breaker.is_closed  # Still closed (only 1 in window)
    
    def test_metrics(self):
        """Metrics should track calls correctly."""
        breaker = CircuitBreaker("test")
        
        breaker.record_success()
        breaker.record_success()
        breaker.record_failure()
        
        metrics = breaker.get_metrics()
        
        assert metrics["name"] == "test"
        assert metrics["total_calls"] == 3
        assert metrics["total_successes"] == 2
        assert metrics["total_failures"] == 1


# =============================================================================
# Backoff Tests
# =============================================================================

class TestBackoff:
    """Tests for retry with backoff."""
    
    def test_calculate_backoff_delay(self):
        """Should calculate exponential delays."""
        config = BackoffConfig(
            base_delay=1.0,
            max_delay=60.0,
            exponential_base=2.0,
            jitter=False,
        )
        
        assert calculate_backoff_delay(0, config) == 1.0
        assert calculate_backoff_delay(1, config) == 2.0
        assert calculate_backoff_delay(2, config) == 4.0
        assert calculate_backoff_delay(3, config) == 8.0
    
    def test_max_delay_cap(self):
        """Delay should not exceed max_delay."""
        config = BackoffConfig(
            base_delay=1.0,
            max_delay=10.0,
            exponential_base=2.0,
            jitter=False,
        )
        
        assert calculate_backoff_delay(10, config) == 10.0  # Capped
    
    def test_jitter_adds_randomness(self):
        """Jitter should add some randomness."""
        config = BackoffConfig(
            base_delay=1.0,
            jitter=True,
            jitter_range=(0.0, 1.0),
        )
        
        delays = [calculate_backoff_delay(0, config) for _ in range(10)]
        
        # All should be between 1.0 and 2.0
        assert all(1.0 <= d <= 2.0 for d in delays)
        
        # Should have some variance (not all the same)
        assert len(set(delays)) > 1


# =============================================================================
# Retry Decorator Tests
# =============================================================================

class TestRetryDecorator:
    """Tests for retry_with_backoff decorator."""
    
    def test_sync_success_no_retry(self):
        """Successful sync call should not retry."""
        call_count = 0
        
        @retry_with_backoff(max_retries=3)
        def func():
            nonlocal call_count
            call_count += 1
            return "success"
        
        result = func()
        assert result == "success"
        assert call_count == 1
    
    def test_sync_retries_on_failure(self):
        """Should retry on failure."""
        call_count = 0
        
        @retry_with_backoff(max_retries=3, base_delay=0.01)
        def func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("fail")
            return "success"
        
        result = func()
        assert result == "success"
        assert call_count == 3
    
    def test_sync_raises_after_max_retries(self):
        """Should raise after exhausting retries."""
        call_count = 0
        
        @retry_with_backoff(max_retries=2, base_delay=0.01)
        def func():
            nonlocal call_count
            call_count += 1
            raise ValueError("always fail")
        
        with pytest.raises(ValueError, match="always fail"):
            func()
        
        assert call_count == 3  # Initial + 2 retries
    
    def test_non_retryable_exceptions(self):
        """Non-retryable exceptions should not trigger retry."""
        call_count = 0
        
        @retry_with_backoff(
            max_retries=3,
            retryable_exceptions=(ConnectionError,),
            non_retryable_exceptions=(ValueError,),
        )
        def func():
            nonlocal call_count
            call_count += 1
            raise ValueError("no retry")
        
        with pytest.raises(ValueError):
            func()
        
        assert call_count == 1  # No retries
    
    @pytest.mark.asyncio
    async def test_async_success(self):
        """Async function should work without retry on success."""
        call_count = 0
        
        @retry_with_backoff(max_retries=3)
        async def func():
            nonlocal call_count
            call_count += 1
            return "success"
        
        result = await func()
        assert result == "success"
        assert call_count == 1
    
    @pytest.mark.asyncio
    async def test_async_retries(self):
        """Async function should retry on failure."""
        call_count = 0
        
        @retry_with_backoff(max_retries=3, base_delay=0.01)
        async def func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionError("retry me")
            return "success"
        
        result = await func()
        assert result == "success"
        assert call_count == 2


# =============================================================================
# Circuit Protected Decorator Tests
# =============================================================================

class TestCircuitProtectedDecorator:
    """Tests for circuit_protected decorator."""
    
    def test_allows_call_when_closed(self):
        """Should allow calls when circuit is closed."""
        breaker = CircuitBreaker("test_decorator")
        
        @circuit_protected(breaker)
        def func():
            return "success"
        
        result = func()
        assert result == "success"
    
    def test_records_success(self):
        """Should record success on successful call."""
        breaker = CircuitBreaker("test_success")
        
        @circuit_protected(breaker)
        def func():
            return "success"
        
        func()
        metrics = breaker.get_metrics()
        assert metrics["total_successes"] == 1
    
    def test_records_failure(self):
        """Should record failure on exception."""
        config = CircuitBreakerConfig(failure_threshold=5)
        breaker = CircuitBreaker("test_failure", config)
        
        @circuit_protected(breaker)
        def func():
            raise ValueError("fail")
        
        with pytest.raises(ValueError):
            func()
        
        metrics = breaker.get_metrics()
        assert metrics["total_failures"] == 1
    
    def test_raises_circuit_breaker_error_when_open(self):
        """Should raise CircuitBreakerError when circuit is open."""
        config = CircuitBreakerConfig(failure_threshold=1, timeout=60.0)
        breaker = CircuitBreaker("test_open", config)
        breaker.record_failure()  # Open the circuit
        
        @circuit_protected(breaker)
        def func():
            return "should not reach"
        
        with pytest.raises(CircuitBreakerError) as exc_info:
            func()
        
        assert exc_info.value.name == "test_open"
        assert exc_info.value.state == CircuitState.OPEN
    
    @pytest.mark.asyncio
    async def test_async_circuit_protected(self):
        """Should work with async functions."""
        breaker = CircuitBreaker("test_async")
        
        @circuit_protected(breaker)
        async def func():
            return "async success"
        
        result = await func()
        assert result == "async success"


# =============================================================================
# Resilient Call Tests
# =============================================================================

class TestResilientCall:
    """Tests for combined resilient_call decorator."""
    
    def test_successful_call(self):
        """Should work on success."""
        @resilient_call("test_resilient")
        def func():
            return "success"
        
        result = func()
        assert result == "success"
    
    def test_retries_before_circuit_opens(self):
        """Should retry before opening circuit."""
        call_count = 0
        
        @resilient_call(
            "test_retry_resilient",
            max_retries=2,
            base_delay=0.01,
        )
        def func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("retry")
            return "success"
        
        result = func()
        assert result == "success"
        assert call_count == 3


# =============================================================================
# Registry Tests
# =============================================================================

class TestCircuitBreakerRegistry:
    """Tests for global circuit breaker registry."""
    
    def setup_method(self):
        """Reset registry before each test."""
        reset_all_circuits()
    
    def test_get_or_create(self):
        """Should get or create circuit breakers."""
        breaker1 = get_circuit_breaker("registry_test")
        breaker2 = get_circuit_breaker("registry_test")
        
        assert breaker1 is breaker2
    
    def test_reset_all(self):
        """Should reset all circuit breakers."""
        breaker = get_circuit_breaker("reset_test", CircuitBreakerConfig(failure_threshold=1))
        breaker.record_failure()
        assert breaker.is_open
        
        reset_all_circuits()
        assert breaker.is_closed
    
    def test_get_all_metrics(self):
        """Should return metrics for all circuits."""
        get_circuit_breaker("metrics_test_1")
        get_circuit_breaker("metrics_test_2")
        
        metrics = get_all_circuit_metrics()
        
        assert "metrics_test_1" in metrics
        assert "metrics_test_2" in metrics


# =============================================================================
# Thread Safety Tests
# =============================================================================

class TestThreadSafety:
    """Tests for thread safety of circuit breaker."""
    
    def test_concurrent_access(self):
        """Circuit breaker should handle concurrent access."""
        import threading
        
        config = CircuitBreakerConfig(failure_threshold=100)
        breaker = CircuitBreaker("concurrent_test", config)
        
        def record_many():
            for _ in range(50):
                if breaker.can_execute():
                    breaker.record_success()
        
        threads = [threading.Thread(target=record_many) for _ in range(10)]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        metrics = breaker.get_metrics()
        assert metrics["total_successes"] == 500
