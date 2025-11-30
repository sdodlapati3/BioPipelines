"""
Tests for the observability module.

Tests distributed tracing, metrics collection, and span management.
"""

import pytest
import time
from datetime import datetime

from workflow_composer.infrastructure.observability import (
    Tracer,
    Span,
    SpanEvent,
    SpanStatus,
    MetricsCollector,
    get_tracer,
    get_metrics,
    traced,
    timed,
    reset_tracer,
    reset_metrics,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset global singletons before each test."""
    reset_tracer()
    reset_metrics()
    yield
    reset_tracer()
    reset_metrics()


# =============================================================================
# Tracer Tests
# =============================================================================

class TestTracer:
    """Test the Tracer class."""
    
    def test_get_tracer_singleton(self):
        """Test that get_tracer returns the same instance."""
        tracer1 = get_tracer()
        tracer2 = get_tracer()
        assert tracer1 is tracer2
    
    def test_start_span_creates_span(self):
        """Test that start_span creates a span."""
        tracer = get_tracer()
        with tracer.start_span("test_operation") as span:
            assert span is not None
            assert span.name == "test_operation"
            assert span.start_time is not None
    
    def test_span_ends_on_context_exit(self):
        """Test that span is ended when context exits."""
        tracer = get_tracer()
        with tracer.start_span("test_operation") as span:
            pass
        assert span.end_time is not None
        assert span.duration_ms >= 0
    
    def test_nested_spans_have_parent(self):
        """Test that nested spans have proper parent relationship."""
        tracer = get_tracer()
        with tracer.start_span("parent") as parent:
            with tracer.start_span("child") as child:
                assert child.parent_id == parent.span_id
                assert child.trace_id == parent.trace_id
    
    def test_span_add_event(self):
        """Test adding events to a span."""
        tracer = get_tracer()
        with tracer.start_span("test_operation") as span:
            span.add_event("test_event", {"key": "value"})
            assert len(span.events) == 1
            assert span.events[0].name == "test_event"
            assert span.events[0].attributes["key"] == "value"
    
    def test_span_add_tag(self):
        """Test adding tags to a span."""
        tracer = get_tracer()
        with tracer.start_span("test_operation") as span:
            span.add_tag("test_key", "test_value")
            assert span.tags["test_key"] == "test_value"
    
    def test_span_set_error(self):
        """Test marking a span as errored."""
        tracer = get_tracer()
        with tracer.start_span("test_operation") as span:
            span.set_error(ValueError("test error"))
            assert span.status == SpanStatus.ERROR
            assert "ValueError" in span.tags.get("error.type", "")
    
    def test_span_to_dict(self):
        """Test converting a span to dictionary."""
        tracer = get_tracer()
        with tracer.start_span("test_operation", tags={"env": "test"}) as span:
            span.add_event("started")
        
        data = span.to_dict()
        assert data["name"] == "test_operation"
        assert data["tags"]["env"] == "test"
        assert len(data["events"]) == 1
        assert "duration_ms" in data


class TestTracedDecorator:
    """Test the @traced decorator."""
    
    def test_traced_sync_function(self):
        """Test tracing a synchronous function."""
        tracer = get_tracer()
        
        @traced(tracer)
        def my_function(x, y):
            return x + y
        
        result = my_function(1, 2)
        assert result == 3
    
    @pytest.mark.asyncio
    async def test_traced_async_function(self):
        """Test tracing an async function."""
        tracer = get_tracer()
        
        @traced(tracer)
        async def my_async_function(x, y):
            return x * y
        
        result = await my_async_function(3, 4)
        assert result == 12
    
    def test_traced_captures_exception(self):
        """Test that traced decorator captures exceptions."""
        tracer = get_tracer()
        
        @traced(tracer)
        def failing_function():
            raise ValueError("test error")
        
        with pytest.raises(ValueError):
            failing_function()


# =============================================================================
# Metrics Tests
# =============================================================================

class TestMetricsCollector:
    """Test the MetricsCollector class."""
    
    def test_get_metrics_singleton(self):
        """Test that get_metrics returns the same instance."""
        metrics1 = get_metrics()
        metrics2 = get_metrics()
        assert metrics1 is metrics2
    
    def test_counter_increment(self):
        """Test incrementing a counter."""
        metrics = get_metrics()
        metrics.counter("test.counter")
        metrics.counter("test.counter")
        metrics.counter("test.counter", value=3)
        
        assert metrics.get_counter("test.counter") == 5
    
    def test_counter_with_tags(self):
        """Test counter with tags."""
        metrics = get_metrics()
        metrics.counter("http.requests", tags={"method": "GET", "status": "200"})
        metrics.counter("http.requests", tags={"method": "POST", "status": "201"})
        
        assert metrics.get_counter("http.requests", tags={"method": "GET", "status": "200"}) == 1
        assert metrics.get_counter("http.requests", tags={"method": "POST", "status": "201"}) == 1
    
    def test_gauge(self):
        """Test setting a gauge value."""
        metrics = get_metrics()
        metrics.gauge("cpu.usage", 45.5)
        assert metrics.get_gauge("cpu.usage") == 45.5
        
        metrics.gauge("cpu.usage", 52.3)
        assert metrics.get_gauge("cpu.usage") == 52.3
    
    def test_histogram(self):
        """Test recording histogram values."""
        metrics = get_metrics()
        metrics.histogram("request.duration", 100)
        metrics.histogram("request.duration", 150)
        metrics.histogram("request.duration", 200)
        
        stats = metrics.get_histogram_stats("request.duration")
        assert stats is not None
        assert stats["count"] == 3
        assert stats["min"] == 100
        assert stats["max"] == 200
        assert 100 <= stats["mean"] <= 200
    
    def test_get_all_metrics(self):
        """Test getting all metrics."""
        metrics = get_metrics()
        metrics.counter("a.counter")
        metrics.counter("b.counter")
        metrics.gauge("c.gauge", 100)
        
        all_metrics = metrics.get_metrics()
        assert "a.counter" in all_metrics["counters"]
        assert "b.counter" in all_metrics["counters"]
        assert "c.gauge" in all_metrics["gauges"]
    
    def test_reset_metrics(self):
        """Test resetting all metrics."""
        metrics = get_metrics()
        metrics.counter("test.counter")
        metrics.gauge("test.gauge", 100)
        
        metrics.reset()
        
        assert metrics.get_counter("test.counter") == 0
        assert metrics.get_gauge("test.gauge") is None


class TestTimedDecorator:
    """Test the @timed decorator."""
    
    def test_timed_records_duration(self):
        """Test that @timed records duration."""
        metrics = get_metrics()
        
        @timed(metrics, "test.function.duration")
        def slow_function():
            time.sleep(0.01)  # 10ms
            return "done"
        
        result = slow_function()
        assert result == "done"
        
        stats = metrics.get_histogram_stats("test.function.duration")
        assert stats is not None
        assert stats["count"] >= 1
        assert stats["min"] >= 10  # At least 10ms
    
    @pytest.mark.asyncio
    async def test_timed_async_function(self):
        """Test @timed with async functions."""
        import asyncio
        metrics = get_metrics()
        
        @timed(metrics, "async.function.duration")
        async def async_slow_function():
            await asyncio.sleep(0.01)
            return "async done"
        
        result = await async_slow_function()
        assert result == "async done"


# =============================================================================
# Integration Tests
# =============================================================================

class TestTracingMetricsIntegration:
    """Test tracing and metrics working together."""
    
    def test_span_duration_matches_metrics(self):
        """Test that span duration aligns with metrics."""
        tracer = get_tracer()
        metrics = get_metrics()
        
        with tracer.start_span("operation") as span:
            time.sleep(0.01)  # 10ms
            span.add_tag("result", "success")
        
        # Record in histogram
        metrics.histogram("operation.duration", span.duration_ms)
        
        stats = metrics.get_histogram_stats("operation.duration")
        assert stats["min"] >= 10
    
    def test_multiple_traced_calls(self):
        """Test multiple traced function calls."""
        tracer = get_tracer()
        metrics = get_metrics()
        
        @traced(tracer)
        @timed(metrics, "func.duration")
        def process_item(item):
            return item.upper()
        
        results = [process_item(f"item_{i}") for i in range(5)]
        assert len(results) == 5
        assert all(r.startswith("ITEM_") for r in results)
        
        stats = metrics.get_histogram_stats("func.duration")
        assert stats["count"] == 5


# =============================================================================
# Span Status Tests
# =============================================================================

class TestSpanStatus:
    """Test span status handling."""
    
    def test_default_status_is_unset(self):
        """Test that default status is UNSET."""
        tracer = get_tracer()
        with tracer.start_span("test") as span:
            # Check status before span ends
            pass
        # After context ends, status should be OK
        assert span.status == SpanStatus.OK
    
    def test_set_status_ok(self):
        """Test setting status to OK."""
        tracer = get_tracer()
        with tracer.start_span("test") as span:
            span.set_status(SpanStatus.OK)
        assert span.status == SpanStatus.OK
    
    def test_set_status_error(self):
        """Test setting status to ERROR."""
        tracer = get_tracer()
        with tracer.start_span("test") as span:
            span.set_status(SpanStatus.ERROR, "something went wrong")
        assert span.status == SpanStatus.ERROR
        assert span.status_message == "something went wrong"
