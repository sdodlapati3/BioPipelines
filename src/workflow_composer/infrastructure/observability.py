"""
Observability Module
====================

Lightweight observability for BioPipelines:

- **Tracer**: Distributed tracing with spans
- **Metrics**: Counters, gauges, histograms
- **Structured Logging**: JSON logs with correlation IDs

Design Goals:
- Zero external dependencies (OpenTelemetry optional)
- Works in HPC environments (file-based export)
- Minimal performance overhead
- Context propagation via contextvars

Usage:
    from workflow_composer.infrastructure.observability import (
        get_tracer,
        traced,
        metrics,
    )
    
    tracer = get_tracer()
    
    # Manual span
    with tracer.start_span("search_databases") as span:
        span.add_tag("source", "encode")
        result = do_search()
        span.add_tag("result_count", len(result))
    
    # Decorator
    @traced("fetch_dataset")
    async def fetch_dataset(dataset_id: str):
        ...
    
    # Metrics
    metrics.counter("api_calls", tags={"source": "encode"})
    metrics.histogram("response_time_ms", 150.0)

Span Structure:
    - trace_id: Unique ID for the entire request
    - span_id: Unique ID for this operation
    - parent_id: ID of parent span (if any)
    - name: Operation name
    - start_time: When operation started
    - end_time: When operation ended
    - tags: Key-value metadata
    - events: Timestamped log entries
    - status: ok | error
"""

import asyncio
import contextvars
import functools
import json
import logging
import os
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    TypeVar,
    Union,
)

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


# =============================================================================
# Context Propagation
# =============================================================================

# Current trace context
_current_trace_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "current_trace_id", default=None
)
_current_span: contextvars.ContextVar[Optional["Span"]] = contextvars.ContextVar(
    "current_span", default=None
)


def get_current_trace_id() -> Optional[str]:
    """Get the current trace ID from context."""
    return _current_trace_id.get()


def get_current_span() -> Optional["Span"]:
    """Get the current span from context."""
    return _current_span.get()


# =============================================================================
# Span
# =============================================================================

class SpanStatus(Enum):
    """Status of a span."""
    OK = "ok"
    ERROR = "error"
    UNSET = "unset"


@dataclass
class SpanEvent:
    """An event within a span."""
    name: str
    timestamp: datetime
    attributes: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "timestamp": self.timestamp.isoformat(),
            "attributes": self.attributes,
        }


@dataclass
class Span:
    """
    A span represents a single operation in a trace.
    
    Spans form a tree structure where each span can have a parent.
    The root span has no parent and defines the trace_id.
    """
    
    name: str
    trace_id: str
    span_id: str = field(default_factory=lambda: str(uuid.uuid4())[:16])
    parent_id: Optional[str] = None
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    
    status: SpanStatus = SpanStatus.UNSET
    status_message: str = ""
    
    tags: Dict[str, Any] = field(default_factory=dict)
    events: List[SpanEvent] = field(default_factory=list)
    
    # Internal state
    _ended: bool = field(default=False, repr=False)
    _token: contextvars.Token = field(default=None, repr=False)
    
    def add_tag(self, key: str, value: Any) -> "Span":
        """Add a tag to the span."""
        self.tags[key] = value
        return self
    
    def add_tags(self, tags: Dict[str, Any]) -> "Span":
        """Add multiple tags to the span."""
        self.tags.update(tags)
        return self
    
    def add_event(
        self,
        name: str,
        attributes: Dict[str, Any] = None,
    ) -> "Span":
        """Add an event to the span."""
        self.events.append(SpanEvent(
            name=name,
            timestamp=datetime.now(),
            attributes=attributes or {},
        ))
        return self
    
    def set_status(
        self,
        status: SpanStatus,
        message: str = "",
    ) -> "Span":
        """Set the span status."""
        self.status = status
        self.status_message = message
        return self
    
    def set_error(self, error: Exception) -> "Span":
        """Mark the span as errored."""
        self.status = SpanStatus.ERROR
        self.status_message = str(error)
        self.add_tag("error.type", type(error).__name__)
        self.add_tag("error.message", str(error))
        return self
    
    def end(self) -> None:
        """End the span."""
        if not self._ended:
            self.end_time = datetime.now()
            self._ended = True
            
            if self.status == SpanStatus.UNSET:
                self.status = SpanStatus.OK
    
    @property
    def duration_ms(self) -> float:
        """Duration in milliseconds."""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds() * 1000
        return 0.0
    
    @property
    def is_ended(self) -> bool:
        """Check if span has ended."""
        return self._ended
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert span to dictionary for export."""
        return {
            "name": self.name,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_id": self.parent_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_ms": self.duration_ms,
            "status": self.status.value,
            "status_message": self.status_message,
            "tags": self.tags,
            "events": [e.to_dict() for e in self.events],
        }
    
    def __enter__(self) -> "Span":
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if exc_val:
            self.set_error(exc_val)
        self.end()


# =============================================================================
# Tracer
# =============================================================================

class SpanExporter:
    """Base class for span exporters."""
    
    def export(self, span: Span) -> None:
        """Export a span."""
        raise NotImplementedError
    
    def flush(self) -> None:
        """Flush any buffered spans."""
        pass
    
    def shutdown(self) -> None:
        """Shutdown the exporter."""
        pass


class ConsoleExporter(SpanExporter):
    """Export spans to console (for debugging)."""
    
    def __init__(self, min_duration_ms: float = 0.0):
        self.min_duration_ms = min_duration_ms
    
    def export(self, span: Span) -> None:
        if span.duration_ms >= self.min_duration_ms:
            status_icon = "✅" if span.status == SpanStatus.OK else "❌"
            print(
                f"{status_icon} {span.name} "
                f"[{span.duration_ms:.2f}ms] "
                f"trace={span.trace_id[:8]}"
            )


class JSONFileExporter(SpanExporter):
    """Export spans to JSON file."""
    
    def __init__(
        self,
        path: Union[str, Path],
        max_buffer_size: int = 100,
    ):
        self.path = Path(path)
        self.max_buffer_size = max_buffer_size
        self._buffer: List[Dict] = []
        self._lock = threading.Lock()
        
        # Ensure directory exists
        self.path.parent.mkdir(parents=True, exist_ok=True)
    
    def export(self, span: Span) -> None:
        with self._lock:
            self._buffer.append(span.to_dict())
            
            if len(self._buffer) >= self.max_buffer_size:
                self.flush()
    
    def flush(self) -> None:
        with self._lock:
            if not self._buffer:
                return
            
            with open(self.path, "a") as f:
                for span_dict in self._buffer:
                    f.write(json.dumps(span_dict) + "\n")
            
            self._buffer.clear()
    
    def shutdown(self) -> None:
        self.flush()


class InMemoryExporter(SpanExporter):
    """Export spans to memory (for testing)."""
    
    def __init__(self, max_spans: int = 1000):
        self.max_spans = max_spans
        self.spans: List[Span] = []
        self._lock = threading.Lock()
    
    def export(self, span: Span) -> None:
        with self._lock:
            self.spans.append(span)
            if len(self.spans) > self.max_spans:
                self.spans = self.spans[-self.max_spans:]
    
    def clear(self) -> None:
        with self._lock:
            self.spans.clear()
    
    def get_spans(self) -> List[Span]:
        with self._lock:
            return list(self.spans)


class Tracer:
    """
    Tracer for creating and managing spans.
    
    Example:
        tracer = Tracer("biopipelines")
        
        with tracer.start_span("process_query") as span:
            span.add_tag("query", user_query)
            result = process(user_query)
            span.add_tag("tool", result.tool_name)
    """
    
    def __init__(
        self,
        service_name: str = "biopipelines",
        exporters: List[SpanExporter] = None,
    ):
        self.service_name = service_name
        self.exporters = exporters or []
        self._lock = threading.Lock()
    
    def add_exporter(self, exporter: SpanExporter) -> None:
        """Add a span exporter."""
        with self._lock:
            self.exporters.append(exporter)
    
    @contextmanager
    def start_span(
        self,
        name: str,
        parent: Span = None,
        tags: Dict[str, Any] = None,
    ) -> Iterator[Span]:
        """
        Start a new span as a context manager.
        
        Args:
            name: Span name
            parent: Parent span (uses current span if not provided)
            tags: Initial tags
        
        Yields:
            The new span
        """
        # Get parent from context if not provided
        if parent is None:
            parent = _current_span.get()
        
        # Generate trace ID (use parent's or create new)
        if parent:
            trace_id = parent.trace_id
            parent_id = parent.span_id
        else:
            trace_id = _current_trace_id.get() or str(uuid.uuid4())[:32]
            parent_id = None
        
        # Create span
        span = Span(
            name=name,
            trace_id=trace_id,
            parent_id=parent_id,
            tags=tags or {},
        )
        span.add_tag("service", self.service_name)
        
        # Set context
        trace_token = _current_trace_id.set(trace_id)
        span_token = _current_span.set(span)
        
        try:
            yield span
        except Exception as e:
            span.set_error(e)
            raise
        finally:
            span.end()
            
            # Export span
            for exporter in self.exporters:
                try:
                    exporter.export(span)
                except Exception as e:
                    logger.warning(f"Failed to export span: {e}")
            
            # Restore context
            _current_trace_id.reset(trace_token)
            _current_span.reset(span_token)
    
    def create_span(
        self,
        name: str,
        parent: Span = None,
        tags: Dict[str, Any] = None,
    ) -> Span:
        """
        Create a span without context management.
        
        The caller is responsible for calling span.end().
        
        Args:
            name: Span name
            parent: Parent span
            tags: Initial tags
        
        Returns:
            New span
        """
        if parent is None:
            parent = _current_span.get()
        
        if parent:
            trace_id = parent.trace_id
            parent_id = parent.span_id
        else:
            trace_id = _current_trace_id.get() or str(uuid.uuid4())[:32]
            parent_id = None
        
        span = Span(
            name=name,
            trace_id=trace_id,
            parent_id=parent_id,
            tags=tags or {},
        )
        span.add_tag("service", self.service_name)
        
        return span
    
    def flush(self) -> None:
        """Flush all exporters."""
        for exporter in self.exporters:
            try:
                exporter.flush()
            except Exception as e:
                logger.warning(f"Failed to flush exporter: {e}")
    
    def shutdown(self) -> None:
        """Shutdown all exporters."""
        for exporter in self.exporters:
            try:
                exporter.shutdown()
            except Exception as e:
                logger.warning(f"Failed to shutdown exporter: {e}")


# =============================================================================
# Global Tracer
# =============================================================================

_global_tracer: Optional[Tracer] = None
_tracer_lock = threading.Lock()


def get_tracer(service_name: str = "biopipelines") -> Tracer:
    """
    Get the global tracer instance.
    
    Creates a new tracer if one doesn't exist.
    """
    global _global_tracer
    
    with _tracer_lock:
        if _global_tracer is None:
            _global_tracer = Tracer(service_name)
            
            # Add default exporter based on environment
            if os.environ.get("BIOPIPELINES_TRACE_FILE"):
                _global_tracer.add_exporter(
                    JSONFileExporter(os.environ["BIOPIPELINES_TRACE_FILE"])
                )
            
            if os.environ.get("BIOPIPELINES_TRACE_CONSOLE"):
                _global_tracer.add_exporter(ConsoleExporter())
        
        return _global_tracer


def set_tracer(tracer: Tracer) -> None:
    """Set the global tracer instance."""
    global _global_tracer
    with _tracer_lock:
        _global_tracer = tracer


def reset_tracer() -> None:
    """Reset the global tracer (for testing)."""
    global _global_tracer
    with _tracer_lock:
        if _global_tracer:
            _global_tracer.shutdown()
        _global_tracer = None


# =============================================================================
# Traced Decorator
# =============================================================================

def traced(
    name: str = None,
    tags: Dict[str, Any] = None,
    record_args: bool = False,
    record_result: bool = False,
) -> Callable[[F], F]:
    """
    Decorator to automatically trace a function.
    
    Args:
        name: Span name (defaults to function name)
        tags: Additional tags to add
        record_args: Whether to record function arguments
        record_result: Whether to record return value
    
    Example:
        @traced("search_encode", tags={"source": "encode"})
        async def search_encode(query: str) -> List[Dataset]:
            ...
        
        @traced(record_args=True, record_result=True)
        def process_query(query: str) -> ToolResult:
            ...
    """
    def decorator(func: F) -> F:
        span_name = name or func.__name__
        
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                tracer = get_tracer()
                
                with tracer.start_span(span_name, tags=tags) as span:
                    span.add_tag("function", func.__name__)
                    span.add_tag("module", func.__module__)
                    
                    if record_args:
                        span.add_tag("args", str(args)[:200])
                        span.add_tag("kwargs", str(kwargs)[:200])
                    
                    result = await func(*args, **kwargs)
                    
                    if record_result:
                        span.add_tag("result", str(result)[:200])
                    
                    return result
            
            return async_wrapper  # type: ignore
        
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                tracer = get_tracer()
                
                with tracer.start_span(span_name, tags=tags) as span:
                    span.add_tag("function", func.__name__)
                    span.add_tag("module", func.__module__)
                    
                    if record_args:
                        span.add_tag("args", str(args)[:200])
                        span.add_tag("kwargs", str(kwargs)[:200])
                    
                    result = func(*args, **kwargs)
                    
                    if record_result:
                        span.add_tag("result", str(result)[:200])
                    
                    return result
            
            return sync_wrapper  # type: ignore
    
    return decorator


# =============================================================================
# Metrics
# =============================================================================

@dataclass
class MetricValue:
    """A single metric value."""
    name: str
    value: float
    timestamp: datetime
    tags: Dict[str, str]
    metric_type: str  # counter, gauge, histogram


class MetricsCollector:
    """
    Collect and aggregate metrics.
    
    Example:
        metrics = MetricsCollector()
        
        metrics.counter("api_calls", tags={"source": "encode"})
        metrics.gauge("active_jobs", 5)
        metrics.histogram("response_time_ms", 150.0)
        
        # Get all metrics
        all_metrics = metrics.get_metrics()
    """
    
    def __init__(self):
        self._counters: Dict[str, float] = {}
        self._gauges: Dict[str, float] = {}
        self._histograms: Dict[str, List[float]] = {}
        self._lock = threading.Lock()
    
    def _make_key(self, name: str, tags: Dict[str, str] = None) -> str:
        """Make a unique key from name and tags."""
        if not tags:
            return name
        sorted_tags = sorted(tags.items())
        tag_str = ",".join(f"{k}={v}" for k, v in sorted_tags)
        return f"{name}:{tag_str}"
    
    def counter(
        self,
        name: str,
        value: float = 1.0,
        tags: Dict[str, str] = None,
    ) -> None:
        """Increment a counter."""
        key = self._make_key(name, tags)
        with self._lock:
            self._counters[key] = self._counters.get(key, 0) + value
    
    def gauge(
        self,
        name: str,
        value: float,
        tags: Dict[str, str] = None,
    ) -> None:
        """Set a gauge value."""
        key = self._make_key(name, tags)
        with self._lock:
            self._gauges[key] = value
    
    def histogram(
        self,
        name: str,
        value: float,
        tags: Dict[str, str] = None,
    ) -> None:
        """Record a histogram value."""
        key = self._make_key(name, tags)
        with self._lock:
            if key not in self._histograms:
                self._histograms[key] = []
            self._histograms[key].append(value)
            
            # Keep only last 1000 values
            if len(self._histograms[key]) > 1000:
                self._histograms[key] = self._histograms[key][-1000:]
    
    def get_counter(self, name: str, tags: Dict[str, str] = None) -> float:
        """Get counter value."""
        key = self._make_key(name, tags)
        with self._lock:
            return self._counters.get(key, 0)
    
    def get_gauge(self, name: str, tags: Dict[str, str] = None) -> Optional[float]:
        """Get gauge value."""
        key = self._make_key(name, tags)
        with self._lock:
            return self._gauges.get(key)
    
    def get_histogram_stats(
        self,
        name: str,
        tags: Dict[str, str] = None,
    ) -> Optional[Dict[str, float]]:
        """Get histogram statistics (count, min, max, mean, p50, p95, p99)."""
        key = self._make_key(name, tags)
        with self._lock:
            values = self._histograms.get(key)
            if not values:
                return None
            
            sorted_values = sorted(values)
            n = len(sorted_values)
            
            return {
                "count": n,
                "min": sorted_values[0],
                "max": sorted_values[-1],
                "mean": sum(sorted_values) / n,
                "p50": sorted_values[int(n * 0.5)],
                "p95": sorted_values[int(n * 0.95)] if n >= 20 else sorted_values[-1],
                "p99": sorted_values[int(n * 0.99)] if n >= 100 else sorted_values[-1],
            }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get all metrics."""
        with self._lock:
            result = {
                "counters": dict(self._counters),
                "gauges": dict(self._gauges),
                "histograms": {},
            }
            
            for key, values in self._histograms.items():
                if values:
                    sorted_values = sorted(values)
                    n = len(sorted_values)
                    result["histograms"][key] = {
                        "count": n,
                        "min": sorted_values[0],
                        "max": sorted_values[-1],
                        "mean": sum(sorted_values) / n,
                    }
            
            return result
    
    def reset(self) -> None:
        """Reset all metrics."""
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._histograms.clear()


# Global metrics instance
_global_metrics: Optional[MetricsCollector] = None
_metrics_lock = threading.Lock()


def get_metrics() -> MetricsCollector:
    """Get the global metrics collector."""
    global _global_metrics
    
    with _metrics_lock:
        if _global_metrics is None:
            _global_metrics = MetricsCollector()
        return _global_metrics


def reset_metrics() -> None:
    """Reset the global metrics collector."""
    global _global_metrics
    with _metrics_lock:
        _global_metrics = None


# Convenience alias
metrics = property(lambda self: get_metrics())


# =============================================================================
# Timed Context Manager
# =============================================================================

@contextmanager
def timed(
    name: str,
    metric_name: str = None,
    tags: Dict[str, str] = None,
) -> Iterator[Dict[str, Any]]:
    """
    Context manager that measures execution time.
    
    Records both a span and a histogram metric.
    
    Example:
        with timed("fetch_dataset", tags={"source": "encode"}) as ctx:
            result = fetch()
        
        print(f"Took {ctx['duration_ms']:.2f}ms")
    """
    context = {
        "start_time": time.time(),
        "duration_ms": 0.0,
        "error": None,
    }
    
    tracer = get_tracer()
    metrics_collector = get_metrics()
    
    with tracer.start_span(name, tags=tags) as span:
        try:
            yield context
        except Exception as e:
            context["error"] = e
            raise
        finally:
            elapsed = (time.time() - context["start_time"]) * 1000
            context["duration_ms"] = elapsed
            
            # Record histogram
            hist_name = metric_name or f"{name}_duration_ms"
            metrics_collector.histogram(hist_name, elapsed, tags)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Span
    "Span",
    "SpanEvent",
    "SpanStatus",
    # Tracer
    "Tracer",
    "get_tracer",
    "set_tracer",
    "reset_tracer",
    # Exporters
    "SpanExporter",
    "ConsoleExporter",
    "JSONFileExporter",
    "InMemoryExporter",
    # Decorator
    "traced",
    # Context
    "get_current_trace_id",
    "get_current_span",
    # Metrics
    "MetricsCollector",
    "MetricValue",
    "get_metrics",
    "reset_metrics",
    # Utilities
    "timed",
]
