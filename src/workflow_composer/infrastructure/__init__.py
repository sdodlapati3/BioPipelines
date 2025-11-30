"""
Infrastructure Layer
====================

Core infrastructure components for BioPipelines:

- **exceptions**: Unified error hierarchy
- **protocols**: Interface definitions (Python Protocols)
- **container**: Dependency injection container
- **logging**: Structured logging with correlation IDs
- **settings**: Unified configuration management
- **resilience**: Circuit breaker, retry with backoff
- **observability**: Distributed tracing, metrics
- **semantic_cache**: TTL + similarity-based caching

Usage:
    from workflow_composer.infrastructure import (
        # Exceptions
        BioPipelinesError,
        ConfigurationError,
        LLMError,
        ToolNotFoundError,
        
        # DI Container
        Container,
        get_container,
        
        # Protocols
        LLMProtocol,
        ToolProtocol,
        
        # Settings
        Settings,
        get_settings,
        
        # Logging
        get_logger,
        operation_context,
        
        # Resilience
        CircuitBreaker,
        retry_with_backoff,
        resilient_call,
        
        # Observability
        get_tracer,
        traced,
        get_metrics,
        
        # Caching
        SemanticCache,
        get_cache,
    )
"""

from .exceptions import (
    BioPipelinesError,
    ConfigurationError,
    ValidationError,
    WorkflowValidationError,
    ToolNotFoundError,
    ModuleNotFoundError,
    LLMError,
    LLMConnectionError,
    LLMRateLimitError,
    LLMResponseError,
    SLURMError,
    SLURMSubmitError,
    SLURMTimeoutError,
    DataError,
    DataNotFoundError,
    DataDownloadError,
    ExecutionError,
    CommandError,
    FileOperationError,
)

from .protocols import (
    LLMProtocol,
    ToolProtocol,
    ToolExecutorProtocol,
    WorkflowGeneratorProtocol,
    IntentParserProtocol,
    StorageProtocol,
    EventPublisherProtocol,
)

from .container import (
    Container,
    get_container,
    reset_container,
    inject,
    Scope,
)

from .logging import (
    get_logger,
    operation_context,
    configure_logging,
    LogLevel,
)

from .settings import (
    Settings,
    LLMSettings,
    SLURMSettings,
    PathSettings,
    get_settings,
    reload_settings,
)

from .resilience import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitState,
    CircuitBreakerError,
    BackoffConfig,
    retry_with_backoff,
    resilient_call,
    get_circuit_breaker,
    reset_all_circuits,
    get_all_circuit_metrics,
    circuit_protected,
    calculate_backoff_delay,
)

from .observability import (
    Tracer,
    Span,
    SpanEvent,
    SpanStatus,
    MetricsCollector,
    MetricValue,
    get_tracer,
    get_metrics,
    traced,
    timed,
    get_current_trace_id,
    get_current_span,
    reset_tracer,
    reset_metrics,
    SpanExporter,
    ConsoleExporter,
    JSONFileExporter,
    InMemoryExporter,
)

from .semantic_cache import (
    SemanticCache,
    CacheEntry,
    EmbeddingModel,
    get_cache,
    clear_all_caches,
    get_all_cache_stats,
    cached,
)

__all__ = [
    # Exceptions
    "BioPipelinesError",
    "ConfigurationError",
    "ValidationError",
    "WorkflowValidationError",
    "ToolNotFoundError",
    "ModuleNotFoundError",
    "LLMError",
    "LLMConnectionError",
    "LLMRateLimitError",
    "LLMResponseError",
    "SLURMError",
    "SLURMSubmitError",
    "SLURMTimeoutError",
    "DataError",
    "DataNotFoundError",
    "DataDownloadError",
    "ExecutionError",
    "CommandError",
    "FileOperationError",
    # Protocols
    "LLMProtocol",
    "ToolProtocol",
    "ToolExecutorProtocol",
    "WorkflowGeneratorProtocol",
    "IntentParserProtocol",
    "StorageProtocol",
    "EventPublisherProtocol",
    # Container
    "Container",
    "get_container",
    "reset_container",
    "inject",
    "Scope",
    # Logging
    "get_logger",
    "operation_context",
    "configure_logging",
    "LogLevel",
    # Settings
    "Settings",
    "LLMSettings",
    "SLURMSettings",
    "PathSettings",
    "get_settings",
    "reload_settings",
    # Resilience
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitBreakerError",
    "CircuitState",
    "BackoffConfig",
    "retry_with_backoff",
    "resilient_call",
    "get_circuit_breaker",
    "reset_all_circuits",
    "get_all_circuit_metrics",
    "circuit_protected",
    "calculate_backoff_delay",
    # Observability
    "Tracer",
    "Span",
    "SpanEvent",
    "SpanStatus",
    "MetricsCollector",
    "MetricValue",
    "get_tracer",
    "get_metrics",
    "traced",
    "timed",
    "get_current_trace_id",
    "get_current_span",
    "reset_tracer",
    "reset_metrics",
    "SpanExporter",
    "ConsoleExporter",
    "JSONFileExporter",
    "InMemoryExporter",
    # Caching
    "SemanticCache",
    "CacheEntry",
    "EmbeddingModel",
    "get_cache",
    "clear_all_caches",
    "get_all_cache_stats",
    "cached",
]
