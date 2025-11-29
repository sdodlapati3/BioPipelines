"""
Infrastructure Layer
====================

Core infrastructure components for BioPipelines:

- **exceptions**: Unified error hierarchy
- **protocols**: Interface definitions (Python Protocols)
- **container**: Dependency injection container
- **logging**: Structured logging with correlation IDs
- **settings**: Unified configuration management

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
]
