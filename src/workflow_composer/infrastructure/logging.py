"""
Structured Logging
==================

Structured logging with correlation IDs and JSON output.

Features:
- Correlation IDs for request tracing
- JSON output for log aggregation
- Context-aware logging
- Operation timing
- Log level configuration

Usage:
    from workflow_composer.infrastructure.logging import (
        get_logger,
        operation_context,
        configure_logging,
    )
    
    logger = get_logger(__name__)
    
    # Simple logging
    logger.info("Processing started", query=query, user_id=user_id)
    
    # With operation context
    with operation_context("generate_workflow", query=query) as ctx:
        logger.info("Parsing intent")
        intent = parse(query)
        logger.info("Intent parsed", analysis_type=intent.type)
        
    # Auto-logs: operation.started, operation.completed/failed
"""

from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
import logging
import sys
import threading
import time
import uuid
from typing import Any, Dict, Generator, Optional, TextIO

# Thread-local storage for context
_context = threading.local()


class LogLevel(Enum):
    """Log levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class LogContext:
    """Context for structured logging."""
    
    correlation_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    operation: Optional[str] = None
    start_time: float = field(default_factory=time.time)
    extra: Dict[str, Any] = field(default_factory=dict)
    
    def bind(self, **kwargs) -> "LogContext":
        """Add extra context."""
        self.extra.update(kwargs)
        return self
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        result = {
            "correlation_id": self.correlation_id,
        }
        if self.operation:
            result["operation"] = self.operation
        result.update(self.extra)
        return result


def get_context() -> Optional[LogContext]:
    """Get current logging context."""
    return getattr(_context, "current", None)


def set_context(ctx: Optional[LogContext]) -> None:
    """Set current logging context."""
    _context.current = ctx


class StructuredFormatter(logging.Formatter):
    """
    JSON formatter for structured logging.
    
    Outputs logs as JSON lines for easy parsing.
    """
    
    def __init__(self, include_timestamp: bool = True):
        super().__init__()
        self.include_timestamp = include_timestamp
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        
        if self.include_timestamp:
            log_data["timestamp"] = datetime.utcnow().isoformat() + "Z"
        
        # Add context if available
        ctx = get_context()
        if ctx:
            log_data.update(ctx.to_dict())
        
        # Add extra fields from record
        if hasattr(record, "extra_fields"):
            log_data.update(record.extra_fields)
        
        # Add exception info
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_data, default=str)


class PrettyFormatter(logging.Formatter):
    """
    Human-readable formatter for development.
    
    Outputs colored, readable logs.
    """
    
    COLORS = {
        "DEBUG": "\033[36m",     # Cyan
        "INFO": "\033[32m",      # Green
        "WARNING": "\033[33m",   # Yellow
        "ERROR": "\033[31m",     # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record for terminal."""
        color = self.COLORS.get(record.levelname, "")
        reset = self.RESET
        
        # Build message parts
        parts = [
            f"{color}{record.levelname:8}{reset}",
            f"{record.name:30}",
            record.getMessage(),
        ]
        
        # Add context
        ctx = get_context()
        if ctx and ctx.correlation_id:
            parts.insert(0, f"[{ctx.correlation_id}]")
        
        # Add extra fields
        extra = {}
        if hasattr(record, "extra_fields"):
            extra.update(record.extra_fields)
        
        if extra:
            extra_str = " ".join(f"{k}={v}" for k, v in extra.items())
            parts.append(f"  ({extra_str})")
        
        return " ".join(parts)


class StructuredLogger:
    """
    Logger wrapper with structured logging support.
    
    Adds extra fields to all log messages.
    """
    
    def __init__(self, logger: logging.Logger):
        self._logger = logger
    
    def _log(self, level: int, msg: str, **kwargs) -> None:
        """Log with extra fields."""
        extra_fields = {}
        exc_info = kwargs.pop("exc_info", None)
        
        for key, value in kwargs.items():
            extra_fields[key] = value
        
        # Create record with extra fields
        record = self._logger.makeRecord(
            name=self._logger.name,
            level=level,
            fn="",
            lno=0,
            msg=msg,
            args=(),
            exc_info=exc_info,
        )
        record.extra_fields = extra_fields
        
        self._logger.handle(record)
    
    def debug(self, msg: str, **kwargs) -> None:
        """Log at DEBUG level."""
        self._log(logging.DEBUG, msg, **kwargs)
    
    def info(self, msg: str, **kwargs) -> None:
        """Log at INFO level."""
        self._log(logging.INFO, msg, **kwargs)
    
    def warning(self, msg: str, **kwargs) -> None:
        """Log at WARNING level."""
        self._log(logging.WARNING, msg, **kwargs)
    
    def warn(self, msg: str, **kwargs) -> None:
        """Alias for warning."""
        self.warning(msg, **kwargs)
    
    def error(self, msg: str, **kwargs) -> None:
        """Log at ERROR level."""
        self._log(logging.ERROR, msg, **kwargs)
    
    def critical(self, msg: str, **kwargs) -> None:
        """Log at CRITICAL level."""
        self._log(logging.CRITICAL, msg, **kwargs)
    
    def exception(self, msg: str, **kwargs) -> None:
        """Log exception with traceback."""
        kwargs["exc_info"] = True
        self._log(logging.ERROR, msg, **kwargs)
    
    def bind(self, **kwargs) -> "BoundLogger":
        """Create a bound logger with extra context."""
        return BoundLogger(self, kwargs)


class BoundLogger:
    """Logger with pre-bound context fields."""
    
    def __init__(self, logger: StructuredLogger, context: Dict[str, Any]):
        self._logger = logger
        self._context = context
    
    def _merge(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Merge bound context with call kwargs."""
        merged = self._context.copy()
        merged.update(kwargs)
        return merged
    
    def debug(self, msg: str, **kwargs) -> None:
        self._logger.debug(msg, **self._merge(kwargs))
    
    def info(self, msg: str, **kwargs) -> None:
        self._logger.info(msg, **self._merge(kwargs))
    
    def warning(self, msg: str, **kwargs) -> None:
        self._logger.warning(msg, **self._merge(kwargs))
    
    def error(self, msg: str, **kwargs) -> None:
        self._logger.error(msg, **self._merge(kwargs))
    
    def critical(self, msg: str, **kwargs) -> None:
        self._logger.critical(msg, **self._merge(kwargs))
    
    def bind(self, **kwargs) -> "BoundLogger":
        """Create new bound logger with additional context."""
        merged = self._context.copy()
        merged.update(kwargs)
        return BoundLogger(self._logger, merged)


# Logger cache
_loggers: Dict[str, StructuredLogger] = {}
_logger_lock = threading.Lock()


def get_logger(name: str) -> StructuredLogger:
    """
    Get a structured logger.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        StructuredLogger instance
        
    Example:
        logger = get_logger(__name__)
        logger.info("Processing", query=query, user_id=user_id)
    """
    if name not in _loggers:
        with _logger_lock:
            if name not in _loggers:
                _loggers[name] = StructuredLogger(logging.getLogger(name))
    return _loggers[name]


@contextmanager
def operation_context(
    operation: str,
    correlation_id: Optional[str] = None,
    **metadata
) -> Generator[LogContext, None, None]:
    """
    Context manager for operation logging.
    
    Automatically logs operation start/complete/failed.
    
    Args:
        operation: Operation name
        correlation_id: Optional correlation ID (generated if not provided)
        **metadata: Additional context fields
        
    Yields:
        LogContext with correlation_id
        
    Example:
        with operation_context("generate_workflow", query=query) as ctx:
            workflow = generate(query)
            
        # Logs:
        # {"event": "operation.started", "operation": "generate_workflow", ...}
        # {"event": "operation.completed", "duration_ms": 1234, ...}
    """
    # Create context
    ctx = LogContext(
        correlation_id=correlation_id or str(uuid.uuid4())[:8],
        operation=operation,
        start_time=time.time(),
        extra=metadata,
    )
    
    # Save previous context
    previous = get_context()
    set_context(ctx)
    
    logger = get_logger("workflow_composer.operation")
    
    try:
        logger.info("operation.started", operation=operation, **metadata)
        yield ctx
        
        duration_ms = (time.time() - ctx.start_time) * 1000
        logger.info(
            "operation.completed",
            operation=operation,
            duration_ms=round(duration_ms, 2),
        )
    except Exception as e:
        duration_ms = (time.time() - ctx.start_time) * 1000
        logger.error(
            "operation.failed",
            operation=operation,
            duration_ms=round(duration_ms, 2),
            error=str(e),
            error_type=type(e).__name__,
        )
        raise
    finally:
        set_context(previous)


def configure_logging(
    level: LogLevel = LogLevel.INFO,
    json_output: bool = False,
    output: TextIO = sys.stderr,
) -> None:
    """
    Configure logging for the application.
    
    Args:
        level: Log level
        json_output: Use JSON formatter (for production)
        output: Output stream
        
    Example:
        # Development
        configure_logging(level=LogLevel.DEBUG, json_output=False)
        
        # Production
        configure_logging(level=LogLevel.INFO, json_output=True)
    """
    # Get root logger for workflow_composer
    root_logger = logging.getLogger("workflow_composer")
    root_logger.setLevel(getattr(logging, level.value))
    
    # Remove existing handlers
    root_logger.handlers.clear()
    
    # Create handler
    handler = logging.StreamHandler(output)
    handler.setLevel(getattr(logging, level.value))
    
    # Set formatter
    if json_output:
        handler.setFormatter(StructuredFormatter())
    else:
        handler.setFormatter(PrettyFormatter())
    
    root_logger.addHandler(handler)
    
    # Prevent propagation to root logger
    root_logger.propagate = False


def with_correlation_id(correlation_id: str):
    """
    Decorator to set correlation ID for a function.
    
    Example:
        @with_correlation_id("req-123")
        def handle_request():
            logger.info("Handling request")  # Includes correlation_id
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            ctx = LogContext(correlation_id=correlation_id)
            previous = get_context()
            set_context(ctx)
            try:
                return func(*args, **kwargs)
            finally:
                set_context(previous)
        return wrapper
    return decorator
