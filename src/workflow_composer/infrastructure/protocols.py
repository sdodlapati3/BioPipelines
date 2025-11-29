"""
Protocol Definitions (Structural Typing)
=========================================

Python Protocol definitions for BioPipelines interfaces.

Protocols define the expected interface without requiring inheritance.
Any class that implements the required methods is compatible, enabling
duck typing with full type safety.

Benefits:
- No coupling through inheritance
- Easy testing (any object with matching methods works)
- More Pythonic (structural subtyping)
- Better IDE support

Usage:
    from workflow_composer.infrastructure.protocols import LLMProtocol
    
    def generate_workflow(llm: LLMProtocol) -> Workflow:
        response = llm.complete("...")  # Type safe
        return parse_workflow(response)
    
    # Works with any class that has the right methods
    class MyCustomLLM:
        def complete(self, prompt: str, **kwargs) -> str:
            return "response"
        
        def chat(self, messages: List[Message], **kwargs) -> str:
            return "response"
    
    workflow = generate_workflow(MyCustomLLM())  # Type checks!
"""

from typing import (
    Protocol,
    runtime_checkable,
    List,
    Dict,
    Any,
    Optional,
    Iterator,
    TypeVar,
    Generic,
    Union,
    Callable,
    Awaitable,
)
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from datetime import datetime


# =============================================================================
# Common Types
# =============================================================================

@dataclass
class Message:
    """Chat message structure."""
    role: str  # "system", "user", "assistant"
    content: str
    name: Optional[str] = None


@dataclass
class ToolResult:
    """Result from tool execution."""
    success: bool
    output: Any
    error: Optional[str] = None
    execution_time_ms: float = 0


@dataclass
class WorkflowStep:
    """A single step in a workflow."""
    name: str
    tool: str
    inputs: Dict[str, Any]
    outputs: Dict[str, str]
    dependencies: List[str]


# =============================================================================
# LLM Protocol
# =============================================================================

@runtime_checkable
class LLMProtocol(Protocol):
    """
    Protocol for LLM adapters.
    
    Any LLM implementation (OpenAI, Anthropic, Ollama, etc.) must provide:
    - complete(): Single prompt completion
    - chat(): Multi-turn conversation
    - model_name: The model identifier
    
    Optional:
    - stream(): Streaming completion
    - is_available(): Health check
    """
    
    @property
    def model_name(self) -> str:
        """Return the model identifier."""
        ...
    
    def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Generate a completion for a single prompt.
        
        Args:
            prompt: The user prompt
            system_prompt: Optional system instructions
            **kwargs: Provider-specific options (temperature, max_tokens, etc.)
            
        Returns:
            The generated text response
        """
        ...
    
    def chat(
        self,
        messages: List[Message],
        **kwargs
    ) -> str:
        """
        Generate a response in a multi-turn conversation.
        
        Args:
            messages: List of Message objects
            **kwargs: Provider-specific options
            
        Returns:
            The assistant's response text
        """
        ...


@runtime_checkable
class StreamingLLMProtocol(LLMProtocol, Protocol):
    """LLM with streaming support."""
    
    def stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Iterator[str]:
        """Stream completion token by token."""
        ...


# =============================================================================
# Tool Protocol
# =============================================================================

@runtime_checkable
class ToolProtocol(Protocol):
    """
    Protocol for individual tools.
    
    A tool represents a bioinformatics tool (STAR, BWA, etc.)
    with its metadata and execution interface.
    """
    
    @property
    def name(self) -> str:
        """Tool name (e.g., 'star', 'bwa-mem2')."""
        ...
    
    @property
    def version(self) -> str:
        """Tool version."""
        ...
    
    @property
    def container(self) -> str:
        """Container image for the tool."""
        ...
    
    @property
    def parameters(self) -> Dict[str, Any]:
        """Tool parameters with defaults."""
        ...
    
    def validate_params(self, params: Dict[str, Any]) -> bool:
        """Validate parameter values."""
        ...


@runtime_checkable
class ToolExecutorProtocol(Protocol):
    """
    Protocol for tool execution.
    
    Handles running tools and collecting results.
    """
    
    async def execute(
        self,
        tool_name: str,
        params: Dict[str, Any],
        **kwargs
    ) -> ToolResult:
        """
        Execute a tool asynchronously.
        
        Args:
            tool_name: Name of the tool to execute
            params: Tool parameters
            **kwargs: Additional options (timeout, retry, etc.)
            
        Returns:
            ToolResult with success/failure and output
        """
        ...
    
    def execute_sync(
        self,
        tool_name: str,
        params: Dict[str, Any],
        **kwargs
    ) -> ToolResult:
        """Synchronous wrapper for execute()."""
        ...


# =============================================================================
# Workflow Protocols
# =============================================================================

@runtime_checkable
class WorkflowProtocol(Protocol):
    """
    Protocol for Workflow objects.
    
    Represents a complete bioinformatics workflow.
    """
    
    @property
    def name(self) -> str:
        """Workflow name."""
        ...
    
    @property
    def steps(self) -> List[WorkflowStep]:
        """Ordered list of workflow steps."""
        ...
    
    def to_nextflow(self) -> str:
        """Generate Nextflow DSL2 code."""
        ...
    
    def save(self, output_dir: Union[str, Path]) -> Path:
        """Save workflow to directory."""
        ...
    
    def validate(self) -> List[str]:
        """Validate workflow, return list of errors."""
        ...


@runtime_checkable
class WorkflowGeneratorProtocol(Protocol):
    """
    Protocol for workflow generation.
    
    Takes parsed intent and tools, produces a workflow.
    """
    
    def generate(
        self,
        analysis_type: str,
        tools: List[ToolProtocol],
        **options
    ) -> WorkflowProtocol:
        """
        Generate a workflow.
        
        Args:
            analysis_type: Type of analysis (e.g., "rna-seq")
            tools: List of tools to include
            **options: Additional options
            
        Returns:
            Generated Workflow
        """
        ...


# =============================================================================
# Intent Parsing Protocol
# =============================================================================

@runtime_checkable
class IntentParserProtocol(Protocol):
    """
    Protocol for intent parsing.
    
    Parses natural language queries into structured intents.
    """
    
    def parse(self, query: str) -> "ParsedIntentProtocol":
        """
        Parse a natural language query.
        
        Args:
            query: User's natural language query
            
        Returns:
            ParsedIntent with structured information
        """
        ...


@runtime_checkable
class ParsedIntentProtocol(Protocol):
    """Protocol for parsed intent results."""
    
    @property
    def analysis_type(self) -> str:
        """The detected analysis type."""
        ...
    
    @property
    def organism(self) -> Optional[str]:
        """Detected organism if any."""
        ...
    
    @property
    def confidence(self) -> float:
        """Confidence score 0-1."""
        ...
    
    @property
    def parameters(self) -> Dict[str, Any]:
        """Extracted parameters."""
        ...


# =============================================================================
# Storage Protocol
# =============================================================================

@runtime_checkable
class StorageProtocol(Protocol):
    """
    Protocol for file/object storage.
    
    Abstracts local filesystem, cloud storage (GCS, S3), etc.
    """
    
    def read(self, path: str) -> bytes:
        """Read file contents."""
        ...
    
    def write(self, path: str, content: bytes) -> None:
        """Write file contents."""
        ...
    
    def exists(self, path: str) -> bool:
        """Check if path exists."""
        ...
    
    def list(self, prefix: str) -> List[str]:
        """List files with prefix."""
        ...
    
    def delete(self, path: str) -> None:
        """Delete file."""
        ...


# =============================================================================
# Event Protocol
# =============================================================================

T = TypeVar("T")


@dataclass
class Event:
    """Base event structure."""
    event_type: str
    timestamp: datetime
    data: Dict[str, Any]
    correlation_id: Optional[str] = None


@runtime_checkable
class EventPublisherProtocol(Protocol):
    """
    Protocol for event publishing.
    
    Publishes events to subscribers (for event-driven architecture).
    """
    
    def publish(self, event: Event) -> None:
        """Publish an event."""
        ...
    
    def subscribe(
        self,
        event_type: str,
        handler: Callable[[Event], None]
    ) -> None:
        """Subscribe to an event type."""
        ...
    
    def unsubscribe(
        self,
        event_type: str,
        handler: Callable[[Event], None]
    ) -> None:
        """Unsubscribe from an event type."""
        ...


# =============================================================================
# Repository Protocol
# =============================================================================

@runtime_checkable
class RepositoryProtocol(Protocol, Generic[T]):
    """
    Generic repository protocol for data access.
    
    Provides CRUD operations for domain entities.
    """
    
    def get(self, id: str) -> Optional[T]:
        """Get entity by ID."""
        ...
    
    def list(self, **filters) -> List[T]:
        """List entities with optional filters."""
        ...
    
    def save(self, entity: T) -> T:
        """Save (create/update) entity."""
        ...
    
    def delete(self, id: str) -> bool:
        """Delete entity by ID."""
        ...


# =============================================================================
# Helpers
# =============================================================================

def implements_protocol(obj: Any, protocol: type) -> bool:
    """
    Check if an object implements a protocol at runtime.
    
    Args:
        obj: The object to check
        protocol: The protocol class
        
    Returns:
        True if obj implements the protocol
        
    Example:
        if implements_protocol(my_llm, LLMProtocol):
            result = my_llm.complete("Hello")
    """
    if hasattr(protocol, "__protocol_attrs__"):
        # Check Protocol attributes
        for attr in protocol.__protocol_attrs__:
            if not hasattr(obj, attr):
                return False
        return True
    else:
        # Use isinstance for @runtime_checkable protocols
        return isinstance(obj, protocol)
