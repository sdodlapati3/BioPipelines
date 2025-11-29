"""
BioPipelines Exception Hierarchy
================================

A unified exception hierarchy for all BioPipelines errors.

Hierarchy:
    BioPipelinesError (base)
    ├── ConfigurationError
    ├── ValidationError
    │   └── WorkflowValidationError
    ├── ToolNotFoundError
    ├── ModuleNotFoundError
    ├── LLMError
    │   ├── LLMConnectionError
    │   ├── LLMRateLimitError
    │   └── LLMResponseError
    ├── SLURMError
    │   ├── SLURMSubmitError
    │   └── SLURMTimeoutError
    ├── DataError
    │   ├── DataNotFoundError
    │   └── DataDownloadError
    └── ExecutionError
        ├── CommandError
        └── FileOperationError

Usage:
    from workflow_composer.infrastructure.exceptions import (
        BioPipelinesError,
        LLMError,
        ToolNotFoundError,
    )
    
    try:
        workflow = generate_workflow(query)
    except ToolNotFoundError as e:
        print(f"Tool not found: {e.tool_name}")
    except LLMError as e:
        print(f"LLM failed: {e}")
    except BioPipelinesError as e:
        print(f"General error: {e}")
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any


class BioPipelinesError(Exception):
    """
    Base exception for all BioPipelines errors.
    
    All custom exceptions should inherit from this class.
    This allows catching all BioPipelines-specific errors with a single handler.
    
    Attributes:
        message: Human-readable error message
        details: Optional dictionary with additional context
        suggestion: Optional suggestion for fixing the error
    """
    
    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        suggestion: Optional[str] = None,
    ):
        self.message = message
        self.details = details or {}
        self.suggestion = suggestion
        super().__init__(self._format_message())
    
    def _format_message(self) -> str:
        """Format the full error message."""
        msg = self.message
        if self.suggestion:
            msg += f"\n\nSuggestion: {self.suggestion}"
        return msg
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "details": self.details,
            "suggestion": self.suggestion,
        }


# =============================================================================
# Configuration Errors
# =============================================================================

class ConfigurationError(BioPipelinesError):
    """
    Invalid or missing configuration.
    
    Raised when:
    - Config file is missing or malformed
    - Required settings are not set
    - Environment variables are missing
    """
    
    def __init__(
        self,
        message: str,
        config_key: Optional[str] = None,
        config_file: Optional[str] = None,
        **kwargs
    ):
        self.config_key = config_key
        self.config_file = config_file
        details = kwargs.pop("details", {})
        if config_key:
            details["config_key"] = config_key
        if config_file:
            details["config_file"] = config_file
        super().__init__(message, details=details, **kwargs)


# =============================================================================
# Validation Errors
# =============================================================================

@dataclass
class ValidationItem:
    """Single validation error."""
    field: str
    message: str
    value: Any = None
    
    def __str__(self) -> str:
        return f"{self.field}: {self.message}"


class ValidationError(BioPipelinesError):
    """
    Validation failed for input data.
    
    Used for:
    - Invalid user input
    - Invalid API request parameters
    - Schema validation failures
    """
    
    def __init__(
        self,
        message: str,
        errors: Optional[List[ValidationItem]] = None,
        **kwargs
    ):
        self.errors = errors or []
        details = kwargs.pop("details", {})
        details["errors"] = [str(e) for e in self.errors]
        super().__init__(message, details=details, **kwargs)


class WorkflowValidationError(ValidationError):
    """
    Workflow failed validation.
    
    Raised when a generated workflow has issues like:
    - Missing required tools
    - Invalid connections between processes
    - Circular dependencies
    - Missing input files
    """
    
    def __init__(
        self,
        message: str,
        workflow_name: Optional[str] = None,
        errors: Optional[List[ValidationItem]] = None,
        **kwargs
    ):
        self.workflow_name = workflow_name
        details = kwargs.pop("details", {})
        if workflow_name:
            details["workflow_name"] = workflow_name
        super().__init__(message, errors=errors, details=details, **kwargs)


# =============================================================================
# Tool & Module Errors
# =============================================================================

class ToolNotFoundError(BioPipelinesError):
    """
    Requested tool was not found in the catalog.
    
    Attributes:
        tool_name: Name of the tool that wasn't found
        available_tools: List of similar available tools (if any)
    """
    
    def __init__(
        self,
        tool_name: str,
        available_tools: Optional[List[str]] = None,
        **kwargs
    ):
        self.tool_name = tool_name
        self.available_tools = available_tools or []
        
        message = f"Tool not found: '{tool_name}'"
        details = {"tool_name": tool_name}
        
        if available_tools:
            details["similar_tools"] = available_tools[:5]
            suggestion = f"Did you mean one of: {', '.join(available_tools[:5])}?"
        else:
            suggestion = "Check the tool catalog with `bp tools --list` or search with `bp tools --search <keyword>`"
        
        super().__init__(message, details=details, suggestion=suggestion, **kwargs)


class ModuleNotFoundError(BioPipelinesError):
    """
    Nextflow module was not found for a tool.
    
    Attributes:
        module_name: Name of the missing module
        tool_name: Associated tool name
    """
    
    def __init__(
        self,
        module_name: str,
        tool_name: Optional[str] = None,
        **kwargs
    ):
        self.module_name = module_name
        self.tool_name = tool_name
        
        message = f"Module not found: '{module_name}'"
        if tool_name:
            message += f" (for tool: {tool_name})"
        
        details = {"module_name": module_name}
        if tool_name:
            details["tool_name"] = tool_name
        
        suggestion = "The module can be auto-generated with --auto-create-modules flag"
        
        super().__init__(message, details=details, suggestion=suggestion, **kwargs)


# =============================================================================
# LLM Errors
# =============================================================================

class LLMError(BioPipelinesError):
    """
    Base class for LLM-related errors.
    
    Attributes:
        provider: Name of the LLM provider (e.g., "openai", "anthropic")
        model: Model name if available
    """
    
    def __init__(
        self,
        message: str,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs
    ):
        self.provider = provider
        self.model = model
        details = kwargs.pop("details", {})
        if provider:
            details["provider"] = provider
        if model:
            details["model"] = model
        super().__init__(message, details=details, **kwargs)


class LLMConnectionError(LLMError):
    """
    Failed to connect to LLM provider.
    
    Common causes:
    - Network issues
    - Provider is down
    - Invalid API endpoint
    """
    
    def __init__(
        self,
        provider: str,
        endpoint: Optional[str] = None,
        **kwargs
    ):
        self.endpoint = endpoint
        message = f"Failed to connect to LLM provider: {provider}"
        if endpoint:
            message += f" at {endpoint}"
        
        suggestion = kwargs.pop("suggestion", None)
        if not suggestion:
            if provider.lower() == "ollama":
                suggestion = "Make sure Ollama is running: `ollama serve`"
            elif provider.lower() == "vllm":
                suggestion = "Make sure vLLM server is running on the expected port"
            else:
                suggestion = "Check your network connection and API endpoint"
        
        super().__init__(message, provider=provider, suggestion=suggestion, **kwargs)


class LLMRateLimitError(LLMError):
    """
    Rate limit exceeded for LLM provider.
    
    Attributes:
        retry_after: Seconds to wait before retrying (if known)
    """
    
    def __init__(
        self,
        provider: str,
        retry_after: Optional[int] = None,
        **kwargs
    ):
        self.retry_after = retry_after
        message = f"Rate limit exceeded for {provider}"
        if retry_after:
            message += f". Retry after {retry_after} seconds."
        
        details = kwargs.pop("details", {})
        if retry_after:
            details["retry_after"] = retry_after
        
        suggestion = "Consider using a different provider or waiting before retrying"
        
        super().__init__(
            message,
            provider=provider,
            details=details,
            suggestion=suggestion,
            **kwargs
        )


class LLMResponseError(LLMError):
    """
    Invalid or unexpected response from LLM.
    
    Raised when:
    - Response format is invalid
    - Response is empty
    - Response parsing failed
    """
    
    def __init__(
        self,
        message: str,
        provider: str,
        raw_response: Optional[str] = None,
        **kwargs
    ):
        self.raw_response = raw_response
        details = kwargs.pop("details", {})
        if raw_response and len(raw_response) < 500:
            details["raw_response"] = raw_response
        super().__init__(message, provider=provider, details=details, **kwargs)


# =============================================================================
# SLURM Errors
# =============================================================================

class SLURMError(BioPipelinesError):
    """
    Base class for SLURM-related errors.
    
    Attributes:
        job_id: SLURM job ID if applicable
    """
    
    def __init__(
        self,
        message: str,
        job_id: Optional[str] = None,
        **kwargs
    ):
        self.job_id = job_id
        details = kwargs.pop("details", {})
        if job_id:
            details["job_id"] = job_id
        super().__init__(message, details=details, **kwargs)


class SLURMSubmitError(SLURMError):
    """
    Failed to submit job to SLURM.
    
    Common causes:
    - Invalid job script
    - Insufficient resources requested
    - Queue/partition doesn't exist
    """
    
    def __init__(
        self,
        message: str,
        script_path: Optional[str] = None,
        stderr: Optional[str] = None,
        **kwargs
    ):
        self.script_path = script_path
        self.stderr = stderr
        details = kwargs.pop("details", {})
        if script_path:
            details["script_path"] = script_path
        if stderr:
            details["stderr"] = stderr[:500]  # Truncate long errors
        super().__init__(message, details=details, **kwargs)


class SLURMTimeoutError(SLURMError):
    """
    SLURM job timed out.
    
    Attributes:
        requested_time: Time limit that was requested
        elapsed_time: How long the job actually ran
    """
    
    def __init__(
        self,
        job_id: str,
        requested_time: Optional[str] = None,
        elapsed_time: Optional[str] = None,
        **kwargs
    ):
        self.requested_time = requested_time
        self.elapsed_time = elapsed_time
        
        message = f"SLURM job {job_id} timed out"
        if requested_time:
            message += f" (limit: {requested_time})"
        
        details = kwargs.pop("details", {})
        if requested_time:
            details["requested_time"] = requested_time
        if elapsed_time:
            details["elapsed_time"] = elapsed_time
        
        suggestion = "Consider increasing --time or optimizing the workflow"
        
        super().__init__(
            message,
            job_id=job_id,
            details=details,
            suggestion=suggestion,
            **kwargs
        )


# =============================================================================
# Data Errors
# =============================================================================

class DataError(BioPipelinesError):
    """Base class for data-related errors."""
    pass


class DataNotFoundError(DataError):
    """
    Requested data was not found.
    
    Common cases:
    - Input file doesn't exist
    - Reference genome not downloaded
    - Sample not in samplesheet
    """
    
    def __init__(
        self,
        message: str,
        path: Optional[str] = None,
        data_type: Optional[str] = None,
        **kwargs
    ):
        self.path = path
        self.data_type = data_type
        details = kwargs.pop("details", {})
        if path:
            details["path"] = path
        if data_type:
            details["data_type"] = data_type
        super().__init__(message, details=details, **kwargs)


class DataDownloadError(DataError):
    """
    Failed to download data.
    
    Attributes:
        url: URL that failed
        status_code: HTTP status code if applicable
    """
    
    def __init__(
        self,
        message: str,
        url: Optional[str] = None,
        status_code: Optional[int] = None,
        **kwargs
    ):
        self.url = url
        self.status_code = status_code
        details = kwargs.pop("details", {})
        if url:
            details["url"] = url
        if status_code:
            details["status_code"] = status_code
        super().__init__(message, details=details, **kwargs)


# =============================================================================
# Execution Errors
# =============================================================================

class ExecutionError(BioPipelinesError):
    """Base class for execution-related errors."""
    pass


class CommandError(ExecutionError):
    """
    Shell command execution failed.
    
    Attributes:
        command: The command that failed
        return_code: Process exit code
        stdout: Standard output
        stderr: Standard error
    """
    
    def __init__(
        self,
        message: str,
        command: Optional[str] = None,
        return_code: Optional[int] = None,
        stdout: Optional[str] = None,
        stderr: Optional[str] = None,
        **kwargs
    ):
        self.command = command
        self.return_code = return_code
        self.stdout = stdout
        self.stderr = stderr
        
        details = kwargs.pop("details", {})
        if command:
            details["command"] = command
        if return_code is not None:
            details["return_code"] = return_code
        if stderr:
            details["stderr"] = stderr[:1000]  # Truncate
        
        super().__init__(message, details=details, **kwargs)


class FileOperationError(ExecutionError):
    """
    File operation failed.
    
    Common causes:
    - Permission denied
    - Disk full
    - Path doesn't exist
    """
    
    def __init__(
        self,
        message: str,
        path: Optional[str] = None,
        operation: Optional[str] = None,
        **kwargs
    ):
        self.path = path
        self.operation = operation
        details = kwargs.pop("details", {})
        if path:
            details["path"] = path
        if operation:
            details["operation"] = operation
        super().__init__(message, details=details, **kwargs)
