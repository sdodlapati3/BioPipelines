"""
Base Types for Agent Tools
==========================

Core types and enums used across all tool modules.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ToolName(Enum):
    """Available agent tools."""
    # Data Discovery
    SCAN_DATA = "scan_data"
    SEARCH_DATABASES = "search_databases"
    SEARCH_TCGA = "search_tcga"
    DESCRIBE_FILES = "describe_files"
    VALIDATE_DATASET = "validate_dataset"
    
    # Data Management
    DOWNLOAD_DATASET = "download_dataset"
    DOWNLOAD_REFERENCE = "download_reference"
    BUILD_INDEX = "build_index"
    CLEANUP_DATA = "cleanup_data"
    CONFIRM_CLEANUP = "confirm_cleanup"
    
    # Workflow
    GENERATE_WORKFLOW = "generate_workflow"
    LIST_WORKFLOWS = "list_workflows"
    CHECK_REFERENCES = "check_references"
    
    # Execution
    SUBMIT_JOB = "submit_job"
    GET_JOB_STATUS = "get_job_status"
    GET_LOGS = "get_logs"
    CANCEL_JOB = "cancel_job"
    CHECK_SYSTEM_HEALTH = "check_system_health"
    RESTART_VLLM = "restart_vllm"
    RESUBMIT_JOB = "resubmit_job"
    WATCH_JOB = "watch_job"
    LIST_JOBS = "list_jobs"
    
    # Monitoring
    MONITOR_JOBS = "monitor_jobs"
    DOWNLOAD_RESULTS = "download_results"
    
    # Diagnostics
    DIAGNOSE_ERROR = "diagnose_error"
    ANALYZE_RESULTS = "analyze_results"
    RECOVER_ERROR = "recover_error"
    
    # Visualization
    VISUALIZE_WORKFLOW = "visualize_workflow"
    
    # Education
    EXPLAIN_CONCEPT = "explain_concept"
    COMPARE_SAMPLES = "compare_samples"
    
    # System
    RUN_COMMAND = "run_command"
    SHOW_HELP = "show_help"


# =============================================================================
# LEGACY TOOL_PATTERNS - DEPRECATED
# =============================================================================
# These patterns are maintained ONLY for backward compatibility with
# AgentTools.detect_tool() as a fallback when HybridQueryParser is unavailable.
#
# NEW CODE should use:
#   from workflow_composer.agents.intent import HybridQueryParser
#   parser = HybridQueryParser()
#   result = parser.parse(query)
#
# The HybridQueryParser provides:
# - Semantic similarity matching (handles paraphrases)
# - Domain-specific NER (extracts organisms, tissues, etc.)
# - Higher accuracy than regex patterns
# - Confidence scores for intent classification
#
# These patterns will be removed in a future version.
# =============================================================================
TOOL_PATTERNS = [
    # Data scanning
    (r"(?:scan|find|look for)\s+(?:data|files?|samples?)", ToolName.SCAN_DATA),
    (r"(?:what\s+)?data\s+(?:is\s+)?available", ToolName.SCAN_DATA),
    
    # Database search
    (r"(?:search|query)\s+(?:for\s+)?(.+?)\s+(?:in|on|from)\s+(?:encode|geo|sra)", ToolName.SEARCH_DATABASES),
    (r"(?:search|query)\s+(?:encode|geo|sra)", ToolName.SEARCH_DATABASES),
    
    # TCGA
    (r"(?:search|query)\s+(?:tcga|cancer)\s+(?:for\s+)?(.+)", ToolName.SEARCH_TCGA),
    
    # Download
    (r"(?:download|get|fetch)\s+(GSE\d+|ENCSR[A-Z0-9]+)", ToolName.DOWNLOAD_DATASET),
    
    # Workflow
    (r"(?:create|generate|build|make)\s+(?:a\s+)?(?:.*?)workflow", ToolName.GENERATE_WORKFLOW),
    (r"list\s+(?:available\s+)?workflows?", ToolName.LIST_WORKFLOWS),
    (r"(?:check|verify)\s+references?", ToolName.CHECK_REFERENCES),
    
    # Execution
    (r"(?:run|execute|submit)\s+(?:the\s+)?(?:workflow|pipeline)", ToolName.SUBMIT_JOB),
    (r"(?:check|show|what)\s+(?:job\s+)?status", ToolName.GET_JOB_STATUS),
    (r"(?:show|get)\s+logs?", ToolName.GET_LOGS),
    (r"cancel\s+(?:the\s+)?job", ToolName.CANCEL_JOB),
    
    # Diagnostics
    (r"(?:diagnose|debug|troubleshoot)\s+(?:this\s+)?error", ToolName.DIAGNOSE_ERROR),
    (r"(?:analyze|interpret)\s+(?:the\s+)?results?", ToolName.ANALYZE_RESULTS),
    
    # Education
    (r"(?:what\s+is|explain|describe)\s+(.+)", ToolName.EXPLAIN_CONCEPT),
    (r"compare\s+(?:samples?|conditions?)", ToolName.COMPARE_SAMPLES),
    
    # Help
    (r"^(?:help|commands?|\?)$", ToolName.SHOW_HELP),
]


@dataclass
class ToolResult:
    """Result from a tool invocation."""
    success: bool
    tool_name: str
    data: Any = None
    message: str = ""
    error: Optional[str] = None
    ui_update: Optional[Dict[str, Any]] = None
    
    @classmethod
    def success_result(cls, tool_name: str, message: str, data: Any = None) -> "ToolResult":
        """Create a success result."""
        return cls(success=True, tool_name=tool_name, message=message, data=data)
    
    @classmethod
    def error_result(cls, tool_name: str, error: str) -> "ToolResult":
        """Create an error result."""
        return cls(success=False, tool_name=tool_name, error=error, message=f"❌ {error}")


@dataclass
class ToolParameter:
    """Definition of a tool parameter."""
    name: str
    type: str  # "string", "boolean", "integer", "array", "object"
    description: str
    required: bool = False
    default: Any = None
    enum: Optional[List[str]] = None


@dataclass
class RegisteredTool:
    """A registered tool with metadata."""
    name: str
    description: str
    parameters: List[ToolParameter]
    patterns: List[str]  # Regex patterns for detection
    handler: callable
    category: str = "general"
    
    def to_openai_format(self) -> Dict[str, Any]:
        """Convert to OpenAI function calling format."""
        properties = {}
        required = []
        
        for param in self.parameters:
            prop = {
                "type": param.type,
                "description": param.description,
            }
            if param.enum:
                prop["enum"] = param.enum
            if param.default is not None:
                prop["default"] = param.default
            properties[param.name] = prop
            
            if param.required:
                required.append(param.name)
        
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                }
            }
        }
    
    def matches(self, message: str) -> Optional[Dict[str, str]]:
        """Check if message matches any pattern, return extracted args."""
        import re
        message_lower = message.lower()
        
        for pattern in self.patterns:
            match = re.search(pattern, message_lower, re.IGNORECASE)
            if match:
                # Return captured groups as arguments
                groups = match.groups()
                if groups:
                    # First group is typically the main argument
                    return {"match": groups[0] if groups[0] else ""}
                return {}
        return None
    
    def validate_arguments(self, args: Dict[str, Any]) -> tuple[bool, str]:
        """
        Validate arguments against parameter definitions.
        
        Returns:
            (is_valid, error_message)
        """
        # Check required parameters
        for param in self.parameters:
            if param.required and param.name not in args:
                return False, f"Missing required parameter: {param.name}"
        
        # Type validation
        for name, value in args.items():
            param = next((p for p in self.parameters if p.name == name), None)
            if param:
                if param.type == "string" and not isinstance(value, str):
                    # Try to convert
                    args[name] = str(value) if value is not None else ""
                elif param.type == "integer":
                    try:
                        args[name] = int(value)
                    except (ValueError, TypeError):
                        return False, f"Parameter '{name}' must be an integer"
                elif param.type == "boolean":
                    if isinstance(value, str):
                        args[name] = value.lower() in ("true", "1", "yes")
                    else:
                        args[name] = bool(value)
                        
                # Enum validation
                if param.enum and args[name] not in param.enum:
                    return False, f"Parameter '{name}' must be one of: {', '.join(param.enum)}"
        
        return True, ""


def validate_path(path: Optional[str]) -> tuple[bool, str]:
    """Validate a file/directory path."""
    if not path:
        return True, ""  # Optional paths are OK
    
    import os
    path = os.path.expanduser(path)
    
    # Security checks
    if ".." in path:
        return False, "Path traversal not allowed"
    if path.startswith("/etc") or path.startswith("/root"):
        return False, "Access to system directories not allowed"
    
    return True, path


def validate_dataset_id(dataset_id: str) -> tuple[bool, str]:
    """Validate a dataset ID (GSE, ENCSR, TCGA)."""
    import re
    
    patterns = [
        r'^GSE\d+$',      # GEO
        r'^ENCSR[A-Z0-9]+$',  # ENCODE
        r'^TCGA-[A-Z]+$',  # TCGA project
    ]
    
    for pattern in patterns:
        if re.match(pattern, dataset_id, re.IGNORECASE):
            return True, dataset_id.upper()
    
    return False, f"Invalid dataset ID: {dataset_id}. Expected GSE#, ENCSR#, or TCGA-#"
