"""
Unified Agent
=============

The main entry point for the BioPipelines AI agent system.
Combines AutonomousAgent orchestration with AgentTools execution.

This module bridges two systems:
1. AutonomousAgent: Task classification, permission control, multi-step reasoning
2. AgentTools: Actual tool implementations (29 tools across 6 categories)

Architecture:
    User Query
        ↓
    UnifiedAgent.process_query()
        ↓
    AutonomousAgent.classify_task()    → Determine task type
        ↓
    PermissionManager.check_permission() → Check if allowed
        ↓
    AgentTools.execute_tool()          → Execute the tool
        ↓
    CommandSandbox (if shell command)  → Safety layer
        ↓
    AuditLogger                        → Log everything
        ↓
    AgentMemory (optional)             → Persist context

Usage:
    from workflow_composer.agents import UnifiedAgent, AutonomyLevel
    
    # Create agent with desired permission level
    agent = UnifiedAgent(autonomy_level=AutonomyLevel.ASSISTED)
    
    # Process a query
    response = await agent.process("scan /data/raw for FASTQ files")
    
    # Or synchronously
    response = agent.process_sync("what jobs are running?")

Example with different autonomy levels:
    # Read-only mode (safe for demos)
    agent = UnifiedAgent(autonomy_level=AutonomyLevel.READONLY)
    
    # Full autonomy (for trusted environments)
    agent = UnifiedAgent(autonomy_level=AutonomyLevel.AUTONOMOUS)
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable

# Executor layer
from .executor import (
    CommandSandbox,
    PermissionManager,
    AutonomyLevel,
    AuditLogger,
    ExecutionResult,
)

# Tools layer
from .tools import get_agent_tools, AgentTools, ToolResult, ToolName

# Autonomous components (lazy loaded)
from .autonomous import HealthChecker, RecoveryManager, JobMonitor

logger = logging.getLogger(__name__)


# =============================================================================
# Types
# =============================================================================

class TaskType(Enum):
    """Classification of user queries."""
    WORKFLOW = "workflow"          # Generate/manage workflows
    DIAGNOSIS = "diagnosis"        # Error diagnosis and recovery
    DATA = "data"                  # Data discovery and management
    JOB = "job"                    # Job submission and monitoring
    ANALYSIS = "analysis"          # Result analysis
    EDUCATION = "education"        # Explain concepts
    CODING = "coding"              # Generate code
    SYSTEM = "system"              # System health, vLLM restart
    GENERAL = "general"            # General questions


class ResponseType(Enum):
    """Type of response."""
    SUCCESS = "success"
    ERROR = "error"
    NEEDS_APPROVAL = "needs_approval"
    PARTIAL = "partial"            # Multi-step, still in progress
    QUESTION = "question"          # Agent needs more info


@dataclass
class ToolExecution:
    """Record of a tool execution."""
    tool_name: str
    parameters: Dict[str, Any]
    result: ToolResult
    duration_ms: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class AgentResponse:
    """
    Response from the unified agent.
    
    Attributes:
        success: Whether the overall operation succeeded
        message: Human-readable response message
        response_type: Type of response (success, error, needs_approval, etc.)
        task_type: What kind of task was detected
        tool_executions: List of tools that were executed
        data: Structured data from tool results
        suggestions: Follow-up suggestions
        requires_approval: True if user approval is needed
        approval_request: Details of what needs approval
    """
    success: bool
    message: str
    response_type: ResponseType = ResponseType.SUCCESS
    task_type: Optional[TaskType] = None
    tool_executions: List[ToolExecution] = field(default_factory=list)
    data: Optional[Dict[str, Any]] = None
    suggestions: List[str] = field(default_factory=list)
    requires_approval: bool = False
    approval_request: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "success": self.success,
            "message": self.message,
            "response_type": self.response_type.value,
            "task_type": self.task_type.value if self.task_type else None,
            "tool_executions": [
                {
                    "tool": te.tool_name,
                    "parameters": te.parameters,
                    "success": te.result.success,
                    "duration_ms": te.duration_ms,
                }
                for te in self.tool_executions
            ],
            "data": self.data,
            "suggestions": self.suggestions,
            "requires_approval": self.requires_approval,
            "approval_request": self.approval_request,
        }


# =============================================================================
# Task Classification
# =============================================================================

# Keywords for task classification
TASK_KEYWORDS = {
    TaskType.WORKFLOW: [
        "workflow", "pipeline", "generate", "create workflow", "run pipeline",
        "nextflow", "snakemake", "rna-seq", "chip-seq", "atac-seq", "wgs",
        "variant calling", "alignment", "rnaseq", "chipseq", "atacseq",
    ],
    TaskType.DIAGNOSIS: [
        "error", "fail", "diagnose", "debug", "fix", "recover",
        "crash", "problem", "issue", "wrong", "broken", "not working",
    ],
    TaskType.DATA: [
        "scan", "find", "search", "data", "download", "dataset",
        "fastq", "bam", "vcf", "reference", "genome", "index",
        "tcga", "geo", "sra", "encode", "files", "samples",
    ],
    TaskType.JOB: [
        "job", "submit", "status", "running", "queue", "slurm",
        "cancel", "resubmit", "watch", "monitor", "pending",
    ],
    TaskType.ANALYSIS: [
        "analyze", "results", "compare", "visualize", "plot",
        "statistics", "quality", "metrics", "report",
    ],
    TaskType.EDUCATION: [
        "explain", "what is", "how does", "help", "tutorial",
        "understand", "learn", "concept", "definition",
    ],
    TaskType.SYSTEM: [
        "system health", "vllm", "restart", "server", "service",
        "gpu", "memory", "disk", "health check",
    ],
    TaskType.CODING: [
        "code", "script", "function", "implement", "write code",
        "python", "bash", "nextflow config", "snakemake rule",
    ],
}


def classify_task(query: str) -> TaskType:
    """
    Classify a user query into a task type.
    
    Args:
        query: User's natural language query
        
    Returns:
        TaskType enum value
    """
    query_lower = query.lower()
    
    # Priority keywords that override other classifications
    # Education keywords should take priority when asking "what is X"
    priority_education_patterns = ["explain", "what is", "how does", "understand", "learn"]
    for pattern in priority_education_patterns:
        if pattern in query_lower:
            return TaskType.EDUCATION
    
    # Count keyword matches for each type
    scores = {}
    for task_type, keywords in TASK_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in query_lower)
        if score > 0:
            scores[task_type] = score
            
    if not scores:
        return TaskType.GENERAL
        
    # Return type with highest score
    return max(scores.items(), key=lambda x: x[1])[0]


# =============================================================================
# Permission Policies for Tools
# =============================================================================

# Map tool categories to permission actions
TOOL_PERMISSION_MAPPING = {
    # Read-only tools (always allowed)
    ToolName.SCAN_DATA: "read",
    ToolName.SEARCH_DATABASES: "read",
    ToolName.SEARCH_TCGA: "read",
    ToolName.DESCRIBE_FILES: "read",
    ToolName.VALIDATE_DATASET: "read",
    ToolName.LIST_WORKFLOWS: "read",
    ToolName.CHECK_REFERENCES: "read",
    ToolName.GET_JOB_STATUS: "read",
    ToolName.GET_LOGS: "read",
    ToolName.CHECK_SYSTEM_HEALTH: "read",
    ToolName.WATCH_JOB: "read",
    ToolName.LIST_JOBS: "read",
    ToolName.EXPLAIN_CONCEPT: "read",
    ToolName.COMPARE_SAMPLES: "read",
    ToolName.SHOW_HELP: "read",
    ToolName.DIAGNOSE_ERROR: "read",
    ToolName.ANALYZE_RESULTS: "read",
    ToolName.VISUALIZE_WORKFLOW: "read",
    ToolName.MONITOR_JOBS: "read",
    ToolName.RUN_COMMAND: "execute",
    
    # Write/execute tools (may need approval)
    ToolName.DOWNLOAD_DATASET: "write",
    ToolName.DOWNLOAD_REFERENCE: "write",
    ToolName.BUILD_INDEX: "execute",
    ToolName.GENERATE_WORKFLOW: "write",
    ToolName.SUBMIT_JOB: "execute",
    ToolName.CANCEL_JOB: "execute",
    ToolName.RESUBMIT_JOB: "execute",
    ToolName.RESTART_VLLM: "execute",
    ToolName.RECOVER_ERROR: "execute",
    ToolName.DOWNLOAD_RESULTS: "write",
    
    # Dangerous tools (always need approval)
    ToolName.CLEANUP_DATA: "delete",
    ToolName.CONFIRM_CLEANUP: "delete",
}

# Permissions per autonomy level
LEVEL_PERMISSIONS = {
    AutonomyLevel.READONLY: {"read"},
    AutonomyLevel.MONITORED: {"read"},
    AutonomyLevel.ASSISTED: {"read", "write"},  # execute needs approval
    AutonomyLevel.SUPERVISED: {"read", "write", "execute"},
    AutonomyLevel.AUTONOMOUS: {"read", "write", "execute", "delete"},
}


# =============================================================================
# Unified Agent
# =============================================================================

class UnifiedAgent:
    """
    The unified entry point for the BioPipelines AI agent.
    
    Combines:
    - AutonomousAgent-style orchestration (task classification, permissions)
    - AgentTools execution (actual tool implementations)
    - Executor layer (sandbox, audit, permissions)
    
    Features:
    - Permission-based access control
    - Sandboxed command execution
    - Complete audit trail
    - Multi-step task support
    - Human-in-the-loop approval workflow
    """
    
    def __init__(
        self,
        autonomy_level: AutonomyLevel = AutonomyLevel.ASSISTED,
        workspace_root: Optional[Path] = None,
        enable_audit: bool = True,
        approval_callback: Optional[Callable] = None,
    ):
        """
        Initialize the unified agent.
        
        Args:
            autonomy_level: Permission level for the agent
            workspace_root: Root directory for file operations
            enable_audit: Whether to enable audit logging
            approval_callback: Function to call for approvals (async)
        """
        self.autonomy_level = autonomy_level
        self.workspace_root = workspace_root or Path.cwd()
        
        # Initialize permission manager
        self.permissions = PermissionManager(
            autonomy_level=autonomy_level,
            approval_callback=approval_callback,
        )
        
        # Initialize command sandbox
        self.sandbox = CommandSandbox(
            workspace_root=self.workspace_root,
        )
        
        # Initialize audit logger
        self.audit = AuditLogger() if enable_audit else None
        if self.audit:
            self.sandbox.audit_logger = self.audit
            
        # Initialize tools (with permission integration)
        self._tools: Optional[AgentTools] = None
        
        # Lazy-loaded autonomous components
        self._health_checker: Optional[HealthChecker] = None
        self._recovery_manager: Optional[RecoveryManager] = None
        self._job_monitor: Optional[JobMonitor] = None
        
        # Execution history
        self._history: List[AgentResponse] = []
        
        logger.info(f"UnifiedAgent initialized with autonomy level: {autonomy_level.value}")
        
    @property
    def tools(self) -> AgentTools:
        """Get or initialize the tools."""
        if self._tools is None:
            self._tools = get_agent_tools()
        return self._tools
        
    @property
    def health_checker(self) -> HealthChecker:
        """Get or initialize the health checker."""
        if self._health_checker is None:
            self._health_checker = HealthChecker()
        return self._health_checker
        
    @property
    def recovery_manager(self) -> RecoveryManager:
        """Get or initialize the recovery manager."""
        if self._recovery_manager is None:
            self._recovery_manager = RecoveryManager()
        return self._recovery_manager
        
    @property
    def job_monitor(self) -> JobMonitor:
        """Get or initialize the job monitor."""
        if self._job_monitor is None:
            self._job_monitor = JobMonitor()
        return self._job_monitor
        
    # =========================================================================
    # Permission Checking
    # =========================================================================
    
    def check_tool_permission(self, tool_name: ToolName) -> Dict[str, Any]:
        """
        Check if a tool is allowed at the current autonomy level.
        
        Args:
            tool_name: The tool to check
            
        Returns:
            Dict with 'allowed', 'requires_approval', and 'reason'
        """
        # Get the permission category for this tool
        permission_category = TOOL_PERMISSION_MAPPING.get(tool_name, "execute")
        
        # Check if this category is allowed at current level
        allowed_categories = LEVEL_PERMISSIONS.get(
            self.autonomy_level, 
            {"read"}
        )
        
        if permission_category in allowed_categories:
            return {
                "allowed": True,
                "requires_approval": False,
                "reason": f"Tool '{tool_name.value}' is allowed at {self.autonomy_level.value} level",
            }
        elif permission_category == "execute" and self.autonomy_level == AutonomyLevel.ASSISTED:
            # Execution needs approval at ASSISTED level
            return {
                "allowed": True,
                "requires_approval": True,
                "reason": f"Tool '{tool_name.value}' requires approval at {self.autonomy_level.value} level",
            }
        else:
            return {
                "allowed": False,
                "requires_approval": False,
                "reason": f"Tool '{tool_name.value}' ({permission_category}) not allowed at {self.autonomy_level.value} level",
            }
            
    # =========================================================================
    # Tool Execution
    # =========================================================================
    
    def execute_tool(
        self,
        tool_name: Union[str, ToolName],
        **kwargs
    ) -> ToolExecution:
        """
        Execute a tool with permission checking and auditing.
        
        Args:
            tool_name: Name of the tool to execute
            **kwargs: Tool parameters
            
        Returns:
            ToolExecution record
        """
        start_time = datetime.now()
        
        # Normalize tool name
        if isinstance(tool_name, str):
            try:
                tool_name = ToolName(tool_name)
            except ValueError:
                # Handle unknown tools
                result = ToolResult(
                    success=False,
                    tool_name=str(tool_name),
                    error=f"Unknown tool: {tool_name}",
                    message=f"❌ Unknown tool: {tool_name}. Use 'help' to see available tools.",
                )
                return ToolExecution(
                    tool_name=str(tool_name),
                    parameters=kwargs,
                    result=result,
                    duration_ms=0,
                )
                
        # Check permission
        perm = self.check_tool_permission(tool_name)
        
        if not perm["allowed"]:
            result = ToolResult(
                success=False,
                tool_name=tool_name.value,
                error=perm["reason"],
                message=f"❌ {perm['reason']}. Request elevated permissions (current: {self.autonomy_level.value}).",
            )
            return ToolExecution(
                tool_name=tool_name.value,
                parameters=kwargs,
                result=result,
                duration_ms=0,
            )
            
        if perm["requires_approval"]:
            # Mark for approval but don't block (approval handled at higher level)
            logger.info(f"Tool {tool_name.value} requires approval")
            
        # Audit the call
        if self.audit:
            self.audit.log_tool_call(tool_name.value, kwargs)
            
        # Execute the tool
        try:
            result = self.tools.execute_tool(tool_name.value, **kwargs)
        except Exception as e:
            logger.error(f"Tool execution failed: {e}")
            result = ToolResult(
                success=False,
                error=str(e),
                data={},
                suggestions=["Check the error message", "Try with different parameters"],
            )
            
        # Calculate duration
        duration_ms = (datetime.now() - start_time).total_seconds() * 1000
        
        # Audit the result
        if self.audit:
            self.audit.log_tool_result(tool_name.value, result.success, result.data)
            
        return ToolExecution(
            tool_name=tool_name.value,
            parameters=kwargs,
            result=result,
            duration_ms=duration_ms,
        )
        
    # =========================================================================
    # Query Processing
    # =========================================================================
    
    async def process_query(self, query: str) -> AgentResponse:
        """
        Process a user query asynchronously.
        
        This is the main entry point for the agent.
        
        Args:
            query: User's natural language query
            
        Returns:
            AgentResponse with results
        """
        logger.info(f"Processing query: {query[:100]}...")
        
        # Classify the task
        task_type = classify_task(query)
        logger.info(f"Task classified as: {task_type.value}")
        
        # Detect tool from query using AgentTools
        detected_tool_name = self.tools.detect_tool(query)
        detected_tool = None
        
        if detected_tool_name:
            try:
                detected_tool = ToolName(detected_tool_name)
            except ValueError:
                logger.warning(f"Unknown tool name: {detected_tool_name}")
        
        if detected_tool:
            logger.info(f"Detected tool: {detected_tool.value}")
            
            # Check permission
            perm = self.check_tool_permission(detected_tool)
            
            if not perm["allowed"]:
                return AgentResponse(
                    success=False,
                    message=perm["reason"],
                    response_type=ResponseType.ERROR,
                    task_type=task_type,
                    suggestions=[
                        "Request elevated access",
                        "Try a read-only query instead",
                    ],
                )
                
            if perm["requires_approval"]:
                return AgentResponse(
                    success=True,
                    message=f"This action requires approval: {detected_tool.value}",
                    response_type=ResponseType.NEEDS_APPROVAL,
                    task_type=task_type,
                    requires_approval=True,
                    approval_request={
                        "tool": detected_tool.value,
                        "query": query,
                        "reason": perm["reason"],
                    },
                    suggestions=["Approve or deny this action"],
                )
                
            # Extract parameters and execute
            params = self._extract_parameters(query, detected_tool)
            execution = self.execute_tool(detected_tool, **params)
            
            # Build response
            response = self._build_response(
                task_type=task_type,
                executions=[execution],
                query=query,
            )
            
        else:
            # No specific tool detected - handle as general query
            response = await self._handle_general_query(query, task_type)
            
        # Save to history
        self._history.append(response)
        
        return response
        
    def process_sync(self, query: str) -> AgentResponse:
        """
        Process a query synchronously.
        
        Convenience method for non-async contexts.
        
        Args:
            query: User's natural language query
            
        Returns:
            AgentResponse with results
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're in an async context, create a new task
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(asyncio.run, self.process_query(query))
                    return future.result()
            else:
                return loop.run_until_complete(self.process_query(query))
        except RuntimeError:
            # No event loop, create one
            return asyncio.run(self.process_query(query))
            
    # =========================================================================
    # Internal Methods
    # =========================================================================
    
    def _extract_parameters(self, query: str, tool: ToolName) -> Dict[str, Any]:
        """Extract tool parameters from the query."""
        params = {}
        query_lower = query.lower()
        
        # Handle education tools - extract the concept
        if tool == ToolName.EXPLAIN_CONCEPT:
            concept = query_lower
            # Remove common question prefixes
            for prefix in ["explain what", "what is", "explain", "how does", "describe", "tell me about"]:
                concept = concept.replace(prefix, "").strip()
            # Remove trailing " is" or " are"
            concept = concept.rstrip(" is").rstrip(" are").strip()
            # Remove question marks
            concept = concept.rstrip("?").strip()
            params["concept"] = concept
            return params
            
        if tool == ToolName.COMPARE_SAMPLES:
            # Extract comparison targets
            params["samples"] = query  # Let the tool handle parsing
            return params
        
        # Common path extraction
        import re
        path_match = re.search(r'(/[\w/.-]+)', query)
        if path_match:
            path = path_match.group(1)
            # Map to appropriate parameter based on tool
            if tool in [ToolName.SCAN_DATA, ToolName.DESCRIBE_FILES]:
                params["path"] = path
            elif tool in [ToolName.DOWNLOAD_REFERENCE]:
                params["output_dir"] = path
            elif tool in [ToolName.SUBMIT_JOB, ToolName.GENERATE_WORKFLOW]:
                params["workflow_dir"] = path
                
        # Extract job IDs
        job_match = re.search(r'\b(\d{6,})\b', query)
        if job_match and tool in [ToolName.GET_JOB_STATUS, ToolName.CANCEL_JOB, 
                                   ToolName.WATCH_JOB, ToolName.RESUBMIT_JOB]:
            params["job_id"] = job_match.group(1)
            
        # Extract analysis types - only for workflow tools
        if tool in [ToolName.GENERATE_WORKFLOW]:
            for analysis_type in ["rna-seq", "chip-seq", "atac-seq", "wgs", "wes"]:
                if analysis_type in query_lower or analysis_type.replace("-", "") in query_lower:
                    params["analysis_type"] = analysis_type.replace("-", "_")
                    break
                
            # Extract organism - only for workflow tools
            for organism in ["human", "mouse", "rat", "zebrafish"]:
                if organism in query_lower:
                    params["organism"] = organism
                    break
                
        return params
        
    def _build_response(
        self,
        task_type: TaskType,
        executions: List[ToolExecution],
        query: str,
    ) -> AgentResponse:
        """Build an AgentResponse from tool executions."""
        if not executions:
            return AgentResponse(
                success=False,
                message="No tools were executed",
                response_type=ResponseType.ERROR,
                task_type=task_type,
            )
            
        # Check if all executions succeeded
        all_success = all(e.result.success for e in executions)
        
        # Combine messages
        messages = []
        suggestions = []
        data = {}
        
        for execution in executions:
            if execution.result.success:
                # Use message if available, otherwise summarize data
                if execution.result.message:
                    messages.append(execution.result.message)
                elif execution.result.data:
                    messages.append(f"{execution.tool_name}: completed successfully")
                    data[execution.tool_name] = execution.result.data
            else:
                messages.append(f"{execution.tool_name} failed: {execution.result.error}")
                
            # ToolResult may not have suggestions attribute
            if hasattr(execution.result, 'suggestions') and execution.result.suggestions:
                suggestions.extend(execution.result.suggestions)
            
        # Deduplicate suggestions
        suggestions = list(dict.fromkeys(suggestions))
        
        return AgentResponse(
            success=all_success,
            message="\n".join(messages) if messages else "Operation completed",
            response_type=ResponseType.SUCCESS if all_success else ResponseType.ERROR,
            task_type=task_type,
            tool_executions=executions,
            data=data if data else None,
            suggestions=suggestions[:5],  # Limit suggestions
        )
        
    async def _handle_general_query(
        self, 
        query: str, 
        task_type: TaskType
    ) -> AgentResponse:
        """Handle queries that don't map to a specific tool."""
        
        # Route based on task type
        if task_type == TaskType.EDUCATION:
            # Try to explain a concept - extract the concept from the query
            # Remove "explain", "what is", etc. to get the concept
            concept = query.lower()
            for prefix in ["explain what", "explain", "what is", "how does", "describe"]:
                concept = concept.replace(prefix, "").strip()
            execution = self.execute_tool(ToolName.EXPLAIN_CONCEPT, concept=concept)
            return self._build_response(task_type, [execution], query)
            
        elif task_type == TaskType.SYSTEM:
            # Check system health
            execution = self.execute_tool(ToolName.CHECK_SYSTEM_HEALTH)
            return self._build_response(task_type, [execution], query)
            
        elif task_type == TaskType.JOB:
            # List jobs if no specific job mentioned
            execution = self.execute_tool(ToolName.LIST_JOBS)
            return self._build_response(task_type, [execution], query)
            
        else:
            # Return help
            execution = self.execute_tool(ToolName.SHOW_HELP)
            return self._build_response(task_type, [execution], query)
            
    # =========================================================================
    # Approval Workflow
    # =========================================================================
    
    def approve_action(self, request_id: str, approver: str = "user") -> bool:
        """Approve a pending action."""
        return self.permissions.approve(request_id, approver)
        
    def deny_action(self, request_id: str, denier: str = "user") -> bool:
        """Deny a pending action."""
        return self.permissions.deny(request_id, denier)
        
    def get_pending_approvals(self) -> List[Dict[str, Any]]:
        """Get all pending approval requests."""
        return [
            {
                "id": r.id,
                "action": r.action,
                "description": r.description,
                "timestamp": r.timestamp,
            }
            for r in self.permissions.get_pending_approvals()
        ]
        
    # =========================================================================
    # Configuration
    # =========================================================================
    
    def set_autonomy_level(self, level: AutonomyLevel):
        """Change the autonomy level."""
        old_level = self.autonomy_level
        self.autonomy_level = level
        self.permissions.autonomy_level = level
        logger.info(f"Autonomy level changed: {old_level.value} → {level.value}")
        
    def get_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent execution history."""
        return [r.to_dict() for r in self._history[-limit:]]
        
    def clear_history(self):
        """Clear execution history."""
        self._history.clear()


# =============================================================================
# Factory Functions
# =============================================================================

_default_agent: Optional[UnifiedAgent] = None


def get_agent(
    autonomy_level: AutonomyLevel = AutonomyLevel.ASSISTED,
) -> UnifiedAgent:
    """
    Get or create the default unified agent.
    
    Args:
        autonomy_level: Permission level (only used on first call)
        
    Returns:
        UnifiedAgent instance
    """
    global _default_agent
    if _default_agent is None:
        _default_agent = UnifiedAgent(autonomy_level=autonomy_level)
    return _default_agent


def reset_agent():
    """Reset the default agent (for testing)."""
    global _default_agent
    _default_agent = None


# =============================================================================
# Convenience Functions
# =============================================================================

async def process_query(query: str) -> AgentResponse:
    """Process a query using the default agent."""
    return await get_agent().process_query(query)


def process_query_sync(query: str) -> AgentResponse:
    """Process a query synchronously using the default agent."""
    return get_agent().process_sync(query)
