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

# Unified classification (Phase 1 refactoring)
from .classification import TaskType, classify_task as _classify_task_impl

# Intent parsing (hybrid parser integration)
from .intent import HybridQueryParser, QueryParseResult, ConversationContext, DialogueManager

# Observability - distributed tracing
from workflow_composer.infrastructure.observability import get_tracer, traced, get_metrics

# RAG system for tool selection and argument optimization
from .rag import RAGOrchestrator

logger = logging.getLogger(__name__)


# =============================================================================
# Intent-to-Tool Mapping
# =============================================================================

# Maps semantic intent names to ToolName enums
INTENT_TO_TOOL: Dict[str, ToolName] = {
    # Data Discovery
    "DATA_SCAN": ToolName.SCAN_DATA,
    "DATA_SEARCH": ToolName.SEARCH_DATABASES,
    "DATA_DOWNLOAD": ToolName.DOWNLOAD_DATASET,
    
    # Workflow Operations
    "WORKFLOW_CREATE": ToolName.GENERATE_WORKFLOW,
    "WORKFLOW_VISUALIZE": ToolName.VISUALIZE_WORKFLOW,
    
    # Job Operations
    "JOB_SUBMIT": ToolName.SUBMIT_JOB,
    "JOB_STATUS": ToolName.GET_JOB_STATUS,
    "JOB_LOGS": ToolName.GET_LOGS,
    
    # Diagnostics
    "DIAGNOSE_ERROR": ToolName.DIAGNOSE_ERROR,
    "ANALYSIS_INTERPRET": ToolName.ANALYZE_RESULTS,
    
    # Reference Management
    "REFERENCE_CHECK": ToolName.CHECK_REFERENCES,
    "REFERENCE_DOWNLOAD": ToolName.DOWNLOAD_REFERENCE,
    
    # Education
    "EDUCATION_EXPLAIN": ToolName.EXPLAIN_CONCEPT,
    "EDUCATION_HELP": ToolName.SHOW_HELP,
    
    # Data Description
    "DATA_DESCRIBE": ToolName.DESCRIBE_FILES,
    
    # Context-Aware (use previous results)
    "CONTEXT_RECALL": None,  # Handled specially - recalls previous search results
    "CONTEXT_METADATA": None,  # Handled specially - gets metadata from context
    
    # Data Description
    "DATA_DESCRIBE": ToolName.DESCRIBE_FILES,
    
    # Context-Aware (use previous results)
    "CONTEXT_RECALL": None,  # Handled specially - recalls previous search results
    "CONTEXT_METADATA": None,  # Handled specially - gets metadata from context
    
    # Meta/Composite (handled specially)
    "META_CONFIRM": "_handle_confirm",  # Special handler for confirmations
    "META_CANCEL": None,
    "META_GREETING": ToolName.SHOW_HELP,
    "META_UNKNOWN": ToolName.SHOW_HELP,
}


# =============================================================================
# Types (TaskType imported from classification.py)
# =============================================================================

# TaskType is now imported from classification.py for single source of truth
# Re-exported here for backward compatibility


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
# Task Classification (delegated to classification.py)
# =============================================================================

# TASK_KEYWORDS kept for backward compatibility - tests may reference it
# The actual classification logic is now in classification.py
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
    
    This function now delegates to classification.py for the actual logic,
    providing a single source of truth while maintaining backward compatibility.
    
    Args:
        query: User's natural language query
        
    Returns:
        TaskType enum value
    """
    return _classify_task_impl(query)


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
    ToolName.GET_DATASET_DETAILS: "read",  # New: dataset details is read-only
    
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
        orchestrator_preset: Optional[str] = None,
    ):
        """
        Initialize the unified agent.
        
        Args:
            autonomy_level: Permission level for the agent
            workspace_root: Root directory for file operations
            enable_audit: Whether to enable audit logging
            approval_callback: Function to call for approvals (async)
            orchestrator_preset: Preset for LLM orchestrator ("development", "production", "critical")
        """
        self.autonomy_level = autonomy_level
        self.workspace_root = workspace_root or Path.cwd()
        self._orchestrator_preset = orchestrator_preset
        
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
        self._orchestrator = None
        
        # RAG orchestrator for tool selection and argument optimization
        self._rag_orchestrator: Optional[RAGOrchestrator] = None
        
        # Hybrid intent parser (lazy loaded)
        self._query_parser: Optional[HybridQueryParser] = None
        
        # Conversation context for multi-turn memory
        self._context: Optional[ConversationContext] = None
        self._dialogue_manager: Optional[DialogueManager] = None
        
        # Execution history
        self._history: List[AgentResponse] = []
        
        # Last search results for "download all" support (deprecated - use context)
        self._last_search_results: List[str] = []
        
        logger.info(f"UnifiedAgent initialized with autonomy level: {autonomy_level.value}")
        
    @property
    def tools(self) -> AgentTools:
        """Get or initialize the tools."""
        if self._tools is None:
            self._tools = get_agent_tools()
        return self._tools
    
    @property
    def rag(self) -> RAGOrchestrator:
        """
        Get or initialize the RAG orchestrator.
        
        Provides:
        - Tool selection boosting based on past successes
        - Argument optimization from similar queries
        - Execution recording for learning
        """
        if self._rag_orchestrator is None:
            try:
                self._rag_orchestrator = RAGOrchestrator()
                # Warm up RAG cache in background
                self._rag_orchestrator.warm_up()
                logger.info("RAGOrchestrator initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize RAGOrchestrator: {e}")
                self._rag_orchestrator = None
        return self._rag_orchestrator
        
    @property
    def query_parser(self) -> HybridQueryParser:
        """Get or initialize the hybrid query parser."""
        if self._query_parser is None:
            try:
                self._query_parser = HybridQueryParser()
                logger.info("HybridQueryParser initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize HybridQueryParser: {e}")
                self._query_parser = None
        return self._query_parser
    
    @property
    def context(self) -> ConversationContext:
        """
        Get or initialize the conversation context.
        
        Provides multi-turn memory including:
        - Entity tracking across turns
        - Coreference resolution ("it", "that", "the data")
        - Working memory for active tasks
        - State management (last search results, current workflow, etc.)
        """
        if self._context is None:
            self._context = ConversationContext(session_id=f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            logger.info(f"ConversationContext initialized: {self._context.session_id}")
        return self._context
    
    @property
    def dialogue_manager(self) -> DialogueManager:
        """
        Get or initialize the dialogue manager.
        
        Coordinates:
        - Intent parsing with context
        - Coreference resolution
        - Task state management
        - Slot filling for incomplete requests
        """
        if self._dialogue_manager is None:
            self._dialogue_manager = DialogueManager(
                intent_parser=None,  # Uses its own parser
                context=self.context
            )
            logger.info("DialogueManager initialized")
        return self._dialogue_manager
    
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
    
    @property
    def orchestrator(self):
        """
        Get or initialize the LLM orchestrator for smart model routing.
        
        The orchestrator provides:
        - Automatic local/cloud model selection
        - Cost-aware routing
        - Fallback handling
        - Ensemble for critical tasks
        
        Returns:
            ModelOrchestrator instance
            
        Example:
            # Use orchestrator for LLM calls
            response = await agent.orchestrator.complete("Generate workflow")
            print(f"Used: {response.provider}, Cost: ${response.cost:.4f}")
        """
        if self._orchestrator is None:
            from ..llm import get_orchestrator
            self._orchestrator = get_orchestrator(preset=self._orchestrator_preset)
        return self._orchestrator
        
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
            
            # Store search results for "download all" support
            if tool_name == ToolName.SEARCH_DATABASES and result.success:
                self._store_search_results(result)
            
            # Record successful execution in RAG system for learning
            if result.success and self.rag:
                try:
                    self.rag.record_execution(
                        query=str(kwargs.get('query', '')),
                        tool_name=tool_name.value,
                        tool_args=kwargs,
                        success=True,
                        duration_ms=(datetime.now() - start_time).total_seconds() * 1000,
                        result_count=len(result.data) if isinstance(result.data, (list, dict)) else 1,
                    )
                except Exception as e:
                    logger.debug(f"RAG recording failed: {e}")
                
        except Exception as e:
            logger.error(f"Tool execution failed: {e}")
            result = ToolResult(
                success=False,
                error=str(e),
                data={},
                suggestions=["Check the error message", "Try with different parameters"],
            )
            
            # Record failure in RAG system for learning
            if self.rag:
                try:
                    self.rag.record_execution(
                        query=str(kwargs.get('query', '')),
                        tool_name=tool_name.value,
                        tool_args=kwargs,
                        success=False,
                        duration_ms=(datetime.now() - start_time).total_seconds() * 1000,
                        error_message=str(e),
                    )
                except Exception as mem_e:
                    logger.debug(f"RAG recording failed: {mem_e}")
            
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
        Uses hybrid intent parsing (semantic + pattern + NER) for robust detection.
        Maintains conversation context across turns.
        Fully instrumented with distributed tracing.
        
        Args:
            query: User's natural language query
            
        Returns:
            AgentResponse with results
        """
        # Create tracing span for the entire query processing
        tracer = get_tracer()
        metrics = get_metrics()
        
        with tracer.start_span(
            "process_query",
            tags={
                "query_length": len(query),
                "autonomy_level": self.autonomy_level.value,
            }
        ) as span:
            span.add_event("query_received", {"query_preview": query[:100]})
            metrics.counter("agent.queries.total")
            
            logger.info(f"Processing query: {query[:100]}...")
            
            # Classify the task
            with tracer.start_span("classify_task") as classify_span:
                task_type = classify_task(query)
                classify_span.add_tag("task_type", task_type.value)
            span.add_event("task_classified", {"task_type": task_type.value})
            logger.info(f"Task classified as: {task_type.value}")
            
            # Try hybrid intent parsing first (semantic + pattern + NER)
            detected_tool = None
            detected_args = []
            hybrid_params = None  # Params from hybrid parser (preferred)
            parse_result: Optional[QueryParseResult] = None
            extracted_entities = []  # For context tracking
            
            if self.query_parser:
                try:
                    with tracer.start_span("hybrid_parse") as parse_span:
                        parse_result = self.query_parser.parse(query)
                        parse_span.add_tag("intent", parse_result.intent)
                        parse_span.add_tag("confidence", parse_result.intent_confidence)
                    logger.info(f"HybridParser: intent={parse_result.intent}, confidence={parse_result.intent_confidence:.2f}")
                    
                    # Extract entities for context
                    extracted_entities = parse_result.entities if parse_result.entities else []
                    
                    # Map intent to tool if confidence is reasonable (0.35 threshold allows semantic matching)
                    if parse_result.intent_confidence >= 0.35 and parse_result.intent in INTENT_TO_TOOL:
                        detected_tool = INTENT_TO_TOOL[parse_result.intent]
                        
                        # Build clean params from slots and entities
                        hybrid_params = self._build_params_from_parse_result(parse_result, query)
                        detected_args = list(hybrid_params.values()) if hybrid_params else []
                        logger.info(f"Mapped intent '{parse_result.intent}' to tool: {detected_tool}, params: {hybrid_params}")
                    
                    # Handle context-aware intents specially
                    if parse_result.intent in ("CONTEXT_RECALL", "CONTEXT_METADATA"):
                        return await self._handle_context_query(parse_result.intent, query, task_type)
                        
                except Exception as e:
                    span.add_event("parser_fallback", {"error": str(e)})
                    logger.warning(f"HybridParser failed, falling back to regex: {e}")
        
        # Add user turn to conversation context
        # Note: We pass entities=[] because BioEntity and Entity have different structures
        # The context still tracks the message and intent
        self.context.add_turn(
            role="user",
            content=query,
            entities=[],  # BioEntity not compatible with Entity, store separately
            intent=parse_result.intent if parse_result else None
        )
        
        # Store extracted entities in context state for later use
        if extracted_entities:
            self.context.update_state("last_entities", [
                {"type": e.entity_type, "text": e.text, "canonical": e.canonical}
                for e in extracted_entities
            ])
        
        # Fallback to regex-based detection if hybrid parsing didn't find a tool
        if not detected_tool:
            detection = self.tools.detect_tool(query)
            if detection:
                detected_tool, detected_args = detection
                logger.info(f"Regex detected tool: {detected_tool.value} with args: {detected_args}")
        
        if detected_tool:
            logger.info(f"Final detected tool: {detected_tool.value} with args: {detected_args}")
            
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
                
            # Use hybrid params if available, else build from args, else extract from query
            if hybrid_params:
                params = hybrid_params
            else:
                params = self._build_params_from_args(detected_tool, detected_args)
                if not params:
                    params = self._extract_parameters(query, detected_tool)
            
            # RAG Enhancement: Optimize arguments based on past successful queries
            rag_enhanced = False
            if self.rag:
                try:
                    with tracer.start_span("rag_enhance") as rag_span:
                        enhancement = self.rag.enhance(
                            query=query,
                            candidate_tools=[detected_tool.value],
                            base_args=params
                        )
                        # Use RAG-suggested arguments (merged with user-provided)
                        if enhancement.suggested_args:
                            # User params override RAG suggestions
                            params = {**enhancement.suggested_args, **params}
                            rag_enhanced = True
                            rag_span.add_tag("args_enhanced", True)
                            logger.info(f"RAG enhanced params: {list(enhancement.suggested_args.keys())}")
                except Exception as e:
                    logger.debug(f"RAG enhancement failed (continuing without): {e}")
            
            # Execute the tool with tracing
            with tracer.start_span(
                "execute_tool",
                tags={"tool": detected_tool.value, "rag_enhanced": rag_enhanced}
            ) as tool_span:
                execution = self.execute_tool(detected_tool, **params)
                tool_span.add_tag("success", execution.result.success)
                if not execution.result.success:
                    tool_span.set_error(Exception(execution.result.message))
                metrics.counter(
                    "agent.tools.executed",
                    tags={"tool": detected_tool.value, "success": str(execution.result.success)}
                )
            
            # Build response
            response = self._build_response(
                task_type=task_type,
                executions=[execution],
                query=query,
            )
            span.add_tag("response_success", response.success)
            
        else:
            # No specific tool detected - handle as general query
            span.add_event("general_query", {"reason": "no_tool_detected"})
            response = await self._handle_general_query(query, task_type)
        
        # Record final metrics
        if span.end_time:
            metrics.histogram("agent.query.duration_ms", span.duration_ms)
        
        # Add assistant response to conversation context
        self.context.add_turn(
            role="assistant",
            content=response.message[:500] if response.message else "",  # Truncate long messages
            entities=[],  # Could extract entities from response if needed
            intent=parse_result.intent if parse_result else None
        )
            
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
    
    def _build_params_from_parse_result(
        self, 
        parse_result: QueryParseResult, 
        original_query: str
    ) -> Dict[str, Any]:
        """
        Build clean tool parameters from parse result.
        
        Uses entities to construct a clean search query instead of raw user input.
        Deduplicates terms to avoid overly restrictive queries.
        """
        params = {}
        slots = parse_result.slots
        entities = parse_result.entities
        intent = parse_result.intent
        
        # For data search, build query from entities if available
        if intent == "DATA_SEARCH" and entities:
            # Build a clean query from entities - deduplicate!
            seen_terms = set()
            query_parts = []
            for entity in entities:
                if entity.entity_type in ("ORGANISM", "TISSUE", "DISEASE", "ASSAY_TYPE"):
                    term = (entity.canonical or entity.text).lower()
                    # Skip duplicates (case-insensitive)
                    if term not in seen_terms:
                        seen_terms.add(term)
                        query_parts.append(entity.canonical or entity.text)
            if query_parts:
                params["query"] = " ".join(query_parts)
            elif "query" in slots:
                params["query"] = slots["query"]
        elif intent == "DATA_DOWNLOAD":
            # For download, check for dataset IDs or "download all"
            if "dataset_id" in slots:
                params["dataset_id"] = slots["dataset_id"]
            else:
                # Check entities for dataset IDs
                for entity in entities:
                    if entity.entity_type == "DATASET_ID":
                        params["dataset_id"] = entity.text
                        break
                        
            # Check for "download all" / "execute commands" / "run downloads"
            if not params.get("dataset_id"):
                query_lower = original_query.lower()
                # Extended patterns for "download all" and "execute commands"
                download_all_patterns = [
                    "all" in query_lower,
                    "everything" in query_lower,
                    "both" in query_lower,
                    "execute" in query_lower and ("command" in query_lower or "download" in query_lower),
                    "run" in query_lower and ("command" in query_lower or "download" in query_lower),
                    slots.get("download_all") is True,  # From intent pattern
                ]
                
                if any(download_all_patterns):
                    # Try context first, then fall back to instance variable
                    context_ids = self.context.get_state("last_search_ids")
                    if context_ids:
                        params["dataset_ids"] = context_ids.copy()
                        params["download_all"] = True
                        logger.info(f"Using {len(context_ids)} dataset IDs from context memory")
                    elif self._last_search_results:
                        params["dataset_ids"] = self._last_search_results.copy()
                        params["download_all"] = True
                        logger.info(f"Using {len(self._last_search_results)} dataset IDs from last search")
        elif intent and intent.startswith("EDUCATION"):
            # For education/explain intents, only pass the concept
            if "concept" in slots:
                params["concept"] = slots["concept"]
            elif "topic" in slots:
                params["concept"] = slots["topic"]
        else:
            # Default: use slots directly
            params = dict(slots)
        
        return params
    
    def _store_search_results(self, result: ToolResult) -> None:
        """Store dataset IDs from search results in conversation context."""
        if result.data and "results" in result.data:
            dataset_ids = [
                r.get("id") for r in result.data["results"] 
                if r.get("id")
            ]
            # Store in both places for compatibility
            self._last_search_results = dataset_ids
            
            # Store in conversation context for proper memory
            self.context.update_state("last_search_results", result.data["results"])
            self.context.update_state("last_search_ids", dataset_ids)
            self.context.update_state("last_search_query", result.data.get("query", ""))
            
            logger.info(f"Stored {len(dataset_ids)} dataset IDs in conversation context")
    
    def _build_params_from_args(self, tool: ToolName, args: List[str]) -> Dict[str, Any]:
        """
        Build kwargs from positional args captured by detect_tool patterns.
        
        This uses the same mapping as AgentTools._TOOL_ARG_MAPPING.
        """
        if not args or not any(args):
            return {}
        
        # Mapping of tool names to their primary argument names
        ARG_MAPPING = {
            ToolName.SCAN_DATA: ["path"],
            ToolName.SEARCH_DATABASES: ["query"],
            ToolName.SEARCH_TCGA: ["query", "cancer_type"],
            ToolName.GET_DATASET_DETAILS: ["dataset_id"],
            ToolName.DESCRIBE_FILES: ["path"],
            ToolName.VALIDATE_DATASET: ["path"],
            ToolName.DOWNLOAD_DATASET: ["dataset_id", "destination"],
            ToolName.DOWNLOAD_REFERENCE: ["genome"],
            ToolName.BUILD_INDEX: ["reference", "tool"],
            ToolName.GENERATE_WORKFLOW: ["pipeline_type", "input_dir"],
            ToolName.CHECK_REFERENCES: ["genome"],
            ToolName.SUBMIT_JOB: ["workflow_dir"],
            ToolName.GET_JOB_STATUS: ["job_id"],
            ToolName.GET_LOGS: ["job_id"],
            ToolName.CANCEL_JOB: ["job_id"],
            ToolName.DIAGNOSE_ERROR: ["job_id", "log_content"],
        }
        
        params = {}
        if tool in ARG_MAPPING:
            arg_names = ARG_MAPPING[tool]
            for i, arg in enumerate(args):
                if arg and i < len(arg_names):
                    params[arg_names[i]] = arg
        
        return params
    
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
        
    async def _handle_context_query(
        self,
        intent: str,
        query: str,
        task_type: TaskType
    ) -> AgentResponse:
        """Handle queries that reference previous context (search results, etc.)."""
        
        # Get previous search results from context
        last_search = self.context.get_state("last_search_results")
        last_datasets = self.context.get_state("last_datasets")
        
        if not last_search and not last_datasets:
            return AgentResponse(
                success=False,
                message="I don't have any previous search results to reference. Try searching for data first:\n\n"
                        "Example: `search ENCODE for H3K27ac ChIP-seq in brain`",
                response_type=ResponseType.ERROR,
                task_type=task_type,
            )
        
        # Build response from context
        datasets = last_datasets or last_search.get("datasets", []) if last_search else []
        
        if intent == "CONTEXT_METADATA":
            # Return detailed metadata about previous results
            if not datasets:
                return AgentResponse(
                    success=True,
                    message="No datasets found in context. The previous search may not have returned results.",
                    response_type=ResponseType.INFO,
                    task_type=task_type,
                )
            
            # Format metadata table
            lines = ["## 📋 Metadata for Previous Search Results\n"]
            lines.append(f"Found **{len(datasets)}** datasets:\n")
            lines.append("| # | ID | Source | Type | Description |")
            lines.append("|---|-----|--------|------|-------------|")
            
            for i, ds in enumerate(datasets[:20], 1):  # Limit to 20
                ds_id = ds.get("id", ds.get("accession", "N/A"))
                source = ds.get("source", "Unknown")
                dtype = ds.get("assay_type", ds.get("data_type", ds.get("type", "N/A")))
                desc = ds.get("description", ds.get("title", ""))[:50]
                lines.append(f"| {i} | `{ds_id}` | {source} | {dtype} | {desc}... |")
            
            lines.append("\n💡 Say `download all` to download these datasets, or `download <ID>` for specific ones.")
            
            return AgentResponse(
                success=True,
                message="\n".join(lines),
                response_type=ResponseType.SUCCESS,
                task_type=task_type,
                data={"datasets": datasets},
            )
        
        elif intent == "CONTEXT_RECALL":
            # Simply recall what was found
            count = len(datasets)
            sources = set(ds.get("source", "Unknown") for ds in datasets)
            
            message = f"From your previous search, I found **{count}** datasets from {', '.join(sources)}.\n\n"
            message += "Would you like me to:\n"
            message += "- Show detailed metadata (`show metadata`)\n"
            message += "- Download all datasets (`download all`)\n"
            message += "- Download specific ones (`download <ID>`)\n"
            
            return AgentResponse(
                success=True,
                message=message,
                response_type=ResponseType.SUCCESS,
                task_type=task_type,
                data={"count": count, "sources": list(sources)},
            )
        
        # Fallback
        return AgentResponse(
            success=True,
            message="I found previous results in context. What would you like to do with them?",
            response_type=ResponseType.INFO,
            task_type=task_type,
        )
    

    async def _handle_confirm(
        self,
        query: str,
        task_type: TaskType
    ) -> AgentResponse:
        """Handle user confirmation - check if there's a pending download."""
        query_lower = query.lower()
        
        # Check for download-related confirmation
        if self._last_search_results:
            dataset_ids = self._last_search_results
            
            # Check last message for download preview
            for msg in reversed(self.conversation_history[-5:]):
                if isinstance(msg, dict) and msg.get("role") == "assistant":
                    if "Download Preview" in msg.get("content", ""):
                        # User is confirming a download
                        file_filter = None
                        # Check for filter modifiers
                        for pattern in ["without", "no ", "only ", "exclude "]:
                            if pattern in query_lower:
                                file_filter = query
                                break
                        
                        result = self.tools.download_dataset(
                            dataset_ids=dataset_ids,
                            file_filter=file_filter,
                            confirm=True
                        )
                        return AgentResponse(
                            success=result.success,
                            message=result.message,
                            response_type=ResponseType.SUCCESS if result.success else ResponseType.ERROR,
                            task_type=task_type,
                        )
        
        return AgentResponse(
            success=False,
            message="What would you like me to confirm? I don't see any pending operations.",
            response_type=ResponseType.QUESTION,
            task_type=task_type,
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
            
        elif task_type == TaskType.DATA:
            # Handle data queries - but validate query first
            # Don't send conversational or non-search queries to ENCODE/GEO APIs
            if not self._is_valid_search_query(query):
                return AgentResponse(
                    success=True,
                    message="I understand you're asking about data. Could you please specify what you'd like to search for?\n\n"
                            "**Examples:**\n"
                            "- `search for human RNA-seq liver`\n"
                            "- `find ATAC-seq brain datasets`\n"
                            "- `search TCGA for GBM methylation`\n\n"
                            "Or use:\n"
                            "- `scan data` - to see your local files\n"
                            "- `show jobs` - to check job status",
                    response_type=ResponseType.INFO,
                    task_type=task_type,
                )
            # For natural language data queries, search databases
            search_query = query
            execution = self.execute_tool(ToolName.SEARCH_DATABASES, query=search_query)
            return self._build_response(task_type, [execution], query)
            
        else:
            # Return help
            execution = self.execute_tool(ToolName.SHOW_HELP)
            return self._build_response(task_type, [execution], query)
    
    def _is_valid_search_query(self, query: str) -> bool:
        """
        Validate if a query is suitable for database search.
        
        Rejects conversational phrases that would cause 404s on ENCODE/GEO APIs.
        """
        query_lower = query.lower().strip()
        
        # Allow dataset IDs even if they're short (1 word)
        dataset_id_patterns = [
            r'\bGSE\d+\b',      # GEO accession
            r'\bENCSR[A-Z0-9]+\b',  # ENCODE accession
            r'\bSR[RXPS]\d+\b',     # SRA accession
            r'\bPRJNA\d+\b',        # BioProject
        ]
        for pattern in dataset_id_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return True
        
        # Reject short queries (less than 2 words typically not useful)
        words = query_lower.split()
        if len(words) < 2:
            return False
        
        # Reject conversational phrases
        conversational_patterns = [
            r'^(?:yes|no|ok|okay|sure|thanks|thank you)',
            r'^(?:you have|i have|we have|there are|there is)',
            r'^(?:job|jobs?)\s+(?:completed|done|finished|running|failed)',
            r'^(?:can you|could you|would you|please)\s+(?:inspect|check|show|look|see)',
            r'^(?:what|how)\s+(?:about|did|do|does|is|are|was|were)',
            r'^(?:it|that|this|they)\s+(?:is|are|was|were|has|have|looks?)',
            r'^(?:great|good|nice|perfect|excellent|awesome)',
            r'^(?:the|a|an)\s+(?:job|file|result|output|log)',
            r'(?:inspect|completed|finished|running|failed|details|metadata)\s*(?:\.|\?|!)?$',
        ]
        
        for pattern in conversational_patterns:
            if re.search(pattern, query_lower):
                return False
        
        # Must contain at least one biological/data-related term
        bio_terms = [
            'rna', 'dna', 'chip', 'atac', 'seq', 'sequencing', 'methylation', 'expression',
            'human', 'mouse', 'rat', 'genome', 'transcriptome', 'epigenome',
            'cancer', 'tumor', 'brain', 'liver', 'heart', 'lung', 'kidney',
            'cell', 'tissue', 'sample', 'experiment', 'dataset',
            'h3k', 'histone', 'chromatin', 'accessibility',
            'encode', 'geo', 'tcga', 'gdc', 'sra',
            'gse', 'encsr', 'srr', 'srp',
            'fastq', 'bam', 'bed', 'bigwig', 'vcf',
        ]
        
        has_bio_term = any(term in query_lower for term in bio_terms)
        
        return has_bio_term
            
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
        
    def get_context_summary(self) -> str:
        """
        Get a summary of the current conversation context.
        
        Useful for:
        - LLM prompting (providing context for responses)
        - Debugging conversation state
        - Generating summaries for users
        
        Returns:
            Formatted context summary
        """
        return self.context.get_context_summary()
    
    def get_recent_entities(self, limit: int = 5) -> List:
        """Get recently mentioned entities from context."""
        return self.context.get_salient_entities(limit)
    
    def reset_context(self):
        """Reset conversation context (start fresh session)."""
        self._context = None
        self._dialogue_manager = None
        self._last_search_results = []
        logger.info("Conversation context reset")


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
