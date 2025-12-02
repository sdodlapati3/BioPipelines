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

# Intent parsing (ensemble parser integration)
from .intent import (
    HybridQueryParser, 
    QueryParseResult, 
    ConversationContext, 
    DialogueManager,
    UnifiedIntentParser,
    UnifiedParseResult,
    # Professional NLU components (Phase 1-5)
    get_active_learner,
    get_slot_prompter,
    get_role_resolver,
    TrainingDataLoader,
    get_training_data_loader,
    # Session memory & recovery (robust agent features)
    SessionMemory,
    get_session_memory,
    ConversationRecovery,
    RecoveryResponse,
    RecoveryStrategy,
    get_conversation_recovery,
)

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
        
        # Hybrid intent parser (legacy, for fallback)
        self._query_parser: Optional[HybridQueryParser] = None
        
        # Unified intent parser (recommended - uses arbiter)
        self._intent_parser: Optional[UnifiedIntentParser] = None
        
        # Conversation context for multi-turn memory
        self._context: Optional[ConversationContext] = None
        self._dialogue_manager: Optional[DialogueManager] = None
        
        # Professional NLU components (lazy-loaded)
        self._active_learner = None
        self._slot_prompter = None
        self._role_resolver = None
        self._training_data = None
        
        # Session memory & recovery (robust agent features)
        self._session_memory: Optional[SessionMemory] = None
        self._recovery: Optional[ConversationRecovery] = None
        
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
        """Get or initialize the hybrid query parser (legacy fallback)."""
        if self._query_parser is None:
            try:
                self._query_parser = HybridQueryParser()
                logger.info("HybridQueryParser initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize HybridQueryParser: {e}")
                self._query_parser = None
        return self._query_parser
    
    @property
    def intent_parser(self) -> Optional[UnifiedIntentParser]:
        """
        Get or initialize the unified intent parser.
        
        Uses hierarchical parsing with LLM arbiter for complex queries.
        Recommended over UnifiedEnsembleParser (deprecated).
        """
        if self._intent_parser is None:
            try:
                self._intent_parser = UnifiedIntentParser(use_cascade=True)
                logger.info("UnifiedIntentParser initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize UnifiedIntentParser: {e}")
                self._intent_parser = None
        return self._intent_parser
    
    # Keep ensemble_parser as alias for backward compatibility
    @property
    def ensemble_parser(self) -> Optional[UnifiedIntentParser]:
        """Deprecated: Use intent_parser instead."""
        return self.intent_parser
    
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
    
    # =========================================================================
    # Professional NLU Components (Phase 1-5)
    # =========================================================================
    
    @property
    def active_learner(self):
        """
        Get the active learner for tracking corrections and feedback.
        
        Records:
        - Corrections when user corrects an intent
        - Confirmations when prediction is correct
        - Confusion matrix analysis
        """
        if self._active_learner is None:
            try:
                self._active_learner = get_active_learner()
                logger.debug("ActiveLearner initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize ActiveLearner: {e}")
        return self._active_learner
    
    @property
    def slot_prompter(self):
        """
        Get the slot prompter for checking required parameters.
        
        Prompts user for missing required slots with natural language.
        """
        if self._slot_prompter is None:
            try:
                self._slot_prompter = get_slot_prompter()
                logger.debug("SlotPrompter initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize SlotPrompter: {e}")
        return self._slot_prompter
    
    @property
    def role_resolver(self):
        """
        Get the entity role resolver.
        
        Determines semantic roles from context:
        - source vs destination paths
        - baseline vs target conditions
        """
        if self._role_resolver is None:
            try:
                self._role_resolver = get_role_resolver()
                logger.debug("EntityRoleResolver initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize EntityRoleResolver: {e}")
        return self._role_resolver
    
    @property
    def training_data(self):
        """
        Get the training data loader.
        
        Provides:
        - Entity alias normalization
        - Intent examples for classification
        """
        if self._training_data is None:
            try:
                self._training_data = get_training_data_loader()
                logger.debug("TrainingDataLoader initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize TrainingDataLoader: {e}")
        return self._training_data
    
    @property
    def session_memory(self) -> SessionMemory:
        """
        Get session-wide memory for persistent context.
        
        Remembers across the entire session:
        - Paths (data directories, output locations)
        - Datasets (IDs from searches)
        - Preferences (organism, assay type)
        - Action history (what was done)
        
        Usage:
            # Remember a path used in a query
            agent.session_memory.remember_path("/data/methylation")
            
            # Later, resolve "that path" 
            path = agent.session_memory.get_remembered_path()
        """
        if self._session_memory is None:
            try:
                self._session_memory = get_session_memory()
                logger.debug("SessionMemory initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize SessionMemory: {e}")
        return self._session_memory
    
    @property
    def recovery(self) -> ConversationRecovery:
        """
        Get the conversation recovery system.
        
        Handles:
        - Low confidence intents (ask for clarification)
        - Errors (graceful acknowledgment, suggestions)
        - User corrections ("No, I meant X")
        - Fallback responses (when nothing else works)
        """
        if self._recovery is None:
            try:
                self._recovery = get_conversation_recovery()
                logger.debug("ConversationRecovery initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize ConversationRecovery: {e}")
        return self._recovery
    
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
        Uses unified ensemble parsing (rule + semantic + NER + LLM + RAG)
        with weighted voting for robust detection.
        Falls back to hybrid parser if ensemble unavailable.
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
            
            # ================================================================
            # STEP 0: Resolve references using session memory
            # Handles: "that path", "the data", "it" -> actual values
            # ================================================================
            original_query = query
            resolved_refs = {}
            if self.session_memory:
                try:
                    query, resolved_refs = self.session_memory.resolve_references_in_query(query)
                    if resolved_refs:
                        logger.info(f"Resolved references: {resolved_refs}")
                        span.add_event("references_resolved", {"resolutions": str(resolved_refs)})
                except Exception as e:
                    logger.debug(f"Reference resolution failed: {e}")
            
            # Classify the task
            with tracer.start_span("classify_task") as classify_span:
                task_type = classify_task(query)
                classify_span.add_tag("task_type", task_type.value)
            span.add_event("task_classified", {"task_type": task_type.value})
            logger.info(f"Task classified as: {task_type.value}")
            
            # Try unified intent parsing first (pattern + semantic + arbiter)
            detected_tool = None
            detected_args = []
            hybrid_params = None  # Params from parser (preferred)
            parse_result: Optional[QueryParseResult] = None
            intent_result: Optional[UnifiedParseResult] = None
            extracted_entities = []  # For context tracking
            
            # Use intent parser (recommended) or fall back to hybrid
            if self.intent_parser:
                try:
                    with tracer.start_span("intent_parse") as parse_span:
                        intent_result = self.intent_parser.parse(query)
                        intent_name = intent_result.primary_intent.name
                        parse_span.add_tag("intent", intent_name)
                        parse_span.add_tag("confidence", intent_result.confidence)
                        parse_span.add_tag("method", intent_result.method)
                        parse_span.add_tag("llm_invoked", intent_result.llm_invoked)
                    
                    logger.info(
                        f"IntentParser: intent={intent_name}, "
                        f"confidence={intent_result.confidence:.2f}, "
                        f"method={intent_result.method}, "
                        f"llm={intent_result.llm_invoked}"
                    )
                    
                    # Extract entities from result
                    extracted_entities = intent_result.entities if intent_result.entities else []
                    
                    # ============================================================
                    # ENHANCED: Use ConversationRecovery for low confidence
                    # ============================================================
                    if intent_result.needs_clarification or intent_result.confidence < 0.35:
                        # Use recovery system for intelligent clarification
                        if self.recovery:
                            alternative_intents = []
                            if hasattr(intent_result, 'all_intents') and intent_result.all_intents:
                                alternative_intents = [
                                    (i.name, c) for i, c in intent_result.all_intents[:3]
                                ]
                            
                            recovery_response = self.recovery.handle_low_confidence(
                                query=query,
                                confidence=intent_result.confidence,
                                detected_intent=intent_name,
                                alternative_intents=alternative_intents,
                            )
                            
                            return AgentResponse(
                                success=True,
                                message=recovery_response.message,
                                response_type=ResponseType.QUESTION,
                                task_type=task_type,
                                suggestions=recovery_response.suggestions,
                            )
                        else:
                            # Fallback to simple clarification
                            return AgentResponse(
                                success=True,
                                message=intent_result.clarification_prompt or "Could you please clarify your request?",
                                response_type=ResponseType.QUESTION,
                                task_type=task_type,
                            )
                    
                    # Map intent to tool if confidence is reasonable
                    if intent_result.confidence >= 0.35 and intent_name in INTENT_TO_TOOL:
                        detected_tool = INTENT_TO_TOOL[intent_name]
                        
                        # Build params from slots
                        hybrid_params = intent_result.slots.copy() if intent_result.slots else {}
                        detected_args = list(hybrid_params.values()) if hybrid_params else []
                        logger.info(f"Mapped intent '{intent_name}' to tool: {detected_tool}, params: {hybrid_params}")
                    
                    # Handle context-aware intents
                    if intent_name in ("CONTEXT_RECALL", "CONTEXT_METADATA"):
                        return await self._handle_context_query(intent_name, query, task_type)
                        
                except Exception as e:
                    span.add_event("intent_fallback", {"error": str(e)})
                    logger.warning(f"IntentParser failed, falling back to HybridParser: {e}")
            
            # Fallback to legacy hybrid parser if ensemble didn't work
            if not detected_tool and self.query_parser:
                try:
                    with tracer.start_span("hybrid_parse") as parse_span:
                        parse_result = self.query_parser.parse(query)
                        parse_span.add_tag("intent", parse_result.intent)
                        parse_span.add_tag("confidence", parse_result.intent_confidence)
                    logger.info(f"HybridParser fallback: intent={parse_result.intent}, confidence={parse_result.intent_confidence:.2f}")
                    
                    # Extract entities for context
                    extracted_entities = parse_result.entities if parse_result.entities else []
                    
                    # Map intent to tool
                    if parse_result.intent_confidence >= 0.35 and parse_result.intent in INTENT_TO_TOOL:
                        detected_tool = INTENT_TO_TOOL[parse_result.intent]
                        hybrid_params = self._build_params_from_parse_result(parse_result, query)
                        detected_args = list(hybrid_params.values()) if hybrid_params else []
                        logger.info(f"Mapped intent '{parse_result.intent}' to tool: {detected_tool}, params: {hybrid_params}")
                    
                    # Handle context-aware intents
                    if parse_result.intent in ("CONTEXT_RECALL", "CONTEXT_METADATA"):
                        return await self._handle_context_query(parse_result.intent, query, task_type)
                        
                except Exception as e:
                    span.add_event("parser_fallback", {"error": str(e)})
                    logger.warning(f"HybridParser failed, falling back to regex: {e}")
        
        # Add user turn to conversation context
        # Note: We pass entities=[] because BioEntity and Entity have different structures
        # The context still tracks the message and intent
        detected_intent = None
        if intent_result:
            detected_intent = intent_result.primary_intent.name
        elif parse_result:
            detected_intent = parse_result.intent
            
        self.context.add_turn(
            role="user",
            content=query,
            entities=[],  # BioEntity not compatible with Entity, store separately
            intent=detected_intent
        )
        
        # Store extracted entities in context state for later use
        if extracted_entities:
            # Handle both Entity (type, value) and BioEntity (entity_type, text) formats
            entity_list = []
            for e in extracted_entities:
                entity_type = getattr(e, 'entity_type', None) or getattr(e, 'type', None)
                entity_text = getattr(e, 'text', None) or getattr(e, 'value', None)
                entity_canonical = getattr(e, 'canonical', entity_text)
                if entity_type and entity_text:
                    # Convert enum to string if needed
                    if hasattr(entity_type, 'name'):
                        entity_type = entity_type.name
                    entity_list.append({
                        "type": entity_type, 
                        "text": entity_text, 
                        "canonical": entity_canonical
                    })
            if entity_list:
                self.context.update_state("last_entities", entity_list)
        
        # Fallback to regex-based detection if hybrid parsing didn't find a tool
        if not detected_tool:
            detection = self.tools.detect_tool(query)
            if detection:
                detected_tool, detected_args = detection
                logger.info(f"Regex detected tool: {detected_tool.value} with args: {detected_args}")
        
        if detected_tool:
            logger.info(f"Final detected tool: {detected_tool.value} with args: {detected_args}")
            
            # Slot prompting: Check if required parameters are missing
            if detected_intent:
                try:
                    slot_check = self.slot_prompter.check_slots(detected_intent, hybrid_params or {})
                    if slot_check.needs_prompting:
                        logger.info(f"Missing required slots for intent '{detected_intent}': {slot_check.missing_slots}")
                        return AgentResponse(
                            success=True,
                            message=slot_check.prompt,
                            response_type=ResponseType.QUESTION,
                            task_type=task_type,
                            suggestions=[f"Provide {slot}" for slot in slot_check.missing_slots[:3]],
                        )
                except Exception as e:
                    logger.warning(f"Slot prompting check failed: {e}")
            
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
                    
                    # ============================================================
                    # ENHANCED ERROR HANDLING: Use ConversationRecovery
                    # ============================================================
                    if self.recovery and execution.result.error:
                        try:
                            recovery_response = self.recovery.handle_error(
                                error=Exception(execution.result.error),
                                query=query,
                                tool_name=detected_tool.value,
                                parameters=params,
                            )
                            
                            # Return user-friendly error response with suggestions
                            return AgentResponse(
                                success=False,
                                message=recovery_response.message,
                                response_type=ResponseType.ERROR,
                                task_type=task_type,
                                suggestions=recovery_response.suggestions,
                            )
                        except Exception as recovery_err:
                            logger.debug(f"Recovery handling failed: {recovery_err}")
                    
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
            
            # Active learning: Record successful query-intent-slots mapping for potential retraining
            if response.success and detected_intent:
                try:
                    self.active_learner.record_confirmation(
                        query=query,
                        intent=detected_intent,
                        confidence=intent_result.confidence if intent_result else 0.5,
                    )
                except Exception as e:
                    logger.debug(f"Active learning recording failed: {e}")
            
            # ================================================================
            # SESSION MEMORY: Auto-remember paths, datasets, results
            # This enables "that path", "the data" in future queries
            # ================================================================
            if self.session_memory and response.success:
                try:
                    # Remember paths from parameters
                    for key in ("path", "data_path", "input_path", "output_path", "directory"):
                        if key in params and params[key]:
                            self.session_memory.remember_path(
                                params[key], 
                                context=key.replace("_path", "").replace("_", " ")
                            )
                    
                    # Remember dataset IDs
                    for key in ("dataset_id", "accession", "sample_id"):
                        if key in params and params[key]:
                            self.session_memory.remember_dataset(params[key])
                    
                    # Remember search results
                    if detected_tool and detected_tool.value in ("search_databases", "scan_data"):
                        result_data = execution.result.data if execution.result else {}
                        if isinstance(result_data, dict):
                            # From search
                            results = result_data.get("results", result_data.get("samples", []))
                            if results:
                                self.session_memory.remember_search_results(results, query)
                    
                    # Record the action
                    self.session_memory.record_action(
                        action_type=detected_tool.value if detected_tool else "general",
                        query=query,
                        tool_used=detected_tool.value if detected_tool else "none",
                        success=response.success,
                        parameters=params,
                        result_summary=response.message[:200] if response.message else None,
                    )
                    
                    # Infer preferences from entities
                    if extracted_entities:
                        self.session_memory.infer_preferences_from_entities(extracted_entities)
                        
                except Exception as e:
                    logger.debug(f"Session memory recording failed: {e}")
            
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
        Handles various event loop scenarios including AnyIO worker threads.
        
        Args:
            query: User's natural language query
            
        Returns:
            AgentResponse with results
        """
        import concurrent.futures
        
        try:
            # Try to get the running loop
            try:
                loop = asyncio.get_running_loop()
                # We're in an async context - use thread pool to run in new loop
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(asyncio.run, self.process_query(query))
                    return future.result(timeout=120)  # 2 min timeout
            except RuntimeError:
                # No running loop - we can create one
                pass
            
            # Try to get existing event loop (may not be running)
            try:
                loop = asyncio.get_event_loop()
                if loop.is_closed():
                    raise RuntimeError("Loop is closed")
                return loop.run_until_complete(self.process_query(query))
            except RuntimeError:
                # No event loop or it's closed - create a new one
                return asyncio.run(self.process_query(query))
                
        except Exception as e:
            # Last resort fallback - create fresh event loop
            logger.warning(f"Event loop handling failed, using fresh loop: {e}")
            try:
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    return new_loop.run_until_complete(self.process_query(query))
                finally:
                    new_loop.close()
            except Exception as inner_e:
                logger.error(f"Failed to process query: {inner_e}")
                return AgentResponse(
                    success=False,
                    message=f"❌ Error processing query: {inner_e}",
                    response_type=ResponseType.ERROR,
                )
            
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
        elif intent == "DATA_SCAN":
            # For data scan, pass path and optional data_type filter
            if "path" in slots:
                params["path"] = slots["path"]
            elif "directory" in slots:
                params["path"] = slots["directory"]
            elif "folder" in slots:
                params["path"] = slots["folder"]
            # Pass data_type for filtering if specified
            if "data_type" in slots:
                params["data_type"] = slots["data_type"]
        else:
            # Default: use slots directly, but filter out common invalid params
            invalid_params = {"concept", "topic", "question"}  # These are for explain_concept only
            params = {k: v for k, v in slots.items() if k not in invalid_params}
        
        return params
    
    def _build_params_from_ensemble_result(
        self,
        ensemble_result,  # EnsembleParseResult or UnifiedParseResult
        original_query: str
    ) -> Dict[str, Any]:
        """
        Build tool parameters from ensemble parse result.
        
        DEPRECATED: Now using intent_result.slots directly.
        Kept for backward compatibility.
        """
        params = {}
        slots = ensemble_result.slots
        entities = ensemble_result.entities
        intent = ensemble_result.intent
        
        # For data search, build query from entities if available
        if intent == "DATA_SEARCH":
            if entities:
                # Build a clean query from entities - deduplicate!
                seen_terms = set()
                query_parts = []
                for entity in entities:
                    if hasattr(entity, 'entity_type'):
                        if entity.entity_type in ("ORGANISM", "TISSUE", "DISEASE", "ASSAY_TYPE"):
                            term = (getattr(entity, 'canonical', None) or entity.text).lower()
                            if term not in seen_terms:
                                seen_terms.add(term)
                                query_parts.append(getattr(entity, 'canonical', None) or entity.text)
                    elif isinstance(entity, dict):
                        if entity.get("type") in ("ORGANISM", "TISSUE", "DISEASE", "ASSAY_TYPE"):
                            term = (entity.get("canonical") or entity.get("text", "")).lower()
                            if term not in seen_terms:
                                seen_terms.add(term)
                                query_parts.append(entity.get("canonical") or entity.get("text"))
                if query_parts:
                    params["query"] = " ".join(query_parts)
                elif "query" in slots:
                    params["query"] = slots["query"]
            elif "query" in slots:
                params["query"] = slots["query"]
                
        elif intent == "DATA_DOWNLOAD":
            if "dataset_id" in slots:
                params["dataset_id"] = slots["dataset_id"]
            else:
                # Check entities for dataset IDs
                for entity in entities:
                    entity_type = entity.entity_type if hasattr(entity, 'entity_type') else entity.get("type")
                    entity_text = entity.text if hasattr(entity, 'text') else entity.get("text")
                    if entity_type == "DATASET_ID":
                        params["dataset_id"] = entity_text
                        break
                
            # Check for "download all"
            if not params.get("dataset_id"):
                query_lower = original_query.lower()
                if any(kw in query_lower for kw in ["all", "everything", "both", "execute", "run"]):
                    context_ids = self.context.get_state("last_search_ids")
                    if context_ids:
                        params["dataset_ids"] = context_ids.copy()
                        params["download_all"] = True
                    elif self._last_search_results:
                        params["dataset_ids"] = self._last_search_results.copy()
                        params["download_all"] = True
                        
        elif intent and intent.startswith("EDUCATION"):
            if "concept" in slots:
                params["concept"] = slots["concept"]
            elif "topic" in slots:
                params["concept"] = slots["topic"]
        elif intent == "DATA_SCAN":
            # For data scan, pass path and optional data_type filter
            if "path" in slots:
                params["path"] = slots["path"]
            elif "directory" in slots:
                params["path"] = slots["directory"]
            elif "folder" in slots:
                params["path"] = slots["folder"]
            # Pass data_type for filtering if specified
            if "data_type" in slots:
                params["data_type"] = slots["data_type"]
        else:
            # Default: filter out common invalid params
            invalid_params = {"concept", "topic", "question"}
            params = {k: v for k, v in slots.items() if k not in invalid_params}
        
        return params
    
    def _build_clarification_request(
        self,
        query: str,
        ensemble_result  # EnsembleParseResult (deprecated)
    ) -> str:
        """
        Build a clarification request when ensemble methods disagree.
        
        DEPRECATED: UnifiedIntentParser handles clarification via needs_clarification flag.
        Kept for backward compatibility.
        """
        conflicting_intents = {}
        for vote in ensemble_result.votes:
            if vote.intent not in conflicting_intents:
                conflicting_intents[vote.intent] = []
            conflicting_intents[vote.intent].append(vote.method)
        
        # Map intents to user-friendly descriptions
        intent_descriptions = {
            "DATA_SEARCH": "search for datasets",
            "DATA_DOWNLOAD": "download a dataset",
            "DATA_SCAN": "scan local files",
            "DATA_DESCRIBE": "describe files",
            "WORKFLOW_CREATE": "create a workflow",
            "JOB_SUBMIT": "submit a job",
            "JOB_STATUS": "check job status",
            "EDUCATION_EXPLAIN": "explain a concept",
        }
        
        options = []
        for i, (intent, methods) in enumerate(conflicting_intents.items(), 1):
            desc = intent_descriptions.get(intent, intent.lower().replace("_", " "))
            options.append(f"{i}. **{desc}**")
        
        message = f"I'm not entirely sure what you'd like to do. Did you mean to:\n\n"
        message += "\n".join(options)
        message += "\n\nPlease clarify or rephrase your request."
        
        return message
    
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
        
        # ================================================================
        # ENHANCED: Use session memory context for better handling
        # ================================================================
        if self.session_memory:
            # Check if query references something we remember
            context_summary = self.session_memory.get_context_summary()
            logger.debug(f"Session context: {context_summary}")
        
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
                # ============================================================
                # ENHANCED: Check session memory for context
                # ============================================================
                context_info = ""
                if self.session_memory:
                    last_path = self.session_memory.get_remembered_path()
                    last_dataset = self.session_memory.get_remembered_dataset()
                    if last_path or last_dataset:
                        context_info = "\n\n📝 **From your session:**\n"
                        if last_path:
                            context_info += f"- Last used path: `{last_path}`\n"
                        if last_dataset:
                            context_info += f"- Last dataset: `{last_dataset}`\n"
                
                return AgentResponse(
                    success=True,
                    message="I understand you're asking about data. Could you please specify what you'd like to search for?\n\n"
                            "**Examples:**\n"
                            "- `search for human RNA-seq liver`\n"
                            "- `find ATAC-seq brain datasets`\n"
                            "- `search TCGA for GBM methylation`\n\n"
                            "Or use:\n"
                            "- `scan data` - to see your local files\n"
                            "- `show jobs` - to check job status"
                            + context_info,
                    response_type=ResponseType.INFO,
                    task_type=task_type,
                )
            # For natural language data queries, search databases
            search_query = query
            execution = self.execute_tool(ToolName.SEARCH_DATABASES, query=search_query)
            return self._build_response(task_type, [execution], query)
            
        else:
            # ============================================================
            # ENHANCED: Use ConversationRecovery for fallback
            # ============================================================
            if self.recovery:
                fallback = self.recovery.get_fallback_response(query)
                return AgentResponse(
                    success=True,
                    message=fallback.message,
                    response_type=ResponseType.INFO,
                    task_type=task_type,
                    suggestions=fallback.suggestions,
                )
            
            # Default: Return help
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
    
    def provide_feedback(
        self, 
        correct_intent: Optional[str] = None,
        correct_slots: Optional[Dict[str, Any]] = None,
        feedback_text: Optional[str] = None,
    ) -> bool:
        """
        Provide explicit feedback on the last query for active learning.
        
        Call this method when the user wants to correct the agent's
        understanding of their last query. The feedback is recorded
        for potential model retraining.
        
        Args:
            correct_intent: The correct intent if the agent predicted wrong
            correct_slots: The correct slot values if extracted incorrectly  
            feedback_text: Optional free-text feedback
            
        Returns:
            True if feedback was recorded successfully
            
        Example:
            # User: "run RNA-seq on sample123"
            # Agent: Detected WORKFLOW_GENERATE (wrong)
            # User: "No, I wanted to analyze existing data"
            agent.provide_feedback(
                correct_intent="DATA_ANALYSIS",
                correct_slots={"sample_id": "sample123"}
            )
        """
        if not self._history:
            logger.warning("No previous query to provide feedback on")
            return False
            
        # Get the last conversation turn
        last_turns = self.context.get_recent_turns(2)
        if len(last_turns) < 2:
            logger.warning("No user query found in context")
            return False
            
        # Find the user's query (second to last turn)
        user_turn = last_turns[-2] if last_turns[-2].get("role") == "user" else None
        if not user_turn:
            logger.warning("Could not find user turn")
            return False
            
        query = user_turn.get("content", "")
        predicted_intent = user_turn.get("intent", "UNKNOWN")
        
        try:
            # Record correction when user provides correct intent
            if correct_intent and correct_intent != predicted_intent:
                self.active_learner.record_correction(
                    query=query,
                    predicted=predicted_intent,
                    corrected=correct_intent,
                )
            logger.info(f"Recorded user feedback: intent={correct_intent}, slots={correct_slots}")
            return True
        except Exception as e:
            logger.error(f"Failed to record feedback: {e}")
            return False
    
    def get_active_learning_stats(self) -> Dict[str, Any]:
        """
        Get statistics about active learning feedback collected.
        
        Returns:
            Dictionary with feedback counts, distribution, etc.
        """
        metrics = self.active_learner.get_metrics()
        return metrics.to_dict()
    
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
