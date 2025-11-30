"""
Agents Module
=============

AI-powered agents for workflow generation and monitoring.

RECOMMENDED USAGE:
==================

For frontend/UI integration, use the BioPipelines facade:

    from workflow_composer import BioPipelines
    
    bp = BioPipelines()
    response = bp.chat("scan /data/raw for FASTQ files")
    print(response.message)

For programmatic access:

    from workflow_composer.agents import UnifiedAgent, AutonomyLevel
    
    agent = UnifiedAgent(autonomy_level=AutonomyLevel.ASSISTED)
    response = agent.process_sync("scan /data/raw for FASTQ files")
    print(response.message)

ADVANCED USAGE:
===============

For direct tool access:
    from workflow_composer.agents import AgentTools, ToolResult, ToolName

For autonomous background jobs:
    from workflow_composer.agents.autonomous import AutonomousAgent
"""

from .tools import AgentTools, ToolResult, ToolName, process_tool_request

# Unified context system from intent module
from .intent import ConversationContext, ContextMemory, EntityTracker, MemoryItem

# Archived/Deprecated: AgentRouter and AgentBridge moved to _archived/
# Use UnifiedAgent instead. See _archived/__init__.py for migration guide.
def __getattr__(name):
    """Lazy load archived modules with deprecation warnings."""
    if name in ("AgentRouter", "RouteResult", "RoutingStrategy", "AGENT_TOOLS", "route_message"):
        import warnings
        warnings.warn(
            f"{name} is deprecated. Use UnifiedAgent with HybridQueryParser instead. "
            "See agents/_archived/__init__.py for migration guide.",
            DeprecationWarning,
            stacklevel=2
        )
        from ._archived import router
        return getattr(router, name)
    if name in ("AgentBridge", "get_agent_bridge", "process_with_agent"):
        import warnings
        warnings.warn(
            f"{name} is deprecated. Use UnifiedAgent instead. "
            "See agents/_archived/__init__.py for migration guide.",
            DeprecationWarning,
            stacklevel=2
        )
        from ._archived import bridge
        return getattr(bridge, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

from .coding_agent import (
    CodingAgent, 
    DiagnosisResult, 
    CodeFix, 
    ErrorType,
    get_coding_agent,
    diagnose_job_error,
)
from .orchestrator import (
    AgentOrchestrator,
    SyncOrchestrator,
    AgentType,
    AgentTask,
    TaskResult,
    get_orchestrator,
    get_sync_orchestrator,
)
from .memory import (
    AgentMemory,
    MemoryEntry,
    SearchResult,
    EmbeddingModel,
)
from .react_agent import (
    ReactAgent,
    SimpleAgent,
    AgentStep,
    AgentState,
    ToolResult as ReactToolResult,
)
from .self_healing import (
    SelfHealer,
    JobMonitor,
    HealingAttempt,
    HealingAction,
    HealingStatus,
    JobInfo,
    get_self_healer,
    start_job_monitor,
    stop_job_monitor,
)
from .multi_model import (
    MultiModelDeployment,
    ModelConfig,
    ModelRole,
    get_deployment,
    get_model_url,
    configure_from_env,
    QUAD_H100_CONFIG,
    DUAL_H100_CONFIG,
    SINGLE_T4_CONFIG,
)
# Executor Layer (Phase 1 - Safe Execution Foundation)
from .executor import (
    CommandSandbox,
    FileOperations,
    ProcessManager,
    AuditLogger,
    PermissionManager,
    AutonomyLevel,
)
# Autonomous System (Phase 3 - Full Autonomy)
from .autonomous import (
    AutonomousAgent,
    create_agent,
    Task,
    TaskStatus,
    Action,
    ActionType,
    JobMonitor as AutonomousJobMonitor,
    JobEvent,
    JobState,
    HealthChecker,
    HealthStatus,
    RecoveryManager,
    RecoveryResult,
    RecoveryLoop,
)
# Enhanced Tools (Phase 2 - Tool Layer)
from .enhanced_tools import (
    EnhancedTool,
    EnhancedToolResult,
    EnhancedToolRegistry,
    ToolStatus,
    get_tool_registry,
    execute_tool,
    execute_tool_sync,
    with_retry,
    # Individual tools
    SLURMSubmitTool,
    SLURMStatusTool,
    SLURMCancelTool,
    SLURMLogsTool,
    VLLMQueryTool,
    VLLMHealthTool,
    FileReadTool,
    FileWriteTool,
    SystemHealthTool,
)
# Unified Agent (Main Entry Point)
from .unified_agent import (
    UnifiedAgent,
    AgentResponse,
    TaskType,
    ResponseType,
    ToolExecution,
    get_agent,
    reset_agent,
    process_query,
    process_query_sync,
)

# Classification (Single Source of Truth)
from .classification import (
    classify_task,
    classify_simple,
    TaskType,
)

# RAG-Enhanced Tool Selection (Phase 6)
from .tool_memory import (
    ToolMemory,
    ToolExecutionRecord,
    RAGToolSelector,
    ToolBoost,
    get_tool_memory,
    get_rag_selector,
)

__all__ = [
    # === PRIMARY: Unified Agent (Recommended Entry Point) ===
    "UnifiedAgent",
    "AgentResponse",
    "TaskType",
    "ResponseType",
    "ToolExecution",
    "get_agent",
    "reset_agent",
    "process_query",
    "process_query_sync",
    # === PRIMARY: Classification (Single Source of Truth) ===
    "classify_task",
    "classify_simple",
    # === PRIMARY: Core Autonomy ===
    "AutonomyLevel",
    # Tools
    "AgentTools",
    "ToolResult",
    "ToolName",
    "process_tool_request",
    # Context
    "ConversationContext",
    # === DEPRECATED: Router & Bridge (use UnifiedAgent instead) ===
    # These are dynamically loaded with deprecation warnings via __getattr__
    # "AgentRouter", "RouteResult", "RoutingStrategy", "AGENT_TOOLS", "route_message",
    # "AgentBridge", "get_agent_bridge", "process_with_agent",
    # Coding Agent
    "CodingAgent",
    "DiagnosisResult",
    "CodeFix",
    "ErrorType",
    "get_coding_agent",
    "diagnose_job_error",
    # Orchestrator
    "AgentOrchestrator",
    "SyncOrchestrator",
    "AgentType",
    "AgentTask",
    "TaskResult",
    "get_orchestrator",
    "get_sync_orchestrator",
    # Memory
    "AgentMemory",
    "MemoryEntry",
    "SearchResult",
    "EmbeddingModel",
    # ReAct Agent
    "ReactAgent",
    "SimpleAgent",
    "AgentStep",
    "AgentState",
    "ReactToolResult",
    # Self-Healing
    "SelfHealer",
    "JobMonitor",
    "HealingAttempt",
    "HealingAction",
    "HealingStatus",
    "JobInfo",
    "get_self_healer",
    "start_job_monitor",
    "stop_job_monitor",
    # Multi-Model Deployment
    "MultiModelDeployment",
    "ModelConfig",
    "ModelRole",
    "get_deployment",
    "get_model_url",
    "configure_from_env",
    "QUAD_H100_CONFIG",
    "DUAL_H100_CONFIG",
    "SINGLE_T4_CONFIG",
    # Executor Layer (Phase 1)
    "CommandSandbox",
    "FileOperations",
    "ProcessManager",
    "AuditLogger",
    "PermissionManager",
    "AutonomyLevel",
    # Autonomous System (Phase 3)
    "AutonomousAgent",
    "create_agent",
    "Task",
    "TaskStatus",
    "Action",
    "ActionType",
    "AutonomousJobMonitor",
    "JobEvent",
    "JobState",
    "HealthChecker",
    "HealthStatus",
    "RecoveryManager",
    "RecoveryResult",
    "RecoveryLoop",
    # Enhanced Tools (Phase 2)
    "EnhancedTool",
    "EnhancedToolResult",
    "EnhancedToolRegistry",
    "ToolStatus",
    "get_tool_registry",
    "execute_tool",
    "execute_tool_sync",
    "with_retry",
    "SLURMSubmitTool",
    "SLURMStatusTool",
    "SLURMCancelTool",
    "SLURMLogsTool",
    "VLLMQueryTool",
    "VLLMHealthTool",
    "FileReadTool",
    "FileWriteTool",
    "SystemHealthTool",
    # RAG-Enhanced Tool Selection (Phase 6)
    "ToolMemory",
    "ToolExecutionRecord",
    "RAGToolSelector",
    "ToolBoost",
    "get_tool_memory",
    "get_rag_selector",
]
