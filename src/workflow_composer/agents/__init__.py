"""
Agents Module
=============

AI-powered agents for workflow generation and monitoring.

RECOMMENDED USAGE (Tier 1 - Primary API):
=========================================

    from workflow_composer.agents import UnifiedAgent, AutonomyLevel
    
    agent = UnifiedAgent(autonomy_level=AutonomyLevel.ASSISTED)
    response = agent.process_sync("scan /data/raw for FASTQ files")
    print(response.message)

ADVANCED USAGE (Tier 2 - Tools & Autonomous):
=============================================

For direct tool access:
    from workflow_composer.agents import AgentTools, ToolResult, ToolName

For autonomous background jobs:
    from workflow_composer.agents.autonomous import AutonomousAgent

INTERNAL/LEGACY (Tier 3 - Not recommended for new code):
========================================================

These are maintained for backward compatibility:
- AgentOrchestrator → Use UnifiedAgent instead
- AgentBridge → Use UnifiedAgent instead
- AgentRouter → Internal use only

Available Components:
--------------------
- UnifiedAgent: **RECOMMENDED** Main entry point combining orchestration + tools
- AutonomyLevel: Permission levels (READONLY, MONITORED, ASSISTED, SUPERVISED, AUTONOMOUS)
- AgentTools: Tools that can be invoked during chat conversations
- AgentOrchestrator: (Legacy) Coordinates multiple agents for complex tasks
- AgentBridge: (Legacy) Bridges LLM routing with tool execution
- AutonomousAgent: Full autonomous agent with execution capabilities
- Executor Layer: Safe file/command execution with audit trail
"""

from .tools import AgentTools, ToolResult, ToolName, process_tool_request
from .context import ConversationContext
from .router import AgentRouter, RouteResult, RoutingStrategy, AGENT_TOOLS, route_message
from .bridge import AgentBridge, get_agent_bridge, process_with_agent
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
from .chat_integration import (
    AgentChatHandler,
    get_chat_handler,
    create_gradio_chat_fn,
    enhanced_chat_with_composer,
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
# Unified Agent (NEW - Main Entry Point)
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
    # Router
    "AgentRouter",
    "RouteResult",
    "RoutingStrategy",
    "AGENT_TOOLS",
    "route_message",
    # Bridge
    "AgentBridge",
    "get_agent_bridge",
    "process_with_agent",
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
    # Memory (new)
    "AgentMemory",
    "MemoryEntry",
    "SearchResult",
    "EmbeddingModel",
    # ReAct Agent (new)
    "ReactAgent",
    "SimpleAgent",
    "AgentStep",
    "AgentState",
    "ReactToolResult",
    # Self-Healing (new)
    "SelfHealer",
    "JobMonitor",
    "HealingAttempt",
    "HealingAction",
    "HealingStatus",
    "JobInfo",
    "get_self_healer",
    "start_job_monitor",
    "stop_job_monitor",
    # Gradio Integration (new)
    "AgentChatHandler",
    "get_chat_handler",
    "create_gradio_chat_fn",
    "enhanced_chat_with_composer",
    # Multi-Model Deployment (new)
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
]
