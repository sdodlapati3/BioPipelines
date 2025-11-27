"""
Autonomous Agent
================

The main autonomous agent that orchestrates:
- Tool execution
- Task loops
- Health monitoring
- Error recovery

This is the "brain" of the autonomous system.
"""

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import json
import traceback

from ..executor import (
    CommandSandbox,
    FileOperations,
    ProcessManager,
    AuditLogger,
    PermissionManager,
    AutonomyLevel,
)
from .job_monitor import JobMonitor, JobEvent, JobState
from .health_checker import HealthChecker, HealthStatus, SystemHealth
from .recovery import RecoveryManager, RecoveryResult

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Status of a task."""
    PENDING = "pending"
    RUNNING = "running"
    WAITING_CONFIRMATION = "waiting_confirmation"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ActionType(Enum):
    """Types of agent actions."""
    READ_FILE = "read_file"
    WRITE_FILE = "write_file"
    EDIT_FILE = "edit_file"
    RUN_COMMAND = "run_command"
    SUBMIT_JOB = "submit_job"
    MONITOR_JOB = "monitor_job"
    CHECK_HEALTH = "check_health"
    DIAGNOSE = "diagnose"
    RECOVER = "recover"
    THINK = "think"
    RESPOND = "respond"


@dataclass
class Action:
    """An action the agent can take."""
    type: ActionType
    params: Dict[str, Any]
    description: str
    requires_confirmation: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "type": self.type.value,
            "params": self.params,
            "description": self.description,
            "requires_confirmation": self.requires_confirmation,
        }


@dataclass
class ActionResult:
    """Result of an action."""
    action: Action
    success: bool
    output: Any = None
    error: Optional[str] = None
    duration_seconds: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "action": self.action.to_dict(),
            "success": self.success,
            "output": str(self.output) if self.output else None,
            "error": self.error,
            "duration_seconds": self.duration_seconds,
        }


@dataclass
class Task:
    """A task for the agent to complete."""
    id: str
    description: str
    goal: str
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    actions: List[ActionResult] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    result: Optional[str] = None
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "description": self.description,
            "goal": self.goal,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "actions": [a.to_dict() for a in self.actions],
            "result": self.result,
            "error": self.error,
        }


class AutonomousAgent:
    """
    Main autonomous agent.
    
    Capabilities:
    - Execute file operations (read, write, edit)
    - Run shell commands
    - Submit and monitor SLURM jobs
    - Check system health
    - Diagnose and recover from errors
    - Continuous task loop
    
    Example:
        agent = AutonomousAgent()
        
        # Single task
        result = await agent.execute_task(
            "Fix the import error in main.py"
        )
        
        # Continuous loop
        await agent.start_loop()
        agent.add_task("Monitor vLLM server health")
    """
    
    def __init__(
        self,
        workspace_root: Optional[Path] = None,
        autonomy_level: AutonomyLevel = AutonomyLevel.ASSISTED,
        confirmation_callback: Optional[Callable[[str, Action], bool]] = None,
        response_callback: Optional[Callable[[str], None]] = None,
        vllm_url: Optional[str] = None,
    ):
        """
        Initialize autonomous agent.
        
        Args:
            workspace_root: Root directory for operations
            autonomy_level: Level of autonomous operation
            confirmation_callback: Callback for action confirmation
            response_callback: Callback for agent responses
            vllm_url: URL of the vLLM server (auto-detected if not provided)
        """
        self.workspace_root = workspace_root or Path.cwd()
        self.vllm_url = vllm_url or os.environ.get("VLLM_URL", "http://localhost:8000/v1")
        
        # Initialize executor layer
        self.permissions = PermissionManager(
            autonomy_level=autonomy_level,
        )
        self.sandbox = CommandSandbox(
            workspace_root=self.workspace_root,
        )
        self.file_ops = FileOperations(
            workspace=self.workspace_root,
        )
        self.process_manager = ProcessManager()
        self.audit = AuditLogger(
            log_dir=self.workspace_root / "logs" / "audit",
        )
        
        # Initialize autonomous components
        self.job_monitor = JobMonitor()
        self.health_checker = HealthChecker()
        self.recovery = RecoveryManager(
            workspace_root=self.workspace_root,
        )
        
        # Initialize reasoning components (lazy-loaded)
        self._react_agent = None
        self._coding_agent = None
        self._memory = None
        self._tools = None
        self._llm_client = None
        
        # Session context - unified context across all operations
        self._session_context: Dict[str, Any] = {
            "workspace": str(self.workspace_root),
            "autonomy_level": autonomy_level.name,
            "started_at": datetime.now().isoformat(),
            "actions_taken": [],
            "errors_encountered": [],
            "jobs_monitored": [],
        }
        
        # Callbacks
        self._confirmation_callback = confirmation_callback
        self._response_callback = response_callback
        
        # State
        self._running = False
        self._task_queue: asyncio.Queue[Task] = asyncio.Queue()
        self._current_task: Optional[Task] = None
        self._completed_tasks: List[Task] = []
        self._loop_task: Optional[asyncio.Task] = None
        
        # Set up recovery confirmation
        if confirmation_callback:
            self.recovery.set_confirmation_callback(
                lambda msg, action: confirmation_callback(msg, Action(
                    type=ActionType.RECOVER,
                    params={"action": action.value},
                    description=msg,
                ))
            )
    
    @property
    def autonomy_level(self) -> AutonomyLevel:
        """Get current autonomy level."""
        return self.permissions.autonomy_level
    
    @autonomy_level.setter
    def autonomy_level(self, level: AutonomyLevel):
        """Set autonomy level."""
        self.permissions.autonomy_level = level
    
    @property
    def react_agent(self):
        """Lazy-load ReactAgent for multi-step reasoning."""
        if self._react_agent is None:
            try:
                from ..react_agent import ReactAgent
                from openai import OpenAI
                
                # Create LLM client
                llm_client = OpenAI(base_url=self.vllm_url, api_key="not-needed")
                
                # Get tools from EnhancedTools if available
                tools = {}
                if self.tools:
                    tools = self.tools.get_tool_definitions()
                
                self._react_agent = ReactAgent(
                    tools=tools,
                    llm_client=llm_client,
                    model="MiniMaxAI/MiniMax-M1-80k",
                    max_steps=5,
                    verbose=True,
                )
                logger.info("ReactAgent initialized for reasoning tasks")
            except ImportError as e:
                logger.warning(f"ReactAgent not available: {e}")
            except Exception as e:
                logger.warning(f"Failed to initialize ReactAgent: {e}")
        return self._react_agent
    
    @property
    def coding_agent(self):
        """Lazy-load CodingAgent for code analysis/fixes."""
        if self._coding_agent is None:
            try:
                from ..coding_agent import CodingAgent
                
                # Use dedicated coder model if available
                coder_url = os.environ.get("VLLM_CODER_URL")
                
                self._coding_agent = CodingAgent(
                    coder_url=coder_url,
                    coder_model="Qwen/Qwen2.5-Coder-32B-Instruct",
                    fallback_url=self.vllm_url,
                    fallback_model="MiniMaxAI/MiniMax-M1-80k",
                )
                logger.info("CodingAgent initialized for code tasks")
            except ImportError as e:
                logger.warning(f"CodingAgent not available: {e}")
            except Exception as e:
                logger.warning(f"Failed to initialize CodingAgent: {e}")
        return self._coding_agent
    
    @property
    def memory(self):
        """Lazy-load AgentMemory for session context."""
        if self._memory is None:
            try:
                from ..memory import AgentMemory
                
                # Use workspace-relative path for memory DB
                memory_dir = self.workspace_root / ".biopipelines"
                memory_dir.mkdir(parents=True, exist_ok=True)
                
                self._memory = AgentMemory(
                    db_path=str(memory_dir / "agent_memory.db"),
                )
                logger.info("AgentMemory initialized for context tracking")
            except ImportError as e:
                logger.warning(f"AgentMemory not available: {e}")
            except Exception as e:
                logger.warning(f"Failed to initialize AgentMemory: {e}")
        return self._memory
        return self._memory
    
    @property
    def tools(self):
        """Lazy-load enhanced tools registry."""
        if self._tools is None:
            try:
                from ..enhanced_tools import get_tool_registry
                self._tools = get_tool_registry()
                logger.info(f"Enhanced tools registry loaded: {self._tools.list_tools()}")
            except ImportError as e:
                logger.warning(f"Enhanced tools not available: {e}")
            except Exception as e:
                logger.warning(f"Failed to load enhanced tools: {e}")
        return self._tools
    
    async def execute_action(self, action: Action) -> ActionResult:
        """
        Execute a single action.
        
        Args:
            action: Action to execute
        
        Returns:
            ActionResult with success/failure and output
        """
        start = time.time()
        
        try:
            # Check permissions
            operation = self._action_to_operation(action.type)
            permission = self.permissions.check_permission(operation)
            
            if not permission.allowed:
                return ActionResult(
                    action=action,
                    success=False,
                    error=f"Permission denied: {permission.reason}",
                    duration_seconds=time.time() - start,
                )
            
            # Handle confirmation if required
            if permission.requires_confirmation or action.requires_confirmation:
                if self._confirmation_callback:
                    confirmed = self._confirmation_callback(action.description, action)
                    if not confirmed:
                        return ActionResult(
                            action=action,
                            success=False,
                            error="Action rejected by user",
                            duration_seconds=time.time() - start,
                        )
                else:
                    # No callback but confirmation required - deny
                    return ActionResult(
                        action=action,
                        success=False,
                        error="Confirmation required but no callback available",
                        duration_seconds=time.time() - start,
                    )
            
            # Execute action
            output = await self._execute_action_internal(action)
            
            # Audit
            self.audit.log_action(
                action=operation,
                details=action.params,
                result="success",
            )
            
            return ActionResult(
                action=action,
                success=True,
                output=output,
                duration_seconds=time.time() - start,
            )
            
        except Exception as e:
            logger.exception(f"Action error: {e}")
            
            # Audit failure
            self.audit.log_action(
                action=self._action_to_operation(action.type),
                details=action.params,
                result="failure",
                error=str(e),
            )
            
            return ActionResult(
                action=action,
                success=False,
                error=str(e),
                duration_seconds=time.time() - start,
            )
    
    async def _execute_action_internal(self, action: Action) -> Any:
        """Execute action implementation."""
        
        if action.type == ActionType.READ_FILE:
            path = Path(action.params["path"])
            return self.file_ops.read_file(path)
        
        elif action.type == ActionType.WRITE_FILE:
            path = Path(action.params["path"])
            content = action.params["content"]
            description = action.params.get("description", "Write file")
            self.file_ops.write_file(path, content, description)
            return f"Written {len(content)} bytes to {path}"
        
        elif action.type == ActionType.EDIT_FILE:
            path = Path(action.params["path"])
            old_content = action.params["old_content"]
            new_content = action.params["new_content"]
            result = self.file_ops.patch_file(path, old_content, new_content)
            return result
        
        elif action.type == ActionType.RUN_COMMAND:
            command = action.params["command"]
            result = self.sandbox.execute(
                command=command,
                working_dir=action.params.get("cwd"),
            )
            return result
        
        elif action.type == ActionType.SUBMIT_JOB:
            script = action.params["script"]
            result = self.sandbox.execute(
                command=f"sbatch {script}",
                working_dir=action.params.get("cwd"),
            )
            # Extract job ID
            import re
            match = re.search(r"Submitted batch job (\d+)", result.stdout)
            if match:
                job_id = match.group(1)
                return {"job_id": job_id, "output": result.stdout}
            return {"output": result.stdout, "error": result.stderr}
        
        elif action.type == ActionType.MONITOR_JOB:
            job_id = action.params["job_id"]
            info = self.job_monitor.get_job_info(job_id)
            return info.state.value if info else "UNKNOWN"
        
        elif action.type == ActionType.CHECK_HEALTH:
            health = await self.health_checker.check_all()
            return health.to_dict()
        
        elif action.type == ActionType.DIAGNOSE:
            error_log = action.params["error_log"]
            # Use recovery manager's diagnosis
            pattern = self.recovery._diagnose_error(error_log)
            if pattern:
                return {
                    "diagnosis": pattern.description,
                    "action": pattern.action.value,
                    "fix": pattern.fix_template,
                }
            return {"diagnosis": "Unknown error", "action": "manual_intervention"}
        
        elif action.type == ActionType.RECOVER:
            job_id = action.params.get("job_id")
            error_log = action.params.get("error_log")
            result = await self.recovery.handle_job_failure(
                job_id=job_id,
                error_log=error_log,
            )
            return result.to_dict()
        
        elif action.type == ActionType.THINK:
            # Just return the thought - this is for planning
            return action.params.get("thought", "")
        
        elif action.type == ActionType.RESPOND:
            # Send response to user
            response = action.params["response"]
            if self._response_callback:
                self._response_callback(response)
            return response
        
        else:
            raise ValueError(f"Unknown action type: {action.type}")
    
    def _action_to_operation(self, action_type: ActionType) -> str:
        """Convert action type to operation string for permissions."""
        return action_type.value
    
    async def execute_task(
        self,
        description: str,
        goal: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Task:
        """
        Execute a complete task.
        
        This is a simplified version - full implementation would use LLM
        for planning and decision making.
        
        Args:
            description: Task description
            goal: Goal to achieve
            context: Additional context
        
        Returns:
            Completed Task
        """
        import uuid
        
        task = Task(
            id=str(uuid.uuid4())[:8],
            description=description,
            goal=goal or description,
            context=context or {},
        )
        
        task.status = TaskStatus.RUNNING
        task.started_at = datetime.now()
        
        try:
            # For now, just acknowledge the task
            # Full implementation would use LLM for planning
            
            # Think about the task
            think_action = Action(
                type=ActionType.THINK,
                params={"thought": f"Analyzing task: {description}"},
                description="Planning approach",
            )
            result = await self.execute_action(think_action)
            task.actions.append(result)
            
            # Respond
            respond_action = Action(
                type=ActionType.RESPOND,
                params={
                    "response": f"Task acknowledged: {description}\n"
                               f"Full autonomous execution requires LLM integration."
                },
                description="Acknowledge task",
            )
            result = await self.execute_action(respond_action)
            task.actions.append(result)
            
            task.status = TaskStatus.COMPLETED
            task.result = "Task acknowledged (LLM integration required for full execution)"
            
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            logger.exception(f"Task failed: {e}")
        
        task.completed_at = datetime.now()
        self._completed_tasks.append(task)
        
        return task
    
    async def start_loop(self) -> None:
        """Start the autonomous task loop."""
        if self._running:
            return
        
        self._running = True
        
        # Start monitoring systems
        await self.job_monitor.start()
        
        # Start task loop
        self._loop_task = asyncio.create_task(self._task_loop())
        
        logger.info("Autonomous agent loop started")
    
    async def stop_loop(self) -> None:
        """Stop the autonomous task loop."""
        self._running = False
        
        await self.job_monitor.stop()
        
        if self._loop_task:
            self._loop_task.cancel()
            try:
                await self._loop_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Autonomous agent loop stopped")
    
    async def _task_loop(self) -> None:
        """Main task processing loop."""
        while self._running:
            try:
                # Get next task (with timeout to allow checking _running)
                try:
                    task = await asyncio.wait_for(
                        self._task_queue.get(),
                        timeout=1.0,
                    )
                except asyncio.TimeoutError:
                    continue
                
                self._current_task = task
                
                try:
                    await self.execute_task(
                        description=task.description,
                        goal=task.goal,
                        context=task.context,
                    )
                except Exception as e:
                    logger.exception(f"Task loop error: {e}")
                finally:
                    self._current_task = None
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Task loop error: {e}")
                await asyncio.sleep(1)
    
    def add_task(
        self,
        description: str,
        goal: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Task:
        """
        Add a task to the queue.
        
        Args:
            description: Task description
            goal: Goal to achieve
            context: Additional context
        
        Returns:
            Created Task
        """
        import uuid
        
        task = Task(
            id=str(uuid.uuid4())[:8],
            description=description,
            goal=goal or description,
            context=context or {},
        )
        
        self._task_queue.put_nowait(task)
        return task
    
    def get_status(self) -> Dict[str, Any]:
        """Get agent status."""
        return {
            "running": self._running,
            "current_task": self._current_task.to_dict() if self._current_task else None,
            "queue_size": self._task_queue.qsize(),
            "completed_tasks": len(self._completed_tasks),
            "autonomy_level": self.permissions.level.value,
        }
    
    async def check_health(self) -> SystemHealth:
        """Check system health."""
        return await self.health_checker.check_all()
    
    # =========================================================================
    # UNIFIED REASONING PIPELINE
    # =========================================================================
    
    async def process_query(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Process a user query through unified reasoning pipeline.
        
        This is the main entry point for intelligent query processing.
        Routes to appropriate reasoning path based on task complexity.
        
        Args:
            query: User's query or request
            context: Additional context (chat history, workspace state, etc.)
        
        Returns:
            Response dict with:
                - response: Text response to user
                - actions_taken: List of actions performed
                - reasoning_trace: Chain of thought (if complex task)
                - success: Whether query was handled successfully
        """
        start_time = time.time()
        context = context or {}
        
        # Add to session context
        self._session_context["actions_taken"].append({
            "type": "query_received",
            "query": query[:200],  # Truncate for context
            "timestamp": datetime.now().isoformat(),
        })
        
        # Store in memory if available
        if self.memory:
            await self._store_context(query, context)
        
        try:
            # Classify task complexity
            task_type = self._classify_task(query, context)
            
            if task_type == "simple":
                # Direct execution - no complex reasoning needed
                result = await self._execute_simple_query(query, context)
            elif task_type == "coding":
                # Code-focused task - use coding agent
                result = await self._execute_coding_task(query, context)
            else:
                # Complex task - use multi-step reasoning
                result = await self._execute_complex_query(query, context)
            
            # Record success
            self._session_context["actions_taken"].append({
                "type": "query_completed",
                "task_type": task_type,
                "success": result.get("success", True),
                "duration": time.time() - start_time,
            })
            
            return result
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Query processing failed: {error_msg}")
            
            # Record error
            self._session_context["errors_encountered"].append({
                "query": query[:200],
                "error": error_msg,
                "timestamp": datetime.now().isoformat(),
            })
            
            return {
                "response": f"I encountered an error: {error_msg}",
                "actions_taken": [],
                "success": False,
                "error": error_msg,
            }
    
    def _classify_task(self, query: str, context: Dict[str, Any]) -> str:
        """
        Classify task type to route to appropriate handler.
        
        Returns:
            - "simple": Direct tool execution
            - "coding": Code analysis/modification
            - "complex": Multi-step reasoning needed
        """
        query_lower = query.lower()
        
        # Coding indicators
        coding_keywords = [
            "fix", "debug", "error", "exception", "traceback",
            "implement", "code", "function", "class", "method",
            "refactor", "optimize", "bug", "crash",
        ]
        
        # Complex task indicators
        complex_keywords = [
            "analyze", "design", "architect", "plan",
            "compare", "evaluate", "recommend", "why",
            "how should", "what if", "strategy",
            "workflow", "pipeline", "system",
        ]
        
        # Simple task indicators
        simple_keywords = [
            "list", "show", "display", "get", "check",
            "status", "run", "execute", "submit",
            "what is", "where is", "which",
        ]
        
        # Check for error context
        if context.get("has_error") or context.get("traceback"):
            return "coding"
        
        # Check keywords
        for kw in coding_keywords:
            if kw in query_lower:
                return "coding"
        
        for kw in simple_keywords:
            if kw in query_lower:
                return "simple"
        
        for kw in complex_keywords:
            if kw in query_lower:
                return "complex"
        
        # Default to simple for short queries, complex for long
        return "simple" if len(query) < 100 else "complex"
    
    async def _execute_simple_query(
        self,
        query: str,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute simple query with direct tool calls."""
        actions_taken = []
        
        # Use enhanced tools if available
        if self.tools:
            # Parse intent and execute appropriate tool
            result = await self._direct_tool_execution(query, context)
            return {
                "response": result.get("output", "Task completed"),
                "actions_taken": result.get("actions", []),
                "success": result.get("success", True),
            }
        
        # Fallback to basic response
        return {
            "response": f"I received: {query}. Enhanced tools not available.",
            "actions_taken": [],
            "success": True,
        }
    
    async def _execute_coding_task(
        self,
        query: str,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute coding task with specialized agent."""
        if not self.coding_agent:
            # Fallback to complex reasoning
            return await self._execute_complex_query(query, context)
        
        try:
            # Extract error info from context if present
            error_info = context.get("traceback") or context.get("error")
            file_path = context.get("file_path")
            
            if error_info:
                # Diagnose and fix error
                result = await self.coding_agent.diagnose_and_fix(
                    error=error_info,
                    file_path=file_path,
                    context=context,
                )
            else:
                # General code task
                result = await self.coding_agent.process(
                    query=query,
                    context=context,
                )
            
            return {
                "response": result.get("explanation", "Code task completed"),
                "actions_taken": result.get("actions", []),
                "code_changes": result.get("changes", []),
                "success": result.get("success", True),
                "reasoning_trace": result.get("reasoning", []),
            }
            
        except Exception as e:
            logger.warning(f"CodingAgent failed, falling back: {e}")
            return await self._execute_complex_query(query, context)
    
    async def _execute_complex_query(
        self,
        query: str,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute complex query with multi-step reasoning."""
        if not self.react_agent:
            # Fallback to basic response
            return {
                "response": "Complex reasoning not available. ReactAgent not initialized.",
                "actions_taken": [],
                "success": False,
            }
        
        try:
            # Run ReAct reasoning loop
            result = await self.react_agent.reason(
                query=query,
                context=context,
                max_steps=10,
            )
            
            return {
                "response": result.get("answer", "Reasoning completed"),
                "actions_taken": result.get("actions", []),
                "reasoning_trace": result.get("steps", []),
                "success": result.get("success", True),
            }
            
        except Exception as e:
            logger.error(f"ReactAgent failed: {e}")
            return {
                "response": f"Reasoning failed: {e}",
                "actions_taken": [],
                "success": False,
                "error": str(e),
            }
    
    async def _direct_tool_execution(
        self,
        query: str,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute query using direct tool mapping."""
        query_lower = query.lower()
        
        # Map common intents to tools via registry
        if "status" in query_lower or "check" in query_lower:
            if "job" in query_lower or "slurm" in query_lower:
                tool = self.tools.get("slurm_status") if self.tools else None
                if tool:
                    job_id = context.get("job_id", "")
                    result = await tool.execute(job_id=job_id)
                    return {
                        "output": result.message if hasattr(result, 'message') else str(result),
                        "success": result.status.value == "success" if hasattr(result, 'status') else True,
                        "actions": ["slurm_status"],
                    }
            
            if "health" in query_lower:
                health = await self.check_health()
                return {"output": str(health), "success": True, "actions": ["check_health"]}
        
        if "list" in query_lower or "show" in query_lower:
            if "file" in query_lower:
                tool = self.tools.get("file_read") if self.tools else None
                if tool:
                    path = context.get("path", str(self.workspace_root))
                    result = await tool.execute(path=path)
                    return {
                        "output": result.data if hasattr(result, 'data') else str(result),
                        "success": True,
                        "actions": ["file_read"],
                    }
            
            if "job" in query_lower:
                tool = self.tools.get("slurm_status") if self.tools else None
                if tool:
                    result = await tool.execute()  # List all jobs
                    return {
                        "output": result.data if hasattr(result, 'data') else str(result),
                        "success": True,
                        "actions": ["slurm_status"],
                    }
        
        if "submit" in query_lower and ("job" in query_lower or "slurm" in query_lower):
            tool = self.tools.get("slurm_submit") if self.tools else None
            if tool:
                script = context.get("script")
                if script:
                    result = await tool.execute(script_path=script)
                    return {
                        "output": result.message if hasattr(result, 'message') else str(result),
                        "success": result.status.value == "success" if hasattr(result, 'status') else True,
                        "actions": ["slurm_submit"],
                    }
        
        if "run" in query_lower or "execute" in query_lower:
            command = context.get("command")
            if command:
                result = await self.sandbox.run(command)
                return {"output": result.stdout, "success": result.success, "actions": ["run_command"]}
        
        # Default: return query acknowledged
        return {
            "output": f"Acknowledged: {query}",
            "success": True,
            "actions": [],
        }
    
    async def _store_context(self, query: str, context: Dict[str, Any]) -> None:
        """Store query context in memory for future reference."""
        if self.memory:
            try:
                await self.memory.store({
                    "type": "query",
                    "content": query,
                    "context": context,
                    "timestamp": datetime.now().isoformat(),
                })
            except Exception as e:
                logger.debug(f"Failed to store context: {e}")
    
    async def get_relevant_context(
        self,
        query: str,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant context from memory."""
        if self.memory:
            try:
                return await self.memory.search(query, limit=limit)
            except Exception:
                pass
        return []
    
    def watch_job(
        self,
        job_id: str,
        on_complete: Optional[Callable[[JobEvent], None]] = None,
        on_failure: Optional[Callable[[JobEvent, str], None]] = None,
    ) -> None:
        """
        Watch a SLURM job.
        
        Args:
            job_id: Job to watch
            on_complete: Completion callback
            on_failure: Failure callback (receives event and error log)
        """
        async def handle_failure(event: JobEvent, error_log: str):
            # Attempt recovery
            result = await self.recovery.handle_job_failure(
                job_id=event.job_id,
                error_log=error_log,
            )
            
            # Notify
            if self._response_callback:
                status = "succeeded" if result.success else "failed"
                self._response_callback(
                    f"Recovery {status} for job {event.job_id}: {result.message}"
                )
            
            # Call user callback
            if on_failure:
                on_failure(event, error_log)
        
        self.job_monitor.watch_job(
            job_id=job_id,
            on_complete=on_complete,
            on_failure=handle_failure,
        )


# Convenience function for quick setup
def create_agent(
    workspace: Optional[Path] = None,
    level: str = "assisted",
) -> AutonomousAgent:
    """
    Create an autonomous agent.
    
    Args:
        workspace: Workspace root
        level: Autonomy level ('readonly', 'monitored', 'assisted', 'supervised', 'autonomous')
    
    Returns:
        Configured AutonomousAgent
    """
    level_map = {
        "readonly": AutonomyLevel.READONLY,
        "monitored": AutonomyLevel.MONITORED,
        "assisted": AutonomyLevel.ASSISTED,
        "supervised": AutonomyLevel.SUPERVISED,
        "autonomous": AutonomyLevel.AUTONOMOUS,
    }
    
    return AutonomousAgent(
        workspace_root=workspace,
        autonomy_level=level_map.get(level, AutonomyLevel.ASSISTED),
    )
