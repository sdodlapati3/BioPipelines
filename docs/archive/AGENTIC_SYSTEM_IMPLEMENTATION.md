# BioPipelines Agentic AI System - Implementation Plan

**Created**: November 27, 2025  
**Status**: Phase 1 & 3 COMPLETE ✅  
**Goal**: Build a professional, fully autonomous coding agent that can diagnose, fix, and resolve issues end-to-end.

---

## Executive Summary

Transform BioPipelines from an "advisory" system to a **fully autonomous agentic AI** that can:
1. ✅ Execute shell commands safely
2. ✅ Read, write, and patch files
3. ✅ Monitor jobs continuously
4. ✅ Apply fixes automatically
5. 🔄 Learn from failures (memory system exists)
6. ✅ Operate in a sandboxed, safe environment

---

## Implementation Status

| Phase | Description | Status |
|-------|-------------|--------|
| Phase 1 | Tool Executor Layer | ✅ COMPLETE |
| Phase 2 | Tools Layer | ⏳ Merged into Phase 1 |
| Phase 3 | Autonomous Agent | ✅ COMPLETE |
| Phase 4 | Integration | 🔄 Pending |

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                      USER INTERFACE (Gradio)                     │
├─────────────────────────────────────────────────────────────────┤
│                     AGENT ORCHESTRATOR                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │   Router    │  │  ReAct      │  │   Memory    │              │
│  │  (Intent)   │  │  (Reason)   │  │  (Learn)    │              │
│  └─────────────┘  └─────────────┘  └─────────────┘              │
├─────────────────────────────────────────────────────────────────┤
│                    SPECIALIZED AGENTS                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │   Coding    │  │   Data      │  │  Workflow   │              │
│  │   Agent     │  │   Agent     │  │   Agent     │              │
│  └─────────────┘  └─────────────┘  └─────────────┘              │
├─────────────────────────────────────────────────────────────────┤
│                    TOOL EXECUTOR LAYER (NEW)                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │  Sandbox    │  │  File Ops   │  │  Process    │              │
│  │  (Security) │  │  (Edit)     │  │  Manager    │              │
│  └─────────────┘  └─────────────┘  └─────────────┘              │
├─────────────────────────────────────────────────────────────────┤
│                    AUTONOMOUS LOOP (NEW)                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │  Job        │  │  Health     │  │  Recovery   │              │
│  │  Monitor    │  │  Checker    │  │  Manager    │              │
│  └─────────────┘  └─────────────┘  └─────────────┘              │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    EXTERNAL SYSTEMS                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │   SLURM     │  │  Nextflow   │  │  File       │              │
│  │   Cluster   │  │  Pipelines  │  │  System     │              │
│  └─────────────┘  └─────────────┘  └─────────────┘              │
└─────────────────────────────────────────────────────────────────┘
```

---

## Implementation Phases

### Phase 1: Tool Executor Layer (Core Infrastructure)

**Duration**: ~4 hours  
**Priority**: Critical - Everything else depends on this

#### 1.1 Sandbox Module (`agents/executor/sandbox.py`)

Safe command execution with:
- Allowed command whitelist
- Blocked patterns (rm -rf /, etc.)
- Timeout enforcement
- Resource limits
- Audit logging

```python
class CommandSandbox:
    """Secure command execution environment."""
    
    ALLOWED_COMMANDS = {
        'ls', 'cat', 'head', 'tail', 'grep', 'find', 'wc',
        'python', 'nextflow', 'snakemake',
        'squeue', 'sinfo', 'sacct', 'scancel', 'sbatch',
        'conda', 'pip', 'git',
        'mkdir', 'cp', 'mv', 'touch',
        'curl', 'wget',
    }
    
    BLOCKED_PATTERNS = [
        r'rm\s+-rf\s+/',        # Dangerous delete
        r'>\s*/dev/sd',          # Write to disk
        r'mkfs',                 # Format disk
        r'dd\s+if=',             # Raw disk ops
        r'chmod\s+777',          # Insecure permissions
        r':\(\)\s*{\s*:\|:',     # Fork bomb
    ]
    
    def execute(self, command: str, timeout: int = 300) -> ExecutionResult:
        """Execute command in sandbox."""
        # 1. Validate command
        # 2. Check against blocked patterns
        # 3. Execute with timeout
        # 4. Log execution
        # 5. Return result
```

#### 1.2 File Operations Module (`agents/executor/file_ops.py`)

Safe file operations with:
- Automatic backup before changes
- Path validation (no escape from workspace)
- Atomic writes
- Diff generation

```python
class FileOperations:
    """Safe file operations with backup and validation."""
    
    def __init__(self, workspace_root: Path, backup_dir: Path):
        self.workspace = workspace_root
        self.backup_dir = backup_dir
        
    def read_file(self, path: str, max_lines: int = 1000) -> FileContent:
        """Read file with size limits."""
        
    def write_file(self, path: str, content: str, backup: bool = True) -> WriteResult:
        """Write file with automatic backup."""
        
    def patch_file(self, path: str, old: str, new: str) -> PatchResult:
        """Apply targeted patch to file."""
        
    def apply_diff(self, path: str, diff: str) -> DiffResult:
        """Apply unified diff to file."""
```

#### 1.3 Process Manager (`agents/executor/process_manager.py`)

Manage long-running processes:
- Start/stop background processes
- Monitor process health
- Capture output streams
- Handle signals

```python
class ProcessManager:
    """Manage long-running processes."""
    
    def start_process(self, command: str, name: str) -> ProcessHandle:
        """Start a managed background process."""
        
    def stop_process(self, handle: ProcessHandle) -> bool:
        """Gracefully stop a process."""
        
    def get_output(self, handle: ProcessHandle, lines: int = 100) -> str:
        """Get recent output from process."""
        
    def health_check(self, handle: ProcessHandle) -> HealthStatus:
        """Check if process is healthy."""
```

---

### Phase 2: Enhanced Tool Definitions

**Duration**: ~2 hours

#### 2.1 New Tool Enum Values

Add to `tools.py`:

```python
class ToolName(Enum):
    # Existing tools...
    SCAN_DATA = "scan_data"
    SEARCH_DATABASES = "search_databases"
    # ...
    
    # NEW: File operations
    READ_FILE = "read_file"
    WRITE_FILE = "write_file"
    PATCH_FILE = "patch_file"
    LIST_DIRECTORY = "list_directory"
    
    # NEW: Command execution
    RUN_COMMAND = "run_command"
    RUN_PYTHON = "run_python"
    
    # NEW: Process management
    START_PROCESS = "start_process"
    STOP_PROCESS = "stop_process"
    CHECK_PROCESS = "check_process"
    
    # NEW: Monitoring
    WATCH_JOB = "watch_job"
    TAIL_LOG = "tail_log"
```

#### 2.2 Tool Patterns Update

```python
NEW_TOOL_PATTERNS = [
    # File operations
    (r"read (?:file|contents? of)\s+(.+)", ToolName.READ_FILE),
    (r"(?:show|display|cat)\s+(.+)", ToolName.READ_FILE),
    (r"write (?:to\s+)?(.+?):\s*```(.+)```", ToolName.WRITE_FILE),
    (r"(?:edit|modify|patch|fix)\s+(.+)", ToolName.PATCH_FILE),
    (r"list (?:files in\s+)?(.+)", ToolName.LIST_DIRECTORY),
    
    # Command execution
    (r"run(?: command)?:\s*`(.+)`", ToolName.RUN_COMMAND),
    (r"execute:\s*(.+)", ToolName.RUN_COMMAND),
    
    # Monitoring
    (r"watch job (\d+)", ToolName.WATCH_JOB),
    (r"tail (?:log\s+)?(.+)", ToolName.TAIL_LOG),
]
```

---

### Phase 3: Autonomous Loop System

**Duration**: ~3 hours

#### 3.1 Job Monitor (`agents/autonomous/job_monitor.py`)

Continuous job monitoring:

```python
class JobMonitor:
    """Continuously monitors SLURM jobs and triggers actions."""
    
    def __init__(self, check_interval: int = 30):
        self.interval = check_interval
        self.watched_jobs: Dict[str, JobWatch] = {}
        self.callbacks: Dict[str, List[Callable]] = {}
        
    async def start(self):
        """Start the monitoring loop."""
        while True:
            await self._check_all_jobs()
            await asyncio.sleep(self.interval)
            
    async def _check_all_jobs(self):
        """Check status of all watched jobs."""
        for job_id, watch in self.watched_jobs.items():
            status = await self._get_job_status(job_id)
            if status != watch.last_status:
                await self._trigger_callbacks(job_id, status)
                
    def on_failure(self, job_id: str, callback: Callable):
        """Register callback for job failure."""
        
    def on_completion(self, job_id: str, callback: Callable):
        """Register callback for job completion."""
```

#### 3.2 Health Checker (`agents/autonomous/health_checker.py`)

System health monitoring:

```python
class HealthChecker:
    """Monitors overall system health."""
    
    def check_vllm_server(self) -> HealthStatus:
        """Check if vLLM server is responding."""
        
    def check_disk_space(self) -> HealthStatus:
        """Check available disk space."""
        
    def check_gpu_memory(self) -> HealthStatus:
        """Check GPU memory usage."""
        
    def check_slurm_queue(self) -> HealthStatus:
        """Check SLURM queue health."""
```

#### 3.3 Recovery Manager (`agents/autonomous/recovery.py`)

Automated recovery:

```python
class RecoveryManager:
    """Manages automated recovery from failures."""
    
    def __init__(self, coding_agent: CodingAgent, file_ops: FileOperations):
        self.coding_agent = coding_agent
        self.file_ops = file_ops
        
    async def handle_failure(self, job_id: str, error_log: str) -> RecoveryResult:
        """Handle a job failure with automated recovery."""
        # 1. Diagnose error using CodingAgent
        diagnosis = await self.coding_agent.diagnose(error_log)
        
        # 2. Generate fix
        if diagnosis.auto_fixable:
            fix = await self.coding_agent.generate_fix(diagnosis)
            
            # 3. Apply fix
            result = await self.file_ops.patch_file(
                fix.file_path, 
                fix.original_code, 
                fix.fixed_code
            )
            
            # 4. Retry job
            if result.success:
                new_job_id = await self._resubmit_job(job_id)
                return RecoveryResult(success=True, retry_job_id=new_job_id)
                
        return RecoveryResult(success=False, requires_human=True)
```

---

### Phase 4: Integration Layer

**Duration**: ~2 hours

#### 4.1 Enhanced Orchestrator

Update `orchestrator.py` to use new capabilities:

```python
class AgentOrchestrator:
    """Enhanced orchestrator with autonomous capabilities."""
    
    def __init__(self):
        # Existing components
        self.router = AgentRouter()
        self.coding_agent = CodingAgent()
        
        # NEW: Executor layer
        self.sandbox = CommandSandbox()
        self.file_ops = FileOperations()
        self.process_manager = ProcessManager()
        
        # NEW: Autonomous components
        self.job_monitor = JobMonitor()
        self.health_checker = HealthChecker()
        self.recovery_manager = RecoveryManager()
        
    async def start_autonomous_mode(self):
        """Start autonomous monitoring and recovery."""
        await asyncio.gather(
            self.job_monitor.start(),
            self.health_checker.start(),
        )
```

#### 4.2 MCP-Style Tool Schema

Define tools in MCP (Model Context Protocol) format for better LLM integration:

```python
TOOL_SCHEMAS = [
    {
        "name": "run_command",
        "description": "Execute a shell command in a sandboxed environment",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute"
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in seconds",
                    "default": 300
                }
            },
            "required": ["command"]
        }
    },
    {
        "name": "patch_file",
        "description": "Apply a targeted fix to a file",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string"},
                "old_content": {"type": "string"},
                "new_content": {"type": "string"},
                "explanation": {"type": "string"}
            },
            "required": ["file_path", "old_content", "new_content"]
        }
    },
    # ... more tools
]
```

---

### Phase 5: Safety & Audit Layer

**Duration**: ~1.5 hours

#### 5.1 Audit Logger (`agents/executor/audit.py`)

Comprehensive audit trail:

```python
class AuditLogger:
    """Log all agent actions for security and debugging."""
    
    def __init__(self, log_path: Path):
        self.log_path = log_path
        
    def log_command(self, command: str, result: ExecutionResult, user: str):
        """Log command execution."""
        
    def log_file_change(self, path: str, change_type: str, backup_path: str):
        """Log file modifications."""
        
    def log_recovery_attempt(self, job_id: str, diagnosis: DiagnosisResult):
        """Log automated recovery attempts."""
```

#### 5.2 Permission System

```python
class PermissionManager:
    """Manage what the agent is allowed to do."""
    
    # Levels of autonomy
    LEVELS = {
        "read_only": ["read_file", "list_directory", "run_command:safe"],
        "suggest": ["read_file", "list_directory", "run_command:safe", "generate_fix"],
        "apply_with_approval": ["*", "-delete", "-format"],
        "full_autonomous": ["*"],
    }
    
    def __init__(self, level: str = "apply_with_approval"):
        self.level = level
        
    def check_permission(self, action: str) -> bool:
        """Check if action is allowed at current level."""
```

---

## File Structure After Implementation

```
src/workflow_composer/agents/
├── __init__.py                 # Updated exports
├── bridge.py                   # Existing
├── chat_integration.py         # Updated for new tools
├── coding_agent.py             # Enhanced with file ops
├── context.py                  # Existing
├── memory.py                   # Existing
├── multi_model.py              # Existing
├── orchestrator.py             # Updated with autonomous mode
├── react_agent.py              # Updated tool definitions
├── router.py                   # Existing
├── self_healing.py             # Updated to use recovery manager
├── tools.py                    # Updated with new tools
│
├── executor/                   # NEW: Tool execution layer
│   ├── __init__.py
│   ├── sandbox.py              # Command sandbox
│   ├── file_ops.py             # File operations
│   ├── process_manager.py      # Process management
│   ├── audit.py                # Audit logging
│   └── permissions.py          # Permission management
│
└── autonomous/                 # NEW: Autonomous operation
    ├── __init__.py
    ├── job_monitor.py          # Job monitoring
    ├── health_checker.py       # System health
    └── recovery.py             # Automated recovery
```

---

## Implementation Order

```
Day 1 (Today):
├── [x] Create implementation plan (this document)
├── [ ] Phase 1.1: sandbox.py
├── [ ] Phase 1.2: file_ops.py
├── [ ] Phase 1.3: process_manager.py
└── [ ] Phase 2: Update tools.py with new tools

Day 2:
├── [ ] Phase 3.1: job_monitor.py
├── [ ] Phase 3.2: health_checker.py
├── [ ] Phase 3.3: recovery.py
├── [ ] Phase 4: Integration
└── [ ] Phase 5: Safety & audit

Day 3:
├── [ ] Testing & validation
├── [ ] UI integration
├── [ ] Documentation
└── [ ] Demo end-to-end flow
```

---

## Success Criteria

1. **Agent can execute commands**: Run `squeue`, `tail`, `python` safely
2. **Agent can edit files**: Read, write, patch files with backup
3. **Agent can recover from failures**: Detect failure → diagnose → fix → retry
4. **All actions are audited**: Complete trail of what agent did
5. **Human can override**: Approval workflow for sensitive operations
6. **Tests pass**: Unit tests for each component

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Agent deletes important files | Automatic backup before any write; blocked patterns |
| Agent runs dangerous commands | Whitelist + blocked patterns + resource limits |
| Agent gets stuck in loop | Max retry limits; human escalation after N failures |
| Security vulnerability | Permission levels; audit logging; code review |
| Model hallucinates fixes | Validate fixes before applying; dry-run mode |

---

*Let's begin implementation!*
