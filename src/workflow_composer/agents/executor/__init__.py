"""
Tool Executor Layer
====================

Safe, audited execution of agent actions:
- CommandSandbox: Secure shell command execution
- FileOperations: Safe file read/write/patch with backups
- ProcessManager: Manage long-running background processes
- AuditLogger: Complete audit trail of all actions
- PermissionManager: Control what the agent can do

Example:
    from workflow_composer.agents.executor import (
        CommandSandbox,
        FileOperations,
        ProcessManager,
        PermissionManager,
        AutonomyLevel,
    )
    
    sandbox = CommandSandbox()
    result = sandbox.execute("ls -la /data")
    
    file_ops = FileOperations(workspace=Path("/project"))
    content = file_ops.read_file("config.yaml")
    file_ops.patch_file("config.yaml", old="param: 1", new="param: 2")
    
    pm = PermissionManager(autonomy_level=AutonomyLevel.ASSISTED)
"""

from .sandbox import CommandSandbox, ExecutionResult, CommandValidationError
from .file_ops import FileOperations, FileContent, WriteResult, PatchResult
from .process_manager import ProcessManager, ProcessHandle, ProcessStatus
from .audit import AuditLogger, AuditEntry
from .permissions import PermissionManager, PermissionLevel, AutonomyLevel

__all__ = [
    # Sandbox
    "CommandSandbox",
    "ExecutionResult", 
    "CommandValidationError",
    
    # File operations
    "FileOperations",
    "FileContent",
    "WriteResult",
    "PatchResult",
    
    # Process management
    "ProcessManager",
    "ProcessHandle",
    "ProcessStatus",
    
    # Audit
    "AuditLogger",
    "AuditEntry",
    
    # Permissions
    "PermissionManager",
    "PermissionLevel",
    "AutonomyLevel",
]
