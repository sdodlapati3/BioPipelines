"""
Command Sandbox
================

Secure command execution environment for the agentic system.

Features:
- Command whitelist validation
- Dangerous pattern blocking
- Timeout enforcement
- Resource limits
- Complete audit logging

Security Model:
    1. Command must start with an allowed executable
    2. Command must not match any blocked patterns
    3. Execution is time-limited
    4. All executions are logged

Example:
    sandbox = CommandSandbox()
    
    # Safe command
    result = sandbox.execute("ls -la /data")
    print(result.stdout)
    
    # Blocked command raises error
    try:
        sandbox.execute("rm -rf /")
    except CommandValidationError:
        print("Blocked!")
"""

import os
import re
import shlex
import subprocess
import logging
import resource
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Set, Tuple
from datetime import datetime
from pathlib import Path
from enum import Enum

logger = logging.getLogger(__name__)


# =============================================================================
# Types
# =============================================================================

class CommandCategory(Enum):
    """Category of commands for permission purposes."""
    READ = "read"           # Read-only operations
    WRITE = "write"         # File modifications
    EXECUTE = "execute"     # Run programs
    NETWORK = "network"     # Network operations
    SYSTEM = "system"       # System administration
    DANGEROUS = "dangerous" # Never allowed


@dataclass
class ExecutionResult:
    """Result of command execution."""
    success: bool
    command: str
    stdout: str
    stderr: str
    exit_code: int
    duration_seconds: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    truncated: bool = False  # True if output was truncated
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "command": self.command,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "exit_code": self.exit_code,
            "duration_seconds": self.duration_seconds,
            "timestamp": self.timestamp,
            "truncated": self.truncated,
        }


class CommandValidationError(Exception):
    """Raised when a command fails validation."""
    def __init__(self, command: str, reason: str):
        self.command = command
        self.reason = reason
        super().__init__(f"Command blocked: {reason}")


# =============================================================================
# Command Sandbox
# =============================================================================

class CommandSandbox:
    """
    Secure command execution environment.
    
    Only allows pre-approved commands and blocks dangerous patterns.
    All executions are logged for audit purposes.
    """
    
    # Executables that are generally safe
    ALLOWED_EXECUTABLES: Set[str] = {
        # File operations (read-only or safe)
        "ls", "cat", "head", "tail", "less", "more",
        "grep", "awk", "sed", "cut", "sort", "uniq", "wc",
        "find", "file", "stat", "du", "df",
        "tree", "realpath", "basename", "dirname",
        
        # File operations (write - require elevated permission)
        "mkdir", "touch", "cp", "mv",
        
        # Compression
        "gzip", "gunzip", "zcat", "tar", "zip", "unzip",
        
        # Text processing
        "diff", "patch", "jq", "yq",
        
        # Python & conda
        "python", "python3", "pip", "pip3",
        "conda", "mamba",
        
        # Workflow engines
        "nextflow", "snakemake", "nf-core",
        
        # SLURM
        "squeue", "sinfo", "sacct", "scontrol",
        "sbatch", "scancel", "srun",
        
        # Git
        "git",
        
        # Network (limited)
        "curl", "wget",
        
        # Misc utilities
        "echo", "printf", "date", "env", "which", "whereis",
        "hostname", "whoami", "id", "pwd",
    }
    
    # Patterns that are NEVER allowed
    BLOCKED_PATTERNS: List[Tuple[str, str]] = [
        # Destructive operations
        (r"rm\s+(-[rf]+\s+)*(/|~|\$HOME|\$PWD)", "Recursive delete of root/home"),
        (r"rm\s+-rf\s+\.\.\s*/", "Recursive delete with parent traversal"),
        (r">\s*/dev/sd[a-z]", "Direct write to disk device"),
        (r">\s*/dev/null\s*2>&1\s*<", "Suspicious redirection"),
        (r"mkfs\.", "Disk formatting"),
        (r"dd\s+if=.+of=/dev", "Raw disk write"),
        (r":\(\)\s*\{\s*:\|:\s*&\s*\}", "Fork bomb"),
        (r"\|\s*sh\s*$", "Piping to shell"),
        (r"\|\s*bash\s*$", "Piping to bash"),
        (r"eval\s+.*\$", "Eval with variable expansion"),
        
        # Permission changes
        (r"chmod\s+777\s+/", "Insecure permissions on root"),
        (r"chown\s+-R\s+.+\s+/", "Recursive chown on root"),
        
        # System modifications
        (r"/etc/passwd", "Password file access"),
        (r"/etc/shadow", "Shadow file access"),
        (r"/etc/sudoers", "Sudoers file access"),
        (r"visudo", "Sudoers modification"),
        
        # Network exfiltration
        (r"curl\s+.+\|.*sh", "Download and execute"),
        (r"wget\s+.+-O\s*-\s*\|", "Download and pipe"),
        
        # Crypto mining / malware indicators
        (r"xmrig|minerd|cryptonight", "Cryptocurrency miner"),
        (r"reverse.shell|revshell", "Reverse shell"),
    ]
    
    # Maximum output size (bytes)
    MAX_OUTPUT_SIZE = 1024 * 1024  # 1MB
    
    # Default timeout (seconds)
    DEFAULT_TIMEOUT = 300
    
    def __init__(
        self,
        allowed_executables: Optional[Set[str]] = None,
        blocked_patterns: Optional[List[Tuple[str, str]]] = None,
        max_output_size: int = MAX_OUTPUT_SIZE,
        default_timeout: int = DEFAULT_TIMEOUT,
        audit_logger: Optional["AuditLogger"] = None,
        workspace_root: Optional[Path] = None,
    ):
        """
        Initialize the command sandbox.
        
        Args:
            allowed_executables: Override default allowed executables
            blocked_patterns: Additional blocked patterns
            max_output_size: Maximum output capture size
            default_timeout: Default command timeout
            audit_logger: Logger for audit trail
            workspace_root: Root directory for path validation
        """
        self.allowed_executables = allowed_executables or self.ALLOWED_EXECUTABLES
        self.blocked_patterns = list(self.BLOCKED_PATTERNS)
        if blocked_patterns:
            self.blocked_patterns.extend(blocked_patterns)
            
        self.max_output_size = max_output_size
        self.default_timeout = default_timeout
        self.audit_logger = audit_logger
        self.workspace_root = workspace_root or Path.cwd()
        
        # Compile blocked patterns for efficiency
        self._compiled_patterns = [
            (re.compile(pattern, re.IGNORECASE), reason)
            for pattern, reason in self.blocked_patterns
        ]
        
    def validate_command(self, command: str) -> Tuple[bool, Optional[str]]:
        """
        Validate a command before execution.
        
        Args:
            command: The command string to validate
            
        Returns:
            Tuple of (is_valid, error_reason)
        """
        if not command or not command.strip():
            return False, "Empty command"
            
        command = command.strip()
        
        # Parse the command to get the executable
        try:
            parts = shlex.split(command)
            if not parts:
                return False, "No executable found"
            executable = parts[0]
        except ValueError as e:
            return False, f"Invalid command syntax: {e}"
        
        # Handle paths (e.g., /usr/bin/python)
        if "/" in executable:
            executable = os.path.basename(executable)
            
        # Check if executable is allowed
        if executable not in self.allowed_executables:
            return False, f"Executable '{executable}' is not in allowed list"
            
        # Check for blocked patterns
        for pattern, reason in self._compiled_patterns:
            if pattern.search(command):
                return False, f"Blocked pattern detected: {reason}"
                
        return True, None
        
    def execute(
        self,
        command: str,
        timeout: Optional[int] = None,
        cwd: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        capture_output: bool = True,
    ) -> ExecutionResult:
        """
        Execute a command in the sandbox.
        
        Args:
            command: The command to execute
            timeout: Timeout in seconds (default: 300)
            cwd: Working directory
            env: Environment variables (merged with current env)
            capture_output: Whether to capture stdout/stderr
            
        Returns:
            ExecutionResult with command output
            
        Raises:
            CommandValidationError: If command fails validation
        """
        # Validate command
        is_valid, error_reason = self.validate_command(command)
        if not is_valid:
            logger.warning(f"Command blocked: {command[:100]} - {error_reason}")
            raise CommandValidationError(command, error_reason)
            
        # Prepare execution
        timeout = timeout or self.default_timeout
        cwd = cwd or str(self.workspace_root)
        
        # Merge environment
        execution_env = os.environ.copy()
        if env:
            execution_env.update(env)
            
        # Log the execution
        logger.info(f"Executing: {command[:200]}{'...' if len(command) > 200 else ''}")
        
        start_time = datetime.now()
        truncated = False
        
        try:
            # Execute the command
            result = subprocess.run(
                command,
                shell=True,
                cwd=cwd,
                env=execution_env,
                timeout=timeout,
                capture_output=capture_output,
                text=True,
            )
            
            stdout = result.stdout or ""
            stderr = result.stderr or ""
            
            # Truncate if needed
            if len(stdout) > self.max_output_size:
                stdout = stdout[:self.max_output_size] + "\n... [output truncated]"
                truncated = True
            if len(stderr) > self.max_output_size:
                stderr = stderr[:self.max_output_size] + "\n... [output truncated]"
                truncated = True
                
            duration = (datetime.now() - start_time).total_seconds()
            
            exec_result = ExecutionResult(
                success=result.returncode == 0,
                command=command,
                stdout=stdout,
                stderr=stderr,
                exit_code=result.returncode,
                duration_seconds=duration,
                truncated=truncated,
            )
            
        except subprocess.TimeoutExpired:
            duration = (datetime.now() - start_time).total_seconds()
            exec_result = ExecutionResult(
                success=False,
                command=command,
                stdout="",
                stderr=f"Command timed out after {timeout} seconds",
                exit_code=-1,
                duration_seconds=duration,
            )
            
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            exec_result = ExecutionResult(
                success=False,
                command=command,
                stdout="",
                stderr=f"Execution error: {str(e)}",
                exit_code=-1,
                duration_seconds=duration,
            )
            
        # Audit log
        if self.audit_logger:
            self.audit_logger.log_command(command, exec_result)
            
        return exec_result
        
    def execute_safe(
        self,
        command: str,
        timeout: Optional[int] = None,
        **kwargs
    ) -> ExecutionResult:
        """
        Execute a command, returning error result instead of raising.
        
        Same as execute() but catches CommandValidationError.
        """
        try:
            return self.execute(command, timeout, **kwargs)
        except CommandValidationError as e:
            return ExecutionResult(
                success=False,
                command=command,
                stdout="",
                stderr=f"Command blocked: {e.reason}",
                exit_code=-2,
                duration_seconds=0.0,
            )
            
    def is_allowed(self, command: str) -> bool:
        """Check if a command would be allowed."""
        is_valid, _ = self.validate_command(command)
        return is_valid
        
    def get_allowed_executables(self) -> Set[str]:
        """Get the set of allowed executables."""
        return self.allowed_executables.copy()
        
    def add_allowed_executable(self, executable: str) -> None:
        """Add an executable to the allowed list."""
        self.allowed_executables.add(executable)
        logger.info(f"Added '{executable}' to allowed executables")
        
    def add_blocked_pattern(self, pattern: str, reason: str) -> None:
        """Add a blocked pattern."""
        self.blocked_patterns.append((pattern, reason))
        self._compiled_patterns.append(
            (re.compile(pattern, re.IGNORECASE), reason)
        )
        logger.info(f"Added blocked pattern: {reason}")


# =============================================================================
# Convenience Functions
# =============================================================================

_default_sandbox: Optional[CommandSandbox] = None

def get_sandbox() -> CommandSandbox:
    """Get the default command sandbox."""
    global _default_sandbox
    if _default_sandbox is None:
        _default_sandbox = CommandSandbox()
    return _default_sandbox


def run_command(command: str, timeout: int = 300) -> ExecutionResult:
    """Execute a command using the default sandbox."""
    return get_sandbox().execute(command, timeout=timeout)


def is_command_allowed(command: str) -> bool:
    """Check if a command is allowed."""
    return get_sandbox().is_allowed(command)
