"""
Process Manager
================

Manage long-running background processes for the agentic system.

Features:
- Start/stop background processes
- Output capture and streaming
- Health monitoring
- Graceful shutdown with timeout
- Resource usage tracking

Example:
    pm = ProcessManager()
    
    # Start a server
    handle = pm.start_process(
        "python -m http.server 8080",
        name="http_server"
    )
    
    # Check status
    status = pm.get_status(handle)
    print(f"Running: {status.is_running}")
    
    # Get output
    output = pm.get_output(handle, lines=50)
    print(output)
    
    # Stop
    pm.stop_process(handle)
"""

import os
import signal
import subprocess
import threading
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Callable, Deque
from datetime import datetime
from pathlib import Path
from enum import Enum
import psutil

logger = logging.getLogger(__name__)


# =============================================================================
# Types
# =============================================================================

class ProcessState(Enum):
    """State of a managed process."""
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"
    UNKNOWN = "unknown"


@dataclass
class ProcessHandle:
    """Handle to a managed process."""
    id: str
    name: str
    command: str
    pid: Optional[int] = None
    start_time: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def __hash__(self):
        return hash(self.id)
        
    def __eq__(self, other):
        if isinstance(other, ProcessHandle):
            return self.id == other.id
        return False


@dataclass
class ProcessStatus:
    """Status of a managed process."""
    handle: ProcessHandle
    state: ProcessState
    is_running: bool
    exit_code: Optional[int] = None
    runtime_seconds: float = 0.0
    cpu_percent: float = 0.0
    memory_mb: float = 0.0
    last_output_line: str = ""
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.handle.id,
            "name": self.handle.name,
            "command": self.handle.command,
            "pid": self.handle.pid,
            "state": self.state.value,
            "is_running": self.is_running,
            "exit_code": self.exit_code,
            "runtime_seconds": self.runtime_seconds,
            "cpu_percent": self.cpu_percent,
            "memory_mb": self.memory_mb,
            "last_output_line": self.last_output_line,
            "error_message": self.error_message,
        }


@dataclass
class ProcessOutput:
    """Output from a managed process."""
    stdout: str
    stderr: str
    combined: str
    line_count: int
    truncated: bool = False


# =============================================================================
# Managed Process
# =============================================================================

class ManagedProcess:
    """
    A background process with output capture and monitoring.
    """
    
    # Maximum lines to keep in buffer
    MAX_BUFFER_LINES = 10000
    
    def __init__(
        self,
        command: str,
        name: str,
        cwd: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        on_exit: Optional[Callable[[int], None]] = None,
    ):
        self.command = command
        self.name = name
        self.cwd = cwd or os.getcwd()
        self.env = env
        self.on_exit = on_exit
        
        # Generate unique ID
        self.id = f"{name}_{int(time.time() * 1000)}"
        
        # Process state
        self.process: Optional[subprocess.Popen] = None
        self.state = ProcessState.STARTING
        self.start_time = datetime.now()
        self.end_time: Optional[datetime] = None
        self.exit_code: Optional[int] = None
        
        # Output buffers (thread-safe)
        self._stdout_buffer: Deque[str] = deque(maxlen=self.MAX_BUFFER_LINES)
        self._stderr_buffer: Deque[str] = deque(maxlen=self.MAX_BUFFER_LINES)
        self._combined_buffer: Deque[str] = deque(maxlen=self.MAX_BUFFER_LINES)
        self._lock = threading.Lock()
        
        # Reader threads
        self._stdout_thread: Optional[threading.Thread] = None
        self._stderr_thread: Optional[threading.Thread] = None
        self._monitor_thread: Optional[threading.Thread] = None
        
    def start(self) -> bool:
        """Start the process."""
        try:
            # Prepare environment
            exec_env = os.environ.copy()
            if self.env:
                exec_env.update(self.env)
                
            # Start process
            self.process = subprocess.Popen(
                self.command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=self.cwd,
                env=exec_env,
                text=True,
                bufsize=1,  # Line buffered
            )
            
            self.start_time = datetime.now()
            self.state = ProcessState.RUNNING
            
            # Start output readers
            self._stdout_thread = threading.Thread(
                target=self._read_stream,
                args=(self.process.stdout, self._stdout_buffer, "stdout"),
                daemon=True,
            )
            self._stderr_thread = threading.Thread(
                target=self._read_stream,
                args=(self.process.stderr, self._stderr_buffer, "stderr"),
                daemon=True,
            )
            
            self._stdout_thread.start()
            self._stderr_thread.start()
            
            # Start monitor thread
            self._monitor_thread = threading.Thread(
                target=self._monitor_process,
                daemon=True,
            )
            self._monitor_thread.start()
            
            logger.info(f"Started process '{self.name}' (PID: {self.process.pid})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start process '{self.name}': {e}")
            self.state = ProcessState.FAILED
            return False
            
    def _read_stream(
        self,
        stream,
        buffer: Deque[str],
        stream_name: str,
    ):
        """Read from a stream and add to buffer."""
        try:
            for line in stream:
                with self._lock:
                    buffer.append(line)
                    self._combined_buffer.append(f"[{stream_name}] {line}")
        except Exception as e:
            logger.debug(f"Stream reader ended: {e}")
            
    def _monitor_process(self):
        """Monitor process and handle exit."""
        if not self.process:
            return
            
        # Wait for process to complete
        self.exit_code = self.process.wait()
        self.end_time = datetime.now()
        
        if self.exit_code == 0:
            self.state = ProcessState.STOPPED
        else:
            self.state = ProcessState.FAILED
            
        logger.info(
            f"Process '{self.name}' exited with code {self.exit_code}"
        )
        
        # Call exit callback
        if self.on_exit:
            try:
                self.on_exit(self.exit_code)
            except Exception as e:
                logger.error(f"Exit callback failed: {e}")
                
    def stop(self, timeout: int = 10) -> bool:
        """
        Stop the process gracefully.
        
        Args:
            timeout: Seconds to wait for graceful shutdown
            
        Returns:
            True if stopped successfully
        """
        if not self.process or self.state == ProcessState.STOPPED:
            return True
            
        self.state = ProcessState.STOPPING
        
        try:
            # Try SIGTERM first
            self.process.terminate()
            
            # Wait for graceful shutdown
            try:
                self.process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                # Force kill
                logger.warning(f"Process '{self.name}' didn't stop, killing...")
                self.process.kill()
                self.process.wait(timeout=5)
                
            self.exit_code = self.process.returncode
            self.end_time = datetime.now()
            self.state = ProcessState.STOPPED
            
            logger.info(f"Stopped process '{self.name}'")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop process '{self.name}': {e}")
            self.state = ProcessState.UNKNOWN
            return False
            
    def get_output(
        self,
        lines: int = 100,
        stream: str = "combined",
    ) -> ProcessOutput:
        """
        Get recent output from the process.
        
        Args:
            lines: Number of lines to return
            stream: "stdout", "stderr", or "combined"
            
        Returns:
            ProcessOutput with the requested output
        """
        with self._lock:
            if stream == "stdout":
                buffer = list(self._stdout_buffer)
            elif stream == "stderr":
                buffer = list(self._stderr_buffer)
            else:
                buffer = list(self._combined_buffer)
                
        # Get last N lines
        if len(buffer) > lines:
            output_lines = buffer[-lines:]
            truncated = True
        else:
            output_lines = buffer
            truncated = False
            
        stdout_text = "".join(list(self._stdout_buffer)[-lines:])
        stderr_text = "".join(list(self._stderr_buffer)[-lines:])
        combined_text = "".join(output_lines)
        
        return ProcessOutput(
            stdout=stdout_text,
            stderr=stderr_text,
            combined=combined_text,
            line_count=len(output_lines),
            truncated=truncated,
        )
        
    def get_status(self) -> ProcessStatus:
        """Get current process status."""
        runtime = 0.0
        cpu_percent = 0.0
        memory_mb = 0.0
        last_line = ""
        
        if self.start_time:
            end = self.end_time or datetime.now()
            runtime = (end - self.start_time).total_seconds()
            
        # Get resource usage if running
        if self.process and self.state == ProcessState.RUNNING:
            try:
                proc = psutil.Process(self.process.pid)
                cpu_percent = proc.cpu_percent()
                memory_mb = proc.memory_info().rss / (1024 * 1024)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
                
        # Get last output line
        with self._lock:
            if self._combined_buffer:
                last_line = self._combined_buffer[-1].strip()
                
        return ProcessStatus(
            handle=ProcessHandle(
                id=self.id,
                name=self.name,
                command=self.command,
                pid=self.process.pid if self.process else None,
                start_time=self.start_time.isoformat(),
            ),
            state=self.state,
            is_running=self.state == ProcessState.RUNNING,
            exit_code=self.exit_code,
            runtime_seconds=runtime,
            cpu_percent=cpu_percent,
            memory_mb=memory_mb,
            last_output_line=last_line,
        )
        
    @property
    def handle(self) -> ProcessHandle:
        """Get the process handle."""
        return ProcessHandle(
            id=self.id,
            name=self.name,
            command=self.command,
            pid=self.process.pid if self.process else None,
            start_time=self.start_time.isoformat() if self.start_time else "",
        )


# =============================================================================
# Process Manager
# =============================================================================

class ProcessManager:
    """
    Manage multiple background processes.
    
    Provides a central place to start, stop, and monitor processes.
    """
    
    def __init__(
        self,
        max_processes: int = 10,
        audit_logger: Optional["AuditLogger"] = None,
    ):
        """
        Initialize the process manager.
        
        Args:
            max_processes: Maximum concurrent processes
            audit_logger: Logger for audit trail
        """
        self.max_processes = max_processes
        self.audit_logger = audit_logger
        self._processes: Dict[str, ManagedProcess] = {}
        self._lock = threading.Lock()
        
    def start_process(
        self,
        command: str,
        name: str,
        cwd: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        on_exit: Optional[Callable[[int], None]] = None,
    ) -> ProcessHandle:
        """
        Start a new managed process.
        
        Args:
            command: The command to run
            name: A name for the process
            cwd: Working directory
            env: Environment variables
            on_exit: Callback when process exits
            
        Returns:
            ProcessHandle for the new process
            
        Raises:
            ValueError: If at max capacity
        """
        with self._lock:
            # Clean up dead processes
            self._cleanup_dead_processes()
            
            # Check capacity
            running = sum(
                1 for p in self._processes.values()
                if p.state == ProcessState.RUNNING
            )
            if running >= self.max_processes:
                raise ValueError(
                    f"At maximum capacity ({self.max_processes} processes)"
                )
                
            # Create and start process
            proc = ManagedProcess(
                command=command,
                name=name,
                cwd=cwd,
                env=env,
                on_exit=on_exit,
            )
            
            if not proc.start():
                raise RuntimeError(f"Failed to start process: {name}")
                
            self._processes[proc.id] = proc
            
            # Audit log
            if self.audit_logger:
                self.audit_logger.log_process_start(proc.handle)
                
            return proc.handle
            
    def stop_process(
        self,
        handle: ProcessHandle,
        timeout: int = 10,
    ) -> bool:
        """
        Stop a managed process.
        
        Args:
            handle: The process handle
            timeout: Seconds to wait for graceful shutdown
            
        Returns:
            True if stopped successfully
        """
        with self._lock:
            proc = self._processes.get(handle.id)
            if not proc:
                logger.warning(f"Process not found: {handle.id}")
                return False
                
            success = proc.stop(timeout=timeout)
            
            # Audit log
            if self.audit_logger:
                self.audit_logger.log_process_stop(handle, proc.exit_code)
                
            return success
            
    def get_status(self, handle: ProcessHandle) -> ProcessStatus:
        """Get status of a process."""
        with self._lock:
            proc = self._processes.get(handle.id)
            if not proc:
                return ProcessStatus(
                    handle=handle,
                    state=ProcessState.UNKNOWN,
                    is_running=False,
                    error_message="Process not found",
                )
            return proc.get_status()
            
    def get_output(
        self,
        handle: ProcessHandle,
        lines: int = 100,
        stream: str = "combined",
    ) -> ProcessOutput:
        """Get output from a process."""
        with self._lock:
            proc = self._processes.get(handle.id)
            if not proc:
                return ProcessOutput(
                    stdout="",
                    stderr="Process not found",
                    combined="Process not found",
                    line_count=0,
                )
            return proc.get_output(lines=lines, stream=stream)
            
    def list_processes(self) -> List[ProcessStatus]:
        """List all managed processes."""
        with self._lock:
            return [p.get_status() for p in self._processes.values()]
            
    def stop_all(self, timeout: int = 10) -> Dict[str, bool]:
        """
        Stop all managed processes.
        
        Returns:
            Dict mapping process ID to success status
        """
        results = {}
        with self._lock:
            for proc_id, proc in self._processes.items():
                results[proc_id] = proc.stop(timeout=timeout)
        return results
        
    def _cleanup_dead_processes(self):
        """Remove processes that have been stopped for a while."""
        cutoff = datetime.now()
        to_remove = []
        
        for proc_id, proc in self._processes.items():
            if proc.state in (ProcessState.STOPPED, ProcessState.FAILED):
                if proc.end_time:
                    age = (cutoff - proc.end_time).total_seconds()
                    if age > 3600:  # 1 hour
                        to_remove.append(proc_id)
                        
        for proc_id in to_remove:
            del self._processes[proc_id]
            logger.debug(f"Cleaned up dead process: {proc_id}")


# =============================================================================
# Convenience Functions
# =============================================================================

_default_manager: Optional[ProcessManager] = None

def get_process_manager() -> ProcessManager:
    """Get the default process manager."""
    global _default_manager
    if _default_manager is None:
        _default_manager = ProcessManager()
    return _default_manager


def start_process(command: str, name: str, **kwargs) -> ProcessHandle:
    """Start a process using the default manager."""
    return get_process_manager().start_process(command, name, **kwargs)


def stop_process(handle: ProcessHandle, timeout: int = 10) -> bool:
    """Stop a process using the default manager."""
    return get_process_manager().stop_process(handle, timeout=timeout)


def get_process_status(handle: ProcessHandle) -> ProcessStatus:
    """Get process status using the default manager."""
    return get_process_manager().get_status(handle)
