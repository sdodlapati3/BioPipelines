"""
Audit Logger
=============

Complete audit trail for all agent actions.

Features:
- Log all command executions
- Log all file modifications
- Log all process starts/stops
- Log all recovery attempts
- Structured JSON format for analysis
- Rotation and retention

Example:
    audit = AuditLogger(log_dir=Path("/logs"))
    
    # Log command execution
    audit.log_command("ls -la", result)
    
    # Log file change
    audit.log_file_change("/path/to/file", "patch", "/backup/path")
    
    # Query logs
    entries = audit.get_entries(since="2024-01-01", action_type="command")
"""

import os
import json
import logging
import threading
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from pathlib import Path
from enum import Enum

logger = logging.getLogger(__name__)


# =============================================================================
# Types
# =============================================================================

class ActionType(Enum):
    """Type of audited action."""
    COMMAND = "command"
    FILE_READ = "file_read"
    FILE_WRITE = "file_write"
    FILE_PATCH = "file_patch"
    FILE_DELETE = "file_delete"
    PROCESS_START = "process_start"
    PROCESS_STOP = "process_stop"
    RECOVERY_ATTEMPT = "recovery_attempt"
    ERROR = "error"


class Severity(Enum):
    """Severity level of the action."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class AuditEntry:
    """A single audit log entry."""
    timestamp: str
    action_type: str
    description: str
    user: str = "agent"
    session_id: str = ""
    success: bool = True
    details: Dict[str, Any] = field(default_factory=dict)
    severity: str = "info"
    duration_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "action_type": self.action_type,
            "description": self.description,
            "user": self.user,
            "session_id": self.session_id,
            "success": self.success,
            "details": self.details,
            "severity": self.severity,
            "duration_ms": self.duration_ms,
        }
        
    def to_json(self) -> str:
        return json.dumps(self.to_dict())


# =============================================================================
# Audit Logger
# =============================================================================

class AuditLogger:
    """
    Comprehensive audit logging for agent actions.
    
    All actions are logged to JSON files for later analysis.
    Supports rotation and retention policies.
    """
    
    # Default retention period (days)
    DEFAULT_RETENTION_DAYS = 30
    
    # Maximum log file size (MB)
    MAX_FILE_SIZE_MB = 100
    
    def __init__(
        self,
        log_dir: Optional[Path] = None,
        session_id: str = "",
        user: str = "agent",
        retention_days: int = DEFAULT_RETENTION_DAYS,
        enabled: bool = True,
    ):
        """
        Initialize the audit logger.
        
        Args:
            log_dir: Directory for audit logs
            session_id: Current session ID
            user: User/agent identifier
            retention_days: Days to retain logs
            enabled: Whether logging is enabled
        """
        self.log_dir = Path(log_dir) if log_dir else Path.home() / ".biopipelines" / "audit"
        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.user = user
        self.retention_days = retention_days
        self.enabled = enabled
        
        self._lock = threading.Lock()
        self._current_file: Optional[Path] = None
        self._file_handle = None
        self._entry_count = 0
        
        # Create log directory
        if self.enabled:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self._rotate_if_needed()
            
    def _get_log_file(self) -> Path:
        """Get the current log file path."""
        date_str = datetime.now().strftime("%Y%m%d")
        return self.log_dir / f"audit_{date_str}.jsonl"
        
    def _rotate_if_needed(self):
        """Rotate log file if needed."""
        log_file = self._get_log_file()
        
        # Check if we need a new file
        if self._current_file != log_file:
            self._close_file()
            self._current_file = log_file
            
        # Check file size
        if log_file.exists():
            size_mb = log_file.stat().st_size / (1024 * 1024)
            if size_mb > self.MAX_FILE_SIZE_MB:
                # Add sequence number
                seq = 1
                while True:
                    new_path = log_file.with_suffix(f".{seq}.jsonl")
                    if not new_path.exists():
                        log_file.rename(new_path)
                        break
                    seq += 1
                    
    def _close_file(self):
        """Close the current file handle."""
        if self._file_handle:
            try:
                self._file_handle.close()
            except Exception:
                pass
            self._file_handle = None
            
    def _write_entry(self, entry: AuditEntry):
        """Write an entry to the log file."""
        if not self.enabled:
            return
            
        with self._lock:
            self._rotate_if_needed()
            
            try:
                with open(self._current_file, "a") as f:
                    f.write(entry.to_json() + "\n")
                self._entry_count += 1
            except Exception as e:
                logger.error(f"Failed to write audit entry: {e}")
                
    def log_command(
        self,
        command: str,
        result: "ExecutionResult",
    ):
        """
        Log a command execution.
        
        Args:
            command: The command that was executed
            result: The execution result
        """
        entry = AuditEntry(
            timestamp=datetime.now().isoformat(),
            action_type=ActionType.COMMAND.value,
            description=f"Executed: {command[:100]}{'...' if len(command) > 100 else ''}",
            user=self.user,
            session_id=self.session_id,
            success=result.success,
            details={
                "command": command,
                "exit_code": result.exit_code,
                "stdout_preview": result.stdout[:500] if result.stdout else "",
                "stderr_preview": result.stderr[:500] if result.stderr else "",
            },
            severity=Severity.INFO.value if result.success else Severity.WARNING.value,
            duration_ms=result.duration_seconds * 1000,
        )
        self._write_entry(entry)
        
    def log_file_change(
        self,
        path: str,
        change_type: str,
        backup_path: Optional[str] = None,
    ):
        """
        Log a file modification.
        
        Args:
            path: Path to the file
            change_type: Type of change (write, patch, delete)
            backup_path: Path to backup if created
        """
        action_map = {
            "write": ActionType.FILE_WRITE,
            "patch": ActionType.FILE_PATCH,
            "delete": ActionType.FILE_DELETE,
            "read": ActionType.FILE_READ,
        }
        action_type = action_map.get(change_type, ActionType.FILE_WRITE)
        
        entry = AuditEntry(
            timestamp=datetime.now().isoformat(),
            action_type=action_type.value,
            description=f"File {change_type}: {path}",
            user=self.user,
            session_id=self.session_id,
            success=True,
            details={
                "path": path,
                "change_type": change_type,
                "backup_path": backup_path,
            },
            severity=Severity.INFO.value,
        )
        self._write_entry(entry)
        
    def log_process_start(self, handle: "ProcessHandle"):
        """Log a process start."""
        entry = AuditEntry(
            timestamp=datetime.now().isoformat(),
            action_type=ActionType.PROCESS_START.value,
            description=f"Started process: {handle.name}",
            user=self.user,
            session_id=self.session_id,
            success=True,
            details={
                "process_id": handle.id,
                "name": handle.name,
                "command": handle.command,
                "pid": handle.pid,
            },
            severity=Severity.INFO.value,
        )
        self._write_entry(entry)
        
    def log_process_stop(
        self,
        handle: "ProcessHandle",
        exit_code: Optional[int] = None,
    ):
        """Log a process stop."""
        entry = AuditEntry(
            timestamp=datetime.now().isoformat(),
            action_type=ActionType.PROCESS_STOP.value,
            description=f"Stopped process: {handle.name}",
            user=self.user,
            session_id=self.session_id,
            success=exit_code == 0 if exit_code is not None else True,
            details={
                "process_id": handle.id,
                "name": handle.name,
                "exit_code": exit_code,
            },
            severity=Severity.INFO.value if exit_code == 0 else Severity.WARNING.value,
        )
        self._write_entry(entry)
        
    def log_recovery_attempt(
        self,
        job_id: str,
        error_type: str,
        action: str,
        success: bool,
        details: Dict[str, Any],
    ):
        """Log a recovery attempt."""
        entry = AuditEntry(
            timestamp=datetime.now().isoformat(),
            action_type=ActionType.RECOVERY_ATTEMPT.value,
            description=f"Recovery for job {job_id}: {action}",
            user=self.user,
            session_id=self.session_id,
            success=success,
            details={
                "job_id": job_id,
                "error_type": error_type,
                "action": action,
                **details,
            },
            severity=Severity.INFO.value if success else Severity.WARNING.value,
        )
        self._write_entry(entry)
        
    def log_error(
        self,
        description: str,
        error: str,
        details: Optional[Dict[str, Any]] = None,
    ):
        """Log an error."""
        entry = AuditEntry(
            timestamp=datetime.now().isoformat(),
            action_type=ActionType.ERROR.value,
            description=description,
            user=self.user,
            session_id=self.session_id,
            success=False,
            details={
                "error": error,
                **(details or {}),
            },
            severity=Severity.ERROR.value,
        )
        self._write_entry(entry)
        
    def get_entries(
        self,
        since: Optional[str] = None,
        until: Optional[str] = None,
        action_type: Optional[str] = None,
        success_only: bool = False,
        limit: int = 1000,
    ) -> List[AuditEntry]:
        """
        Query audit entries.
        
        Args:
            since: Start date (ISO format)
            until: End date (ISO format)
            action_type: Filter by action type
            success_only: Only return successful entries
            limit: Maximum entries to return
            
        Returns:
            List of matching AuditEntry objects
        """
        entries = []
        
        # Parse date filters
        since_dt = datetime.fromisoformat(since) if since else None
        until_dt = datetime.fromisoformat(until) if until else None
        
        # Find relevant log files
        log_files = sorted(self.log_dir.glob("audit_*.jsonl"))
        
        for log_file in log_files:
            try:
                with open(log_file, "r") as f:
                    for line in f:
                        if len(entries) >= limit:
                            break
                            
                        try:
                            data = json.loads(line.strip())
                            entry_dt = datetime.fromisoformat(data["timestamp"])
                            
                            # Apply filters
                            if since_dt and entry_dt < since_dt:
                                continue
                            if until_dt and entry_dt > until_dt:
                                continue
                            if action_type and data["action_type"] != action_type:
                                continue
                            if success_only and not data["success"]:
                                continue
                                
                            entries.append(AuditEntry(**data))
                            
                        except (json.JSONDecodeError, KeyError):
                            continue
                            
            except Exception as e:
                logger.warning(f"Failed to read log file {log_file}: {e}")
                
        return entries
        
    def cleanup_old_logs(self):
        """Remove logs older than retention period."""
        if not self.enabled:
            return
            
        cutoff = datetime.now() - timedelta(days=self.retention_days)
        
        for log_file in self.log_dir.glob("audit_*.jsonl"):
            try:
                # Parse date from filename
                date_str = log_file.stem.split("_")[1]
                file_date = datetime.strptime(date_str, "%Y%m%d")
                
                if file_date < cutoff:
                    log_file.unlink()
                    logger.info(f"Cleaned up old audit log: {log_file}")
                    
            except (ValueError, IndexError):
                continue
                
    def get_summary(
        self,
        since: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get a summary of audit entries.
        
        Args:
            since: Start date (default: today)
            
        Returns:
            Summary statistics
        """
        if not since:
            since = datetime.now().replace(hour=0, minute=0, second=0).isoformat()
            
        entries = self.get_entries(since=since, limit=10000)
        
        summary = {
            "total_entries": len(entries),
            "successful": sum(1 for e in entries if e.success),
            "failed": sum(1 for e in entries if not e.success),
            "by_action_type": {},
            "by_severity": {},
        }
        
        for entry in entries:
            # Count by action type
            at = entry.action_type
            summary["by_action_type"][at] = summary["by_action_type"].get(at, 0) + 1
            
            # Count by severity
            sev = entry.severity
            summary["by_severity"][sev] = summary["by_severity"].get(sev, 0) + 1
            
        return summary


# =============================================================================
# Convenience Functions
# =============================================================================

_default_logger: Optional[AuditLogger] = None

def get_audit_logger() -> AuditLogger:
    """Get the default audit logger."""
    global _default_logger
    if _default_logger is None:
        _default_logger = AuditLogger()
    return _default_logger


def log_command(command: str, result: "ExecutionResult"):
    """Log a command using the default logger."""
    get_audit_logger().log_command(command, result)


def log_file_change(path: str, change_type: str, backup_path: Optional[str] = None):
    """Log a file change using the default logger."""
    get_audit_logger().log_file_change(path, change_type, backup_path)
