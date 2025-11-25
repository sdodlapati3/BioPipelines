"""
Auto-fix execution engine for applying suggested fixes.

Handles safe automatic fixes and user-confirmed fixes with
proper logging and rollback capabilities.
"""

import os
import subprocess
import shutil
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum

from .categories import FixSuggestion, FixRiskLevel, ErrorDiagnosis

logger = logging.getLogger(__name__)


class FixStatus(Enum):
    """Status of a fix execution."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


@dataclass
class FixResult:
    """Result of executing a fix."""
    fix: FixSuggestion
    status: FixStatus
    message: str = ""
    output: str = ""
    error: str = ""
    execution_time: float = 0.0
    rollback_info: Optional[Dict[str, Any]] = None
    
    @property
    def success(self) -> bool:
        return self.status == FixStatus.SUCCESS


@dataclass
class FixSession:
    """A session of fix executions."""
    session_id: str
    job_id: str
    diagnosis: ErrorDiagnosis
    results: List[FixResult] = field(default_factory=list)
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    
    @property
    def success_count(self) -> int:
        return sum(1 for r in self.results if r.success)
    
    @property
    def failure_count(self) -> int:
        return sum(1 for r in self.results if r.status == FixStatus.FAILED)


class AutoFixEngine:
    """
    Engine for executing suggested fixes.
    
    Handles:
    - Safe automatic fixes (mkdir, retry, etc.)
    - User-confirmed fixes (install, modify config)
    - Logging and rollback capabilities
    
    Example:
        engine = AutoFixEngine()
        result = await engine.execute(fix, context)
        
        if result.success:
            print(f"Fixed: {fix.description}")
    """
    
    # Commands that are safe to auto-execute
    SAFE_COMMANDS = {
        "mkdir",
        "ls",
        "cat",
        "head",
        "tail",
        "grep",
        "wc",
        "pwd",
        "echo",
        "md5sum",
        "file",
        "readlink",
        "which",
        "module",
        "conda",  # Only for activate
        "ping",
        "quota",
    }
    
    # Commands that need confirmation
    RISKY_COMMANDS = {
        "rm",
        "mv",
        "pip",
        "conda",  # For install
        "singularity",
        "sbatch",
        "chmod",
        "chown",
    }
    
    def __init__(
        self,
        dry_run: bool = False,
        log_dir: Optional[Path] = None,
        max_execution_time: int = 300,
    ):
        """
        Initialize the auto-fix engine.
        
        Args:
            dry_run: If True, don't actually execute commands
            log_dir: Directory for fix logs
            max_execution_time: Max seconds for command execution
        """
        self.dry_run = dry_run
        self.log_dir = log_dir or Path("logs/auto_fix")
        self.max_execution_time = max_execution_time
        self.sessions: Dict[str, FixSession] = {}
        
        # Ensure log directory exists
        self.log_dir.mkdir(parents=True, exist_ok=True)
    
    async def execute(
        self,
        fix: FixSuggestion,
        context: Optional[Dict[str, str]] = None,
        confirmed: bool = False,
    ) -> FixResult:
        """
        Execute a single fix.
        
        Args:
            fix: The fix to execute
            context: Variable substitutions for command template
            confirmed: Whether user has confirmed risky fixes
            
        Returns:
            FixResult with execution status
        """
        context = context or {}
        start_time = datetime.now()
        
        # Check if fix can be executed
        if not fix.command:
            return FixResult(
                fix=fix,
                status=FixStatus.SKIPPED,
                message="No command provided - manual action required",
            )
        
        # Substitute variables in command
        command = self._substitute_variables(fix.command, context)
        
        # Check risk level
        if fix.risk_level == FixRiskLevel.HIGH and not confirmed:
            return FixResult(
                fix=fix,
                status=FixStatus.SKIPPED,
                message="High-risk fix requires explicit confirmation",
            )
        
        if fix.risk_level == FixRiskLevel.MEDIUM and not confirmed and not fix.auto_executable:
            return FixResult(
                fix=fix,
                status=FixStatus.SKIPPED,
                message="Medium-risk fix requires confirmation",
            )
        
        # Validate command safety
        if not self._is_command_safe(command, fix.risk_level, confirmed):
            return FixResult(
                fix=fix,
                status=FixStatus.SKIPPED,
                message="Command not approved for auto-execution",
            )
        
        # Execute command
        if self.dry_run:
            return FixResult(
                fix=fix,
                status=FixStatus.SUCCESS,
                message=f"[DRY RUN] Would execute: {command}",
                output=command,
            )
        
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=self.max_execution_time,
            )
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            if result.returncode == 0:
                return FixResult(
                    fix=fix,
                    status=FixStatus.SUCCESS,
                    message="Fix applied successfully",
                    output=result.stdout[:2000],
                    execution_time=execution_time,
                )
            else:
                return FixResult(
                    fix=fix,
                    status=FixStatus.FAILED,
                    message=f"Command failed with exit code {result.returncode}",
                    output=result.stdout[:1000],
                    error=result.stderr[:1000],
                    execution_time=execution_time,
                )
                
        except subprocess.TimeoutExpired:
            return FixResult(
                fix=fix,
                status=FixStatus.FAILED,
                message=f"Command timed out after {self.max_execution_time} seconds",
            )
        except Exception as e:
            return FixResult(
                fix=fix,
                status=FixStatus.FAILED,
                message=f"Execution error: {str(e)}",
                error=str(e),
            )
    
    async def execute_all_safe(
        self,
        diagnosis: ErrorDiagnosis,
        context: Optional[Dict[str, str]] = None,
    ) -> List[FixResult]:
        """
        Execute all safe fixes from a diagnosis.
        
        Args:
            diagnosis: ErrorDiagnosis with fix suggestions
            context: Variable substitutions
            
        Returns:
            List of FixResult for each attempted fix
        """
        results = []
        
        for fix in diagnosis.suggested_fixes:
            if fix.risk_level == FixRiskLevel.SAFE or fix.auto_executable:
                result = await self.execute(fix, context, confirmed=False)
                results.append(result)
                
                # Stop on first success (usually that's the main fix)
                if result.success:
                    logger.info(f"Auto-fix successful: {fix.description}")
                    break
        
        return results
    
    async def execute_with_confirmation(
        self,
        diagnosis: ErrorDiagnosis,
        fix_indices: List[int],
        context: Optional[Dict[str, str]] = None,
    ) -> List[FixResult]:
        """
        Execute specific fixes that user has confirmed.
        
        Args:
            diagnosis: ErrorDiagnosis
            fix_indices: Indices of fixes to execute
            context: Variable substitutions
            
        Returns:
            List of FixResult
        """
        results = []
        
        for idx in fix_indices:
            if 0 <= idx < len(diagnosis.suggested_fixes):
                fix = diagnosis.suggested_fixes[idx]
                result = await self.execute(fix, context, confirmed=True)
                results.append(result)
        
        return results
    
    def _substitute_variables(
        self,
        command: str,
        context: Dict[str, str]
    ) -> str:
        """
        Substitute {variable} placeholders in command.
        
        Args:
            command: Command template
            context: Variable values
            
        Returns:
            Command with variables substituted
        """
        result = command
        
        for key, value in context.items():
            result = result.replace(f"{{{key}}}", str(value))
        
        # Also substitute environment variables
        result = os.path.expandvars(result)
        
        return result
    
    def _is_command_safe(
        self,
        command: str,
        risk_level: FixRiskLevel,
        confirmed: bool
    ) -> bool:
        """
        Check if a command is safe to execute.
        
        Args:
            command: The command to check
            risk_level: Declared risk level
            confirmed: Whether user confirmed
            
        Returns:
            True if command can be executed
        """
        # If it starts with #, it's a comment/instruction
        if command.strip().startswith('#'):
            return False
        
        # Get first word (the command)
        parts = command.strip().split()
        if not parts:
            return False
        
        base_command = parts[0].split('/')[-1]  # Handle full paths
        
        # Always allow safe commands
        if base_command in self.SAFE_COMMANDS:
            return True
        
        # Check for risky commands
        if base_command in self.RISKY_COMMANDS:
            # Require confirmation for risky commands
            return confirmed or risk_level == FixRiskLevel.SAFE
        
        # For unknown commands, use risk level
        if risk_level == FixRiskLevel.SAFE:
            return True
        
        return confirmed
    
    def get_executable_fixes(
        self,
        diagnosis: ErrorDiagnosis,
        include_risky: bool = False
    ) -> List[FixSuggestion]:
        """
        Get list of fixes that can be auto-executed.
        
        Args:
            diagnosis: ErrorDiagnosis
            include_risky: Include risky fixes (need confirmation)
            
        Returns:
            List of executable fixes
        """
        result = []
        
        for fix in diagnosis.suggested_fixes:
            if not fix.command:
                continue
            
            if fix.risk_level == FixRiskLevel.SAFE:
                result.append(fix)
            elif fix.risk_level == FixRiskLevel.LOW and fix.auto_executable:
                result.append(fix)
            elif include_risky:
                result.append(fix)
        
        return result
    
    def format_fix_for_display(
        self,
        fix: FixSuggestion,
        context: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Format a fix for user display.
        
        Args:
            fix: The fix to format
            context: Variable substitutions
            
        Returns:
            Formatted string
        """
        risk_icons = {
            FixRiskLevel.SAFE: "🟢",
            FixRiskLevel.LOW: "🟡",
            FixRiskLevel.MEDIUM: "🟠",
            FixRiskLevel.HIGH: "🔴",
        }
        
        icon = risk_icons.get(fix.risk_level, "⚪")
        auto = " *(auto)*" if fix.auto_executable else ""
        
        lines = [f"{icon} **{fix.description}**{auto}"]
        
        if fix.command:
            command = fix.command
            if context:
                command = self._substitute_variables(command, context)
            lines.append(f"```bash\n{command}\n```")
        
        return "\n".join(lines)


# Singleton instance
_auto_fix_engine: Optional[AutoFixEngine] = None


def get_auto_fix_engine(dry_run: bool = False) -> AutoFixEngine:
    """Get or create the auto-fix engine singleton."""
    global _auto_fix_engine
    
    if _auto_fix_engine is None or _auto_fix_engine.dry_run != dry_run:
        _auto_fix_engine = AutoFixEngine(dry_run=dry_run)
    
    return _auto_fix_engine
