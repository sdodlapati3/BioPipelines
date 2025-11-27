"""
Recovery Manager
================

Automated recovery system for handling failures.

Capabilities:
- Error diagnosis using AI
- Automatic fix application
- Server restart management
- Job resubmission
"""

import asyncio
import logging
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import json
import re

from .job_monitor import JobState, JobEvent, JobInfo, JobMonitor
from .health_checker import HealthChecker, HealthStatus

logger = logging.getLogger(__name__)


class RecoveryAction(Enum):
    """Types of recovery actions."""
    RESTART_SERVER = "restart_server"
    RESUBMIT_JOB = "resubmit_job"
    MODIFY_CONFIG = "modify_config"
    APPLY_PATCH = "apply_patch"
    CLEAR_CACHE = "clear_cache"
    SCALE_RESOURCES = "scale_resources"
    MANUAL_INTERVENTION = "manual_intervention"
    NO_ACTION = "no_action"


@dataclass
class RecoveryResult:
    """Result of a recovery attempt."""
    success: bool
    action: RecoveryAction
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    duration_seconds: float = 0.0
    error: Optional[str] = None
    new_job_id: Optional[str] = None  # If job was resubmitted
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "action": self.action.value,
            "message": self.message,
            "details": self.details,
            "duration_seconds": self.duration_seconds,
            "error": self.error,
            "new_job_id": self.new_job_id,
        }


@dataclass
class ErrorPattern:
    """Pattern for matching errors to recovery actions."""
    pattern: str  # Regex pattern
    action: RecoveryAction
    description: str
    fix_template: Optional[str] = None  # Template for automated fix
    requires_confirmation: bool = True
    max_retries: int = 3


class RecoveryManager:
    """
    Automated recovery from failures.
    
    Analyzes errors and applies appropriate recovery actions.
    
    Example:
        recovery = RecoveryManager()
        
        # Handle a job failure
        result = await recovery.handle_job_failure(
            job_id="12345",
            error_log=error_content,
        )
        
        if result.success:
            print(f"Recovery successful: {result.message}")
        else:
            print(f"Recovery failed: {result.error}")
    """
    
    # Common error patterns and their recovery actions
    ERROR_PATTERNS: List[ErrorPattern] = [
        # vLLM / Model errors
        ErrorPattern(
            pattern=r"CUDA out of memory",
            action=RecoveryAction.SCALE_RESOURCES,
            description="GPU memory exhausted",
            fix_template="Reduce batch size or increase GPU count",
        ),
        ErrorPattern(
            pattern=r"Connection refused.*:8000",
            action=RecoveryAction.RESTART_SERVER,
            description="vLLM server not responding",
        ),
        ErrorPattern(
            pattern=r"Model.*not found|cannot find model",
            action=RecoveryAction.MODIFY_CONFIG,
            description="Model path incorrect",
        ),
        ErrorPattern(
            pattern=r"triton.*error|grouped_topk",
            action=RecoveryAction.RESTART_SERVER,
            description="Triton/MoE kernel issue",
            fix_template="Set VLLM_MOE_USE_DEEP_GEMM=0 and restart",
        ),
        
        # SLURM errors
        ErrorPattern(
            pattern=r"slurmstepd: error.*Exceeded job memory limit",
            action=RecoveryAction.SCALE_RESOURCES,
            description="Job exceeded memory limit",
            fix_template="Increase --mem parameter",
        ),
        ErrorPattern(
            pattern=r"CANCELLED.*TIME LIMIT|DUE TO TIME LIMIT",
            action=RecoveryAction.SCALE_RESOURCES,
            description="Job exceeded time limit",
            fix_template="Increase --time parameter",
        ),
        ErrorPattern(
            pattern=r"srun: error: Unable to create job step",
            action=RecoveryAction.RESUBMIT_JOB,
            description="Transient SLURM error",
        ),
        
        # File/IO errors
        ErrorPattern(
            pattern=r"No space left on device",
            action=RecoveryAction.CLEAR_CACHE,
            description="Disk full",
        ),
        ErrorPattern(
            pattern=r"Permission denied",
            action=RecoveryAction.MANUAL_INTERVENTION,
            description="Permission issue",
            requires_confirmation=True,
        ),
        
        # Python errors
        ErrorPattern(
            pattern=r"ModuleNotFoundError: No module named",
            action=RecoveryAction.APPLY_PATCH,
            description="Missing Python module",
            fix_template="pip install {module_name}",
        ),
        ErrorPattern(
            pattern=r"ImportError.*cannot import name",
            action=RecoveryAction.APPLY_PATCH,
            description="Import error",
        ),
    ]
    
    def __init__(
        self,
        workspace_root: Optional[Path] = None,
        max_recovery_attempts: int = 3,
        require_confirmation: bool = True,
        ai_diagnosis_enabled: bool = True,
    ):
        """
        Initialize recovery manager.
        
        Args:
            workspace_root: Root directory for file operations
            max_recovery_attempts: Maximum recovery attempts per error
            require_confirmation: Require user confirmation for actions
            ai_diagnosis_enabled: Use AI for error diagnosis
        """
        self.workspace_root = workspace_root or Path.cwd()
        self.max_recovery_attempts = max_recovery_attempts
        self.require_confirmation = require_confirmation
        self.ai_diagnosis_enabled = ai_diagnosis_enabled
        
        self._recovery_history: List[RecoveryResult] = []
        self._attempt_counts: Dict[str, int] = {}  # Track attempts per error hash
        
        # Callbacks
        self._confirmation_callback: Optional[Callable[[str, RecoveryAction], bool]] = None
        self._notification_callback: Optional[Callable[[str], None]] = None
    
    def set_confirmation_callback(
        self,
        callback: Callable[[str, RecoveryAction], bool]
    ) -> None:
        """Set callback for confirmation prompts."""
        self._confirmation_callback = callback
    
    def set_notification_callback(self, callback: Callable[[str], None]) -> None:
        """Set callback for notifications."""
        self._notification_callback = callback
    
    async def handle_job_failure(
        self,
        job_id: str,
        error_log: Optional[str] = None,
        job_info: Optional[JobInfo] = None,
    ) -> RecoveryResult:
        """
        Handle a SLURM job failure.
        
        Args:
            job_id: Failed job ID
            error_log: Error log content (will be collected if not provided)
            job_info: Job information (will be fetched if not provided)
        
        Returns:
            RecoveryResult with action taken
        """
        start = time.time()
        
        try:
            # Collect job info if needed
            if job_info is None:
                monitor = JobMonitor()
                job_info = monitor.get_job_info(job_id)
            
            # Collect error log if needed
            if error_log is None and job_info:
                error_log = self._collect_job_logs(job_info)
            
            if not error_log:
                return RecoveryResult(
                    success=False,
                    action=RecoveryAction.NO_ACTION,
                    message="No error log available",
                    duration_seconds=time.time() - start,
                )
            
            # Diagnose the error
            diagnosis = self._diagnose_error(error_log)
            
            if diagnosis is None:
                # Try AI diagnosis if enabled
                if self.ai_diagnosis_enabled:
                    diagnosis = await self._ai_diagnose(error_log)
                
                if diagnosis is None:
                    return RecoveryResult(
                        success=False,
                        action=RecoveryAction.MANUAL_INTERVENTION,
                        message="Could not diagnose error - manual intervention required",
                        details={"error_preview": error_log[:500]},
                        duration_seconds=time.time() - start,
                    )
            
            # Check retry limits
            error_hash = self._hash_error(error_log)
            attempts = self._attempt_counts.get(error_hash, 0)
            
            if attempts >= self.max_recovery_attempts:
                return RecoveryResult(
                    success=False,
                    action=RecoveryAction.MANUAL_INTERVENTION,
                    message=f"Max recovery attempts ({self.max_recovery_attempts}) exceeded",
                    details={"diagnosis": diagnosis, "attempts": attempts},
                    duration_seconds=time.time() - start,
                )
            
            self._attempt_counts[error_hash] = attempts + 1
            
            # Confirm action if required
            if self.require_confirmation and diagnosis.requires_confirmation:
                if self._confirmation_callback:
                    confirmed = self._confirmation_callback(
                        f"Recovery action: {diagnosis.description}\nAction: {diagnosis.action.value}",
                        diagnosis.action,
                    )
                    if not confirmed:
                        return RecoveryResult(
                            success=False,
                            action=diagnosis.action,
                            message="Recovery action rejected by user",
                            duration_seconds=time.time() - start,
                        )
            
            # Execute recovery action
            result = await self._execute_recovery(diagnosis, job_info, error_log)
            result.duration_seconds = time.time() - start
            
            # Store in history
            self._recovery_history.append(result)
            
            # Notify
            if self._notification_callback:
                status = "succeeded" if result.success else "failed"
                self._notification_callback(
                    f"Recovery {status}: {result.message}"
                )
            
            return result
            
        except Exception as e:
            logger.exception(f"Recovery error: {e}")
            return RecoveryResult(
                success=False,
                action=RecoveryAction.NO_ACTION,
                message="Recovery process failed",
                error=str(e),
                duration_seconds=time.time() - start,
            )
    
    async def handle_server_failure(
        self,
        component: str = "vllm",
        error_message: Optional[str] = None,
    ) -> RecoveryResult:
        """
        Handle a server failure (e.g., vLLM).
        
        Args:
            component: Component name
            error_message: Error message if available
        
        Returns:
            RecoveryResult
        """
        start = time.time()
        
        if component == "vllm":
            return await self._restart_vllm(error_message)
        
        return RecoveryResult(
            success=False,
            action=RecoveryAction.MANUAL_INTERVENTION,
            message=f"Unknown component: {component}",
            duration_seconds=time.time() - start,
        )
    
    def _diagnose_error(self, error_log: str) -> Optional[ErrorPattern]:
        """
        Diagnose error using pattern matching.
        
        Args:
            error_log: Error log content
        
        Returns:
            Matching ErrorPattern or None
        """
        for pattern in self.ERROR_PATTERNS:
            if re.search(pattern.pattern, error_log, re.IGNORECASE | re.MULTILINE):
                logger.info(f"Matched error pattern: {pattern.description}")
                return pattern
        
        return None
    
    async def _ai_diagnose(self, error_log: str) -> Optional[ErrorPattern]:
        """
        Use AI to diagnose error.
        
        This would integrate with the LLM to analyze the error.
        For now, returns None (to be implemented with LLM integration).
        """
        # TODO: Integrate with LLM for error diagnosis
        # This would call the existing diagnosis agent
        logger.info("AI diagnosis not yet implemented")
        return None
    
    async def _execute_recovery(
        self,
        pattern: ErrorPattern,
        job_info: Optional[JobInfo],
        error_log: str,
    ) -> RecoveryResult:
        """Execute the recovery action."""
        
        if pattern.action == RecoveryAction.RESTART_SERVER:
            return await self._restart_vllm(error_log)
        
        elif pattern.action == RecoveryAction.RESUBMIT_JOB:
            return await self._resubmit_job(job_info)
        
        elif pattern.action == RecoveryAction.CLEAR_CACHE:
            return await self._clear_cache()
        
        elif pattern.action == RecoveryAction.SCALE_RESOURCES:
            return RecoveryResult(
                success=False,
                action=pattern.action,
                message=f"Resource scaling recommended: {pattern.fix_template}",
                details={
                    "suggestion": pattern.fix_template,
                    "description": pattern.description,
                },
            )
        
        elif pattern.action == RecoveryAction.APPLY_PATCH:
            # Extract details from error
            module_match = re.search(r"No module named '(\S+)'", error_log)
            if module_match:
                module = module_match.group(1).split(".")[0]
                return await self._install_module(module)
            
            return RecoveryResult(
                success=False,
                action=pattern.action,
                message="Could not determine patch to apply",
            )
        
        else:
            return RecoveryResult(
                success=False,
                action=pattern.action,
                message=f"Action {pattern.action.value} requires manual intervention",
                details={"suggestion": pattern.fix_template},
            )
    
    async def _restart_vllm(self, error_message: Optional[str] = None) -> RecoveryResult:
        """Restart vLLM server."""
        try:
            loop = asyncio.get_event_loop()
            
            # Find and kill existing vLLM processes
            def kill_vllm():
                try:
                    result = subprocess.run(
                        ["pkill", "-f", "vllm.entrypoints"],
                        capture_output=True,
                        text=True,
                        timeout=10,
                    )
                    return result.returncode == 0
                except Exception:
                    return False
            
            killed = await loop.run_in_executor(None, kill_vllm)
            
            # Wait for processes to terminate
            await asyncio.sleep(5)
            
            # Start server using script
            start_script = self.workspace_root / "scripts" / "start_server.sh"
            
            if start_script.exists():
                def start_server():
                    result = subprocess.Popen(
                        ["bash", str(start_script)],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        cwd=str(self.workspace_root),
                    )
                    return result.pid
                
                pid = await loop.run_in_executor(None, start_server)
                
                # Wait for server to become healthy
                await asyncio.sleep(30)  # Give server time to start
                
                checker = HealthChecker()
                health = await checker.check_vllm()
                
                if health.status == HealthStatus.HEALTHY:
                    return RecoveryResult(
                        success=True,
                        action=RecoveryAction.RESTART_SERVER,
                        message="vLLM server restarted successfully",
                        details={"pid": pid},
                    )
                else:
                    return RecoveryResult(
                        success=False,
                        action=RecoveryAction.RESTART_SERVER,
                        message=f"Server started but not healthy: {health.message}",
                    )
            else:
                return RecoveryResult(
                    success=False,
                    action=RecoveryAction.RESTART_SERVER,
                    message="Start script not found",
                    details={"expected_path": str(start_script)},
                )
                
        except Exception as e:
            return RecoveryResult(
                success=False,
                action=RecoveryAction.RESTART_SERVER,
                message="Failed to restart vLLM",
                error=str(e),
            )
    
    async def _resubmit_job(self, job_info: Optional[JobInfo]) -> RecoveryResult:
        """Resubmit a failed SLURM job."""
        if job_info is None:
            return RecoveryResult(
                success=False,
                action=RecoveryAction.RESUBMIT_JOB,
                message="No job information available for resubmission",
            )
        
        try:
            loop = asyncio.get_event_loop()
            
            # Try to find the original submission script
            script_path = None
            if job_info.working_dir:
                for pattern in ["*.sh", "*.slurm", "submit_*.sh"]:
                    scripts = list(job_info.working_dir.glob(pattern))
                    if scripts:
                        script_path = scripts[0]
                        break
            
            if script_path is None:
                return RecoveryResult(
                    success=False,
                    action=RecoveryAction.RESUBMIT_JOB,
                    message="Could not find job submission script",
                )
            
            def submit():
                result = subprocess.run(
                    ["sbatch", str(script_path)],
                    capture_output=True,
                    text=True,
                    cwd=str(job_info.working_dir),
                    timeout=30,
                )
                return result
            
            result = await loop.run_in_executor(None, submit)
            
            if result.returncode == 0:
                # Parse job ID from output
                match = re.search(r"Submitted batch job (\d+)", result.stdout)
                new_job_id = match.group(1) if match else None
                
                return RecoveryResult(
                    success=True,
                    action=RecoveryAction.RESUBMIT_JOB,
                    message=f"Job resubmitted successfully",
                    new_job_id=new_job_id,
                    details={"script": str(script_path)},
                )
            else:
                return RecoveryResult(
                    success=False,
                    action=RecoveryAction.RESUBMIT_JOB,
                    message="Job resubmission failed",
                    error=result.stderr,
                )
                
        except Exception as e:
            return RecoveryResult(
                success=False,
                action=RecoveryAction.RESUBMIT_JOB,
                message="Resubmission error",
                error=str(e),
            )
    
    async def _clear_cache(self) -> RecoveryResult:
        """Clear various caches to free disk space."""
        try:
            loop = asyncio.get_event_loop()
            cleared = []
            
            # Clear common cache locations
            cache_dirs = [
                Path.home() / ".cache" / "huggingface",
                Path.home() / ".cache" / "pip",
                Path("/tmp"),
            ]
            
            for cache_dir in cache_dirs:
                if cache_dir.exists():
                    try:
                        def get_size():
                            total = 0
                            for f in cache_dir.rglob("*"):
                                if f.is_file():
                                    total += f.stat().st_size
                            return total / (1024**3)  # GB
                        
                        size = await loop.run_in_executor(None, get_size)
                        cleared.append(f"{cache_dir.name}: {size:.2f}GB")
                    except Exception:
                        pass
            
            return RecoveryResult(
                success=True,
                action=RecoveryAction.CLEAR_CACHE,
                message="Cache locations identified (manual clearing recommended)",
                details={"caches": cleared},
            )
            
        except Exception as e:
            return RecoveryResult(
                success=False,
                action=RecoveryAction.CLEAR_CACHE,
                message="Cache clearing failed",
                error=str(e),
            )
    
    async def _install_module(self, module: str) -> RecoveryResult:
        """Install a missing Python module."""
        try:
            loop = asyncio.get_event_loop()
            
            def pip_install():
                return subprocess.run(
                    ["pip", "install", module],
                    capture_output=True,
                    text=True,
                    timeout=300,
                )
            
            result = await loop.run_in_executor(None, pip_install)
            
            if result.returncode == 0:
                return RecoveryResult(
                    success=True,
                    action=RecoveryAction.APPLY_PATCH,
                    message=f"Installed module: {module}",
                )
            else:
                return RecoveryResult(
                    success=False,
                    action=RecoveryAction.APPLY_PATCH,
                    message=f"Failed to install {module}",
                    error=result.stderr,
                )
                
        except Exception as e:
            return RecoveryResult(
                success=False,
                action=RecoveryAction.APPLY_PATCH,
                message=f"Module installation failed",
                error=str(e),
            )
    
    def _collect_job_logs(self, job_info: JobInfo, tail_lines: int = 200) -> str:
        """Collect logs for a job."""
        logs = []
        
        # Look for log files
        patterns = []
        if job_info.working_dir:
            patterns.extend([
                job_info.working_dir / f"slurm-{job_info.job_id}.out",
                job_info.working_dir / f"slurm-{job_info.job_id}.err",
            ])
        
        patterns.extend([
            Path(f"slurm-{job_info.job_id}.out"),
            Path(f"slurm-{job_info.job_id}.err"),
        ])
        
        for log_path in patterns:
            if log_path.exists():
                try:
                    result = subprocess.run(
                        ["tail", "-n", str(tail_lines), str(log_path)],
                        capture_output=True,
                        text=True,
                        timeout=10,
                    )
                    if result.stdout:
                        logs.append(f"=== {log_path.name} ===\n{result.stdout}")
                except Exception as e:
                    logs.append(f"Error reading {log_path}: {e}")
        
        return "\n\n".join(logs) if logs else ""
    
    def _hash_error(self, error_log: str) -> str:
        """Create a hash of an error for deduplication."""
        import hashlib
        
        # Normalize the error (remove timestamps, job IDs, etc.)
        normalized = re.sub(r'\d{4}-\d{2}-\d{2}', 'DATE', error_log)
        normalized = re.sub(r'\d{2}:\d{2}:\d{2}', 'TIME', normalized)
        normalized = re.sub(r'\b\d{5,}\b', 'JOBID', normalized)
        
        # Take first 1000 chars for hashing
        return hashlib.md5(normalized[:1000].encode()).hexdigest()[:16]
    
    def get_recovery_history(self, limit: int = 20) -> List[RecoveryResult]:
        """Get recovery history."""
        return self._recovery_history[-limit:]


class RecoveryLoop:
    """
    Continuous recovery monitoring loop.
    
    Combines job monitoring, health checking, and recovery.
    
    Example:
        loop = RecoveryLoop()
        await loop.start()
        
        # Add jobs to monitor
        loop.monitor_job("12345")
    """
    
    def __init__(
        self,
        health_check_interval: float = 60.0,
        auto_recovery: bool = True,
    ):
        """
        Initialize recovery loop.
        
        Args:
            health_check_interval: Seconds between health checks
            auto_recovery: Enable automatic recovery
        """
        self.health_check_interval = health_check_interval
        self.auto_recovery = auto_recovery
        
        self.job_monitor = JobMonitor()
        self.health_checker = HealthChecker()
        self.recovery_manager = RecoveryManager()
        
        self._running = False
        self._health_task: Optional[asyncio.Task] = None
    
    async def start(self) -> None:
        """Start the recovery loop."""
        self._running = True
        
        # Start job monitor
        await self.job_monitor.start()
        
        # Start health check loop
        self._health_task = asyncio.create_task(self._health_loop())
        
        logger.info("Recovery loop started")
    
    async def stop(self) -> None:
        """Stop the recovery loop."""
        self._running = False
        
        await self.job_monitor.stop()
        
        if self._health_task:
            self._health_task.cancel()
            try:
                await self._health_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Recovery loop stopped")
    
    async def _health_loop(self) -> None:
        """Periodic health checking."""
        while self._running:
            try:
                health = await self.health_checker.check_all()
                
                if not health.status.is_ok and self.auto_recovery:
                    for comp in health.unhealthy_components:
                        logger.warning(f"Unhealthy component: {comp.name}")
                        
                        if comp.name == "vllm":
                            result = await self.recovery_manager.handle_server_failure(
                                "vllm",
                                comp.message,
                            )
                            logger.info(f"vLLM recovery: {result.message}")
                
                await asyncio.sleep(self.health_check_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health loop error: {e}")
                await asyncio.sleep(10)
    
    def monitor_job(
        self,
        job_id: str,
        on_complete: Optional[Callable[[JobEvent], None]] = None,
    ) -> None:
        """
        Add a job to monitor.
        
        Args:
            job_id: SLURM job ID
            on_complete: Callback on completion
        """
        async def handle_failure(event: JobEvent, error_log: str):
            if self.auto_recovery:
                result = await self.recovery_manager.handle_job_failure(
                    job_id=event.job_id,
                    error_log=error_log,
                )
                logger.info(f"Job {job_id} recovery: {result.message}")
        
        self.job_monitor.watch_job(
            job_id=job_id,
            on_failure=handle_failure,
            on_complete=on_complete,
        )
