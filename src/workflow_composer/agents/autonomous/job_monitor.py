"""
Job Monitoring System
=====================

Continuous SLURM job monitoring with event-driven callbacks.

Features:
- Watch multiple jobs concurrently
- Event callbacks for state changes
- Automatic log collection on failure
- Integration with recovery system
"""

import asyncio
import logging
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set
import json
import re

logger = logging.getLogger(__name__)


class JobState(Enum):
    """SLURM job states."""
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETING = "COMPLETING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    TIMEOUT = "TIMEOUT"
    NODE_FAIL = "NODE_FAIL"
    PREEMPTED = "PREEMPTED"
    OUT_OF_MEMORY = "OUT_OF_MEMORY"
    UNKNOWN = "UNKNOWN"
    
    @classmethod
    def from_string(cls, state_str: str) -> "JobState":
        """Parse SLURM state string."""
        # Handle compound states like "CANCELLED by 12345"
        base_state = state_str.split()[0].upper()
        
        # Map SLURM abbreviations
        state_map = {
            "PD": cls.PENDING,
            "R": cls.RUNNING,
            "CG": cls.COMPLETING,
            "CD": cls.COMPLETED,
            "F": cls.FAILED,
            "CA": cls.CANCELLED,
            "TO": cls.TIMEOUT,
            "NF": cls.NODE_FAIL,
            "PR": cls.PREEMPTED,
            "OOM": cls.OUT_OF_MEMORY,
        }
        
        if base_state in state_map:
            return state_map[base_state]
        
        try:
            return cls(base_state)
        except ValueError:
            return cls.UNKNOWN
    
    @property
    def is_terminal(self) -> bool:
        """Check if this is a terminal state."""
        return self in {
            JobState.COMPLETED,
            JobState.FAILED,
            JobState.CANCELLED,
            JobState.TIMEOUT,
            JobState.NODE_FAIL,
            JobState.PREEMPTED,
            JobState.OUT_OF_MEMORY,
        }
    
    @property
    def is_failure(self) -> bool:
        """Check if this is a failure state."""
        return self in {
            JobState.FAILED,
            JobState.TIMEOUT,
            JobState.NODE_FAIL,
            JobState.OUT_OF_MEMORY,
        }


@dataclass
class JobEvent:
    """Event representing a job state change."""
    job_id: str
    timestamp: datetime
    previous_state: Optional[JobState]
    current_state: JobState
    exit_code: Optional[int] = None
    node: Optional[str] = None
    runtime: Optional[str] = None
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "job_id": self.job_id,
            "timestamp": self.timestamp.isoformat(),
            "previous_state": self.previous_state.value if self.previous_state else None,
            "current_state": self.current_state.value,
            "exit_code": self.exit_code,
            "node": self.node,
            "runtime": self.runtime,
            "error_message": self.error_message,
        }


@dataclass
class JobInfo:
    """Complete job information."""
    job_id: str
    name: str
    state: JobState
    partition: str
    user: str
    node: Optional[str] = None
    submit_time: Optional[datetime] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    exit_code: Optional[int] = None
    stdout_path: Optional[Path] = None
    stderr_path: Optional[Path] = None
    working_dir: Optional[Path] = None
    
    @classmethod
    def from_sacct(cls, data: Dict[str, str]) -> "JobInfo":
        """Parse from sacct output."""
        def parse_time(time_str: str) -> Optional[datetime]:
            if not time_str or time_str == "Unknown":
                return None
            try:
                return datetime.fromisoformat(time_str.replace("T", " "))
            except ValueError:
                return None
        
        def parse_exit_code(code_str: str) -> Optional[int]:
            if not code_str or ":" not in code_str:
                return None
            try:
                return int(code_str.split(":")[0])
            except ValueError:
                return None
        
        return cls(
            job_id=data.get("JobID", "").split(".")[0],  # Remove step suffix
            name=data.get("JobName", "unknown"),
            state=JobState.from_string(data.get("State", "UNKNOWN")),
            partition=data.get("Partition", "unknown"),
            user=data.get("User", "unknown"),
            node=data.get("NodeList") or None,
            submit_time=parse_time(data.get("Submit", "")),
            start_time=parse_time(data.get("Start", "")),
            end_time=parse_time(data.get("End", "")),
            exit_code=parse_exit_code(data.get("ExitCode", "")),
            working_dir=Path(data["WorkDir"]) if data.get("WorkDir") else None,
        )


@dataclass
class JobWatch:
    """Configuration for watching a job."""
    job_id: str
    poll_interval: float = 10.0  # seconds
    on_state_change: Optional[Callable[[JobEvent], None]] = None
    on_complete: Optional[Callable[[JobEvent], None]] = None
    on_failure: Optional[Callable[[JobEvent, str], None]] = None  # event, error_log
    collect_logs_on_failure: bool = True
    auto_recover: bool = False
    
    # Internal state
    last_state: Optional[JobState] = field(default=None, repr=False)
    started_at: datetime = field(default_factory=datetime.now, repr=False)


class JobMonitor:
    """
    Continuous job monitoring system.
    
    Watches SLURM jobs and triggers callbacks on state changes.
    
    Example:
        monitor = JobMonitor()
        
        # Watch a job
        watch = JobWatch(
            job_id="12345",
            on_failure=lambda event, log: print(f"Job failed: {log}"),
            on_complete=lambda event: print("Job complete!"),
        )
        monitor.add_watch(watch)
        
        # Start monitoring
        await monitor.start()
    """
    
    def __init__(self, default_poll_interval: float = 10.0):
        """
        Initialize job monitor.
        
        Args:
            default_poll_interval: Default time between job state checks
        """
        self.default_poll_interval = default_poll_interval
        self._watches: Dict[str, JobWatch] = {}
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()
        self._event_history: List[JobEvent] = []
    
    def add_watch(self, watch: JobWatch) -> None:
        """Add a job to monitor."""
        self._watches[watch.job_id] = watch
        logger.info(f"Added watch for job {watch.job_id}")
    
    def remove_watch(self, job_id: str) -> Optional[JobWatch]:
        """Remove a job from monitoring."""
        watch = self._watches.pop(job_id, None)
        if watch:
            logger.info(f"Removed watch for job {job_id}")
        return watch
    
    def watch_job(
        self,
        job_id: str,
        on_failure: Optional[Callable[[JobEvent, str], None]] = None,
        on_complete: Optional[Callable[[JobEvent], None]] = None,
        on_state_change: Optional[Callable[[JobEvent], None]] = None,
        poll_interval: Optional[float] = None,
        auto_recover: bool = False,
    ) -> JobWatch:
        """
        Convenience method to watch a job.
        
        Args:
            job_id: SLURM job ID to monitor
            on_failure: Callback on job failure (receives event and error log)
            on_complete: Callback on successful completion
            on_state_change: Callback on any state change
            poll_interval: Polling interval in seconds
            auto_recover: Attempt automatic recovery on failure
        
        Returns:
            The created JobWatch instance
        """
        watch = JobWatch(
            job_id=job_id,
            poll_interval=poll_interval or self.default_poll_interval,
            on_state_change=on_state_change,
            on_complete=on_complete,
            on_failure=on_failure,
            auto_recover=auto_recover,
        )
        self.add_watch(watch)
        return watch
    
    async def start(self) -> None:
        """Start the monitoring loop."""
        if self._running:
            return
        
        self._running = True
        self._task = asyncio.create_task(self._monitor_loop())
        logger.info("Job monitor started")
    
    async def stop(self) -> None:
        """Stop the monitoring loop."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Job monitor stopped")
    
    async def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while self._running:
            try:
                async with self._lock:
                    watches = list(self._watches.values())
                
                for watch in watches:
                    try:
                        await self._check_job(watch)
                    except Exception as e:
                        logger.error(f"Error checking job {watch.job_id}: {e}")
                
                # Calculate next poll time (use minimum interval)
                if watches:
                    interval = min(w.poll_interval for w in watches)
                else:
                    interval = self.default_poll_interval
                
                await asyncio.sleep(interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitor loop error: {e}")
                await asyncio.sleep(5)  # Back off on error
    
    async def _check_job(self, watch: JobWatch) -> None:
        """Check a single job's status."""
        info = self.get_job_info(watch.job_id)
        
        if info is None:
            logger.warning(f"Could not get info for job {watch.job_id}")
            return
        
        current_state = info.state
        
        # Detect state change
        if watch.last_state != current_state:
            event = JobEvent(
                job_id=watch.job_id,
                timestamp=datetime.now(),
                previous_state=watch.last_state,
                current_state=current_state,
                exit_code=info.exit_code,
                node=info.node,
            )
            
            logger.info(f"Job {watch.job_id}: {watch.last_state} -> {current_state}")
            self._event_history.append(event)
            
            # Trigger state change callback
            if watch.on_state_change:
                try:
                    watch.on_state_change(event)
                except Exception as e:
                    logger.error(f"State change callback error: {e}")
            
            # Handle terminal states
            if current_state.is_terminal:
                if current_state.is_failure:
                    # Collect error logs
                    error_log = ""
                    if watch.collect_logs_on_failure:
                        error_log = self._collect_error_log(info)
                        event.error_message = error_log[:1000]  # Truncate for event
                    
                    # Trigger failure callback
                    if watch.on_failure:
                        try:
                            watch.on_failure(event, error_log)
                        except Exception as e:
                            logger.error(f"Failure callback error: {e}")
                
                elif current_state == JobState.COMPLETED:
                    # Trigger completion callback
                    if watch.on_complete:
                        try:
                            watch.on_complete(event)
                        except Exception as e:
                            logger.error(f"Complete callback error: {e}")
                
                # Remove from watches after terminal state
                async with self._lock:
                    self._watches.pop(watch.job_id, None)
            
            # Update last state
            watch.last_state = current_state
    
    def get_job_info(self, job_id: str) -> Optional[JobInfo]:
        """
        Get detailed job information from SLURM.
        
        Args:
            job_id: SLURM job ID
        
        Returns:
            JobInfo or None if not found
        """
        try:
            # Use sacct for detailed info
            result = subprocess.run(
                [
                    "sacct",
                    "-j", job_id,
                    "--format=JobID,JobName,State,Partition,User,NodeList,Submit,Start,End,ExitCode,WorkDir",
                    "--parsable2",
                    "--noheader",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            
            if result.returncode != 0:
                logger.warning(f"sacct failed for job {job_id}: {result.stderr}")
                return None
            
            # Parse output
            lines = result.stdout.strip().split("\n")
            if not lines or not lines[0]:
                return None
            
            # Get the main job line (not step lines)
            fields = [
                "JobID", "JobName", "State", "Partition", "User",
                "NodeList", "Submit", "Start", "End", "ExitCode", "WorkDir"
            ]
            
            for line in lines:
                parts = line.split("|")
                if len(parts) >= len(fields):
                    data = dict(zip(fields, parts))
                    # Skip step lines (contain ".")
                    if "." not in data["JobID"]:
                        return JobInfo.from_sacct(data)
            
            return None
            
        except subprocess.TimeoutExpired:
            logger.error(f"Timeout getting info for job {job_id}")
            return None
        except FileNotFoundError:
            logger.error("sacct command not found - SLURM not available?")
            return None
        except Exception as e:
            logger.error(f"Error getting job info: {e}")
            return None
    
    def _collect_error_log(self, info: JobInfo, tail_lines: int = 100) -> str:
        """
        Collect error logs for a failed job.
        
        Args:
            info: Job information
            tail_lines: Number of lines to collect from each log
        
        Returns:
            Combined error log content
        """
        logs = []
        
        # Try to find log files
        log_patterns = [
            info.stderr_path,
            info.stdout_path,
            # Common patterns
            Path(f"slurm-{info.job_id}.out"),
            Path(f"slurm-{info.job_id}.err"),
        ]
        
        if info.working_dir:
            log_patterns.extend([
                info.working_dir / f"slurm-{info.job_id}.out",
                info.working_dir / f"slurm-{info.job_id}.err",
            ])
        
        for log_path in log_patterns:
            if log_path and log_path.exists():
                try:
                    result = subprocess.run(
                        ["tail", "-n", str(tail_lines), str(log_path)],
                        capture_output=True,
                        text=True,
                        timeout=5,
                    )
                    if result.stdout:
                        logs.append(f"=== {log_path.name} ===\n{result.stdout}")
                except Exception as e:
                    logs.append(f"=== {log_path.name} ===\nError reading: {e}")
        
        if not logs:
            logs.append("No log files found")
        
        return "\n\n".join(logs)
    
    def get_active_watches(self) -> List[JobWatch]:
        """Get list of active job watches."""
        return list(self._watches.values())
    
    def get_event_history(
        self,
        job_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[JobEvent]:
        """
        Get event history.
        
        Args:
            job_id: Filter by job ID (optional)
            limit: Maximum number of events to return
        
        Returns:
            List of events, most recent first
        """
        events = self._event_history
        
        if job_id:
            events = [e for e in events if e.job_id == job_id]
        
        return events[-limit:][::-1]
    
    @staticmethod
    def list_running_jobs(user: Optional[str] = None) -> List[Dict[str, str]]:
        """
        List currently running SLURM jobs.
        
        Args:
            user: Filter by user (default: current user)
        
        Returns:
            List of job info dictionaries
        """
        try:
            cmd = ["squeue", "--format=%i|%j|%t|%M|%P|%N", "--noheader"]
            if user:
                cmd.extend(["-u", user])
            else:
                cmd.append("--me")
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
            jobs = []
            for line in result.stdout.strip().split("\n"):
                if line:
                    parts = line.split("|")
                    if len(parts) >= 6:
                        jobs.append({
                            "job_id": parts[0],
                            "name": parts[1],
                            "state": parts[2],
                            "time": parts[3],
                            "partition": parts[4],
                            "nodelist": parts[5],
                        })
            
            return jobs
            
        except Exception as e:
            logger.error(f"Error listing jobs: {e}")
            return []


# Convenience functions for synchronous usage
def get_job_status(job_id: str) -> Optional[JobState]:
    """Get current job state (synchronous)."""
    monitor = JobMonitor()
    info = monitor.get_job_info(job_id)
    return info.state if info else None


def wait_for_job(
    job_id: str,
    poll_interval: float = 10.0,
    timeout: Optional[float] = None,
) -> JobEvent:
    """
    Wait for a job to complete (synchronous, blocking).
    
    Args:
        job_id: Job to wait for
        poll_interval: Seconds between checks
        timeout: Maximum wait time in seconds
    
    Returns:
        Final job event
    
    Raises:
        TimeoutError: If timeout exceeded
    """
    monitor = JobMonitor()
    start = time.time()
    last_state = None
    
    while True:
        info = monitor.get_job_info(job_id)
        
        if info is None:
            time.sleep(poll_interval)
            continue
        
        current_state = info.state
        
        if current_state != last_state:
            print(f"Job {job_id}: {current_state.value}")
            last_state = current_state
        
        if current_state.is_terminal:
            return JobEvent(
                job_id=job_id,
                timestamp=datetime.now(),
                previous_state=last_state,
                current_state=current_state,
                exit_code=info.exit_code,
                node=info.node,
            )
        
        if timeout and (time.time() - start) > timeout:
            raise TimeoutError(f"Timeout waiting for job {job_id}")
        
        time.sleep(poll_interval)
