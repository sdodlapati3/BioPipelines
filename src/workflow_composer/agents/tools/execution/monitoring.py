"""
Job Monitoring Tools
====================

Tools for viewing job logs and detailed job monitoring.

Functions:
    - get_logs_impl: Get logs from SLURM job
    - watch_job_impl: Get detailed job information with monitoring
"""

import logging
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from ..base import ToolResult

logger = logging.getLogger(__name__)


# =============================================================================
# PATTERNS
# =============================================================================

GET_LOGS_PATTERNS = [
    r"(?:show|get|view|display)\s+(?:me\s+)?(?:the\s+)?logs?\s*(?:for|of)?\s*(?:job\s*)?(\d+)?",
    r"(?:what(?:'s| is))\s+(?:in\s+)?(?:the\s+)?logs?",
]

WATCH_JOB_PATTERNS = [
    r"watch\s+(?:the\s+)?job\s*(?:id\s*)?(\d+)?",
    r"monitor\s+(?:the\s+)?job\s*(?:id\s*)?(\d+)?",
    r"track\s+(?:the\s+)?job\s*(?:id\s*)?(\d+)?",
    r"(?:detailed|full)\s+(?:job\s+)?status\s+(?:for\s+)?(\d+)?",
]


# =============================================================================
# GET_LOGS
# =============================================================================

def get_logs_impl(job_id: str = None, lines: int = 30) -> ToolResult:
    """
    Get logs from SLURM job.
    
    Args:
        job_id: Job ID to get logs for
        lines: Number of lines to show
        
    Returns:
        ToolResult with log content
    """
    log_dir = Path.cwd() / "logs"
    
    if job_id:
        # Look for specific job log
        log_files = list(log_dir.glob(f"*{job_id}*.log")) + list(log_dir.glob(f"*{job_id}*.out"))
    else:
        # Get most recent log
        log_files = sorted(
            list(log_dir.glob("*.log")) + list(log_dir.glob("*.out")),
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )
    
    if not log_files:
        # Try nextflow work directory
        work_log = Path.cwd() / ".nextflow.log"
        if work_log.exists():
            log_files = [work_log]
    
    if not log_files:
        return ToolResult(
            success=False,
            tool_name="get_logs",
            error="No log files found",
            message="❌ No log files found. Has a job been run?"
        )
    
    log_file = log_files[0]
    
    try:
        with open(log_file, 'r') as f:
            content = f.readlines()
        
        # Get last N lines
        last_lines = content[-lines:]
        log_content = "".join(last_lines)
        
        message = f"""📜 **Log: {log_file.name}**

```
{log_content}
```

Showing last {len(last_lines)} lines.
"""
        
        return ToolResult(
            success=True,
            tool_name="get_logs",
            data={"file": str(log_file), "lines": len(last_lines)},
            message=message
        )
        
    except Exception as e:
        return ToolResult(
            success=False,
            tool_name="get_logs",
            error=str(e),
            message=f"❌ Error reading log: {e}"
        )


# =============================================================================
# WATCH_JOB
# =============================================================================

def watch_job_impl(
    job_id: str = None,
    include_logs: bool = False,
    tail_lines: int = 50,
) -> ToolResult:
    """
    Get detailed job information using JobMonitor.
    
    Provides richer information than get_job_status:
    - Full job metadata (name, partition, user, nodes)
    - Submit/start/end times
    - Exit codes
    - Optional log tailing
    
    Args:
        job_id: SLURM job ID to monitor
        include_logs: Include recent log output
        tail_lines: Number of log lines to include
        
    Returns:
        ToolResult with detailed job information
    """
    if not job_id:
        # Try to find recent jobs
        try:
            result = subprocess.run(
                ["squeue", "--me", "--format=%i|%j|%t|%M|%P", "--noheader"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0 and result.stdout.strip():
                jobs = []
                for line in result.stdout.strip().split("\n")[:10]:
                    parts = line.split("|")
                    if len(parts) >= 5:
                        jobs.append({
                            "id": parts[0],
                            "name": parts[1],
                            "state": parts[2],
                            "time": parts[3],
                            "partition": parts[4],
                        })
                
                if jobs:
                    job_list = "\n".join([
                        f"  • **{j['id']}**: {j['name']} ({j['state']}) - {j['time']}"
                        for j in jobs
                    ])
                    return ToolResult(
                        success=True,
                        tool_name="watch_job",
                        data={"jobs": jobs},
                        message=f"""📋 **Your Running Jobs**

{job_list}

Use `watch job <id>` to get detailed info on a specific job.
"""
                    )
        except Exception:
            pass
        
        return ToolResult(
            success=False,
            tool_name="watch_job",
            error="No job ID provided",
            message="""❌ **No Job ID Provided**

Usage: `watch job 12345`

To see your running jobs: `watch job` or `squeue --me`
"""
        )
    
    try:
        from workflow_composer.agents.autonomous.job_monitor import JobMonitor, JobState
        
        monitor = JobMonitor()
        job_info = monitor.get_job_info(job_id)
        
        if job_info is None:
            return ToolResult(
                success=False,
                tool_name="watch_job",
                error=f"Job {job_id} not found",
                message=f"""❌ **Job {job_id} Not Found**

The job may have:
- Completed and aged out of SLURM history
- Never existed
- Been submitted by a different user

Try `sacct -j {job_id}` for historical jobs.
"""
            )
        
        # Format times
        def format_time(dt):
            if dt:
                return dt.strftime("%Y-%m-%d %H:%M:%S")
            return "N/A"
        
        # State emoji
        state_emoji = {
            JobState.PENDING: "⏳",
            JobState.RUNNING: "🏃",
            JobState.COMPLETING: "🔄",
            JobState.COMPLETED: "✅",
            JobState.FAILED: "❌",
            JobState.CANCELLED: "🚫",
            JobState.TIMEOUT: "⏰",
            JobState.NODE_FAIL: "💥",
            JobState.OUT_OF_MEMORY: "💾",
        }.get(job_info.state, "❓")
        
        # Calculate runtime
        runtime = "N/A"
        if job_info.start_time:
            end = job_info.end_time or datetime.now()
            delta = end - job_info.start_time
            hours, remainder = divmod(int(delta.total_seconds()), 3600)
            minutes, seconds = divmod(remainder, 60)
            runtime = f"{hours}h {minutes}m {seconds}s"
        
        # Build message
        message = f"""📊 **Job Details: {job_id}**

| Field | Value |
|-------|-------|
| **Name** | {job_info.name} |
| **State** | {state_emoji} {job_info.state.value} |
| **Partition** | {job_info.partition} |
| **User** | {job_info.user} |
| **Node(s)** | {job_info.node or 'N/A'} |
| **Submit Time** | {format_time(job_info.submit_time)} |
| **Start Time** | {format_time(job_info.start_time)} |
| **End Time** | {format_time(job_info.end_time)} |
| **Runtime** | {runtime} |
| **Exit Code** | {job_info.exit_code if job_info.exit_code is not None else 'N/A'} |
"""
        
        # Add status-specific messages
        if job_info.state == JobState.PENDING:
            message += "\n⏳ Job is waiting in queue. Check partition availability with `sinfo`."
        elif job_info.state == JobState.RUNNING:
            message += "\n🏃 Job is running. Use `get logs` to see output."
        elif job_info.state.is_failure:
            message += f"\n❌ Job failed. Use `diagnose error job_id={job_id}` to analyze."
            if include_logs:
                # Collect logs
                log_content = monitor._collect_error_log(job_info, tail_lines)
                if log_content:
                    message += f"\n\n**Recent Logs:**\n```\n{log_content[:1500]}{'...' if len(log_content) > 1500 else ''}\n```"
        elif job_info.state == JobState.COMPLETED:
            message += "\n✅ Job completed successfully. Use `analyze results` to review output."
        
        return ToolResult(
            success=True,
            tool_name="watch_job",
            data={
                "job_id": job_id,
                "name": job_info.name,
                "state": job_info.state.value,
                "partition": job_info.partition,
                "user": job_info.user,
                "node": job_info.node,
                "submit_time": format_time(job_info.submit_time),
                "start_time": format_time(job_info.start_time),
                "end_time": format_time(job_info.end_time),
                "runtime": runtime,
                "exit_code": job_info.exit_code,
                "is_terminal": job_info.state.is_terminal,
                "is_failure": job_info.state.is_failure,
            },
            message=message
        )
        
    except ImportError as e:
        # Fallback to sacct
        logger.warning(f"JobMonitor not available: {e}")
        
        try:
            result = subprocess.run(
                ["sacct", "-j", job_id, "--format=JobID,JobName,State,Elapsed,ExitCode,NodeList", 
                 "--parsable2", "--noheader"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            
            if result.returncode == 0 and result.stdout.strip():
                lines = result.stdout.strip().split("\n")
                # Get main job line (not steps)
                for line in lines:
                    parts = line.split("|")
                    if len(parts) >= 6 and "." not in parts[0]:
                        return ToolResult(
                            success=True,
                            tool_name="watch_job",
                            data={"raw": parts},
                            message=f"""📊 **Job {job_id}** (Basic Info)

| Field | Value |
|-------|-------|
| **Name** | {parts[1]} |
| **State** | {parts[2]} |
| **Elapsed** | {parts[3]} |
| **Exit Code** | {parts[4]} |
| **Node(s)** | {parts[5]} |
"""
                        )
            
            return ToolResult(
                success=False,
                tool_name="watch_job",
                error=f"Job {job_id} not found",
                message=f"❌ Job {job_id} not found in SLURM history"
            )
            
        except Exception as e2:
            return ToolResult(
                success=False,
                tool_name="watch_job",
                error=str(e2),
                message=f"❌ Error querying job: {e2}"
            )
    
    except Exception as e:
        logger.error(f"watch_job failed: {e}")
        return ToolResult(
            success=False,
            tool_name="watch_job",
            error=str(e),
            message=f"❌ Watch job failed: {e}"
        )


__all__ = [
    # Patterns
    "GET_LOGS_PATTERNS",
    "WATCH_JOB_PATTERNS",
    # Functions
    "get_logs_impl",
    "watch_job_impl",
]
