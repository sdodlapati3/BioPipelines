"""
Execution Tools
===============

Tools for submitting and managing SLURM jobs.
"""

import logging
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import ToolResult

logger = logging.getLogger(__name__)


# =============================================================================
# SUBMIT_JOB
# =============================================================================

SUBMIT_JOB_PATTERNS = [
    r"(?:run|execute|submit|start)\s+(?:it|the workflow|this|pipeline)\s*(?:on|with|using)?\s*(slurm|local|docker)?",
    r"(?:run|execute|submit)\s+(?:workflow|pipeline)?\s*['\"]?([^'\"]+)['\"]?\s*(?:on|with)?\s*(slurm|local|docker)?",
]


def submit_job_impl(
    workflow_path: str = None,
    profile: str = "slurm",
) -> ToolResult:
    """
    Submit a workflow job to SLURM.
    
    Args:
        workflow_path: Path to workflow main.nf
        profile: Execution profile (slurm, local, docker)
        
    Returns:
        ToolResult with job submission status
    """
    # Find most recent workflow if not specified
    if not workflow_path:
        generated_dir = Path.cwd() / "generated_workflows"
        if generated_dir.exists():
            workflows = sorted(
                [d for d in generated_dir.iterdir() if d.is_dir() and (d / "main.nf").exists()],
                key=lambda x: x.stat().st_mtime,
                reverse=True
            )
            if workflows:
                workflow_path = str(workflows[0] / "main.nf")
    
    if not workflow_path:
        return ToolResult(
            success=False,
            tool_name="submit_job",
            error="No workflow found",
            message="❌ No workflow to run. Generate one first with 'create <type> workflow'"
        )
    
    workflow_path = Path(workflow_path)
    if not workflow_path.exists():
        return ToolResult(
            success=False,
            tool_name="submit_job",
            error=f"Workflow not found: {workflow_path}",
            message=f"❌ Workflow file not found: `{workflow_path}`"
        )
    
    # Build nextflow command
    cmd = f"nextflow run {workflow_path} -profile {profile}"
    
    message = f"""🚀 **Ready to Submit Workflow**

**Command:**
```bash
{cmd}
```

**Options:**
- `-resume` - Resume from previous run
- `-with-report` - Generate HTML report
- `-with-timeline` - Generate timeline

**To submit to SLURM:**
```bash
sbatch --wrap="{cmd}" --job-name=biopipeline --partition=main
```

Would you like me to submit this job?
"""
    
    return ToolResult(
        success=True,
        tool_name="submit_job",
        data={"workflow": str(workflow_path), "profile": profile, "command": cmd},
        message=message
    )


# =============================================================================
# GET_JOB_STATUS
# =============================================================================

GET_JOB_STATUS_PATTERNS = [
    r"(?:what(?:'s| is)|show|check)\s+(?:the\s+)?(?:status|progress)\s*(?:of)?\s*(?:job\s*)?(\d+)?",
    r"(?:how(?:'s| is))\s+(?:the\s+)?(?:job|workflow|pipeline)\s*(?:doing|going|running)?",
]


def get_job_status_impl(job_id: str = None) -> ToolResult:
    """
    Get status of SLURM job(s).
    
    Args:
        job_id: Specific job ID to check
        
    Returns:
        ToolResult with job status
    """
    try:
        if job_id:
            cmd = ["squeue", "-j", str(job_id), "-o", "%.18i %.9P %.30j %.8u %.8T %.10M %.9l %.6D %R"]
        else:
            cmd = ["squeue", "--me", "-o", "%.18i %.9P %.30j %.8u %.8T %.10M %.9l %.6D %R"]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        
        if result.returncode != 0:
            return ToolResult(
                success=False,
                tool_name="get_job_status",
                error=result.stderr,
                message=f"❌ Failed to get job status: {result.stderr}"
            )
        
        lines = result.stdout.strip().split('\n')
        
        if len(lines) <= 1:
            message = "✅ No running jobs found."
            if job_id:
                message = f"✅ Job {job_id} has completed or doesn't exist."
        else:
            job_table = "\n".join(lines)
            message = f"""📊 **Job Status**

```
{job_table}
```

**Legend:**
- **PD**: Pending
- **R**: Running
- **CG**: Completing
"""
        
        return ToolResult(
            success=True,
            tool_name="get_job_status",
            data={"output": result.stdout},
            message=message
        )
        
    except FileNotFoundError:
        return ToolResult(
            success=False,
            tool_name="get_job_status",
            error="SLURM not available",
            message="❌ SLURM is not available on this system."
        )
    except subprocess.TimeoutExpired:
        return ToolResult(
            success=False,
            tool_name="get_job_status",
            error="Command timed out",
            message="❌ Job status check timed out."
        )
    except Exception as e:
        return ToolResult(
            success=False,
            tool_name="get_job_status",
            error=str(e),
            message=f"❌ Error checking job status: {e}"
        )


# =============================================================================
# GET_LOGS
# =============================================================================

GET_LOGS_PATTERNS = [
    r"(?:show|get|view|display)\s+(?:me\s+)?(?:the\s+)?logs?\s*(?:for|of)?\s*(?:job\s*)?(\d+)?",
    r"(?:what(?:'s| is))\s+(?:in\s+)?(?:the\s+)?logs?",
]


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
# CANCEL_JOB
# =============================================================================

CANCEL_JOB_PATTERNS = [
    r"(?:cancel|stop|abort|kill)\s+(?:the\s+)?(?:job|workflow|pipeline)\s*(\d+)?",
]


def cancel_job_impl(job_id: str = None) -> ToolResult:
    """
    Cancel a SLURM job.
    
    Args:
        job_id: Job ID to cancel
        
    Returns:
        ToolResult with cancellation status
    """
    if not job_id:
        return ToolResult(
            success=False,
            tool_name="cancel_job",
            error="No job ID specified",
            message="❌ Please specify a job ID to cancel. Use 'check job status' to see running jobs."
        )
    
    try:
        result = subprocess.run(
            ["scancel", str(job_id)],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode != 0:
            return ToolResult(
                success=False,
                tool_name="cancel_job",
                error=result.stderr,
                message=f"❌ Failed to cancel job {job_id}: {result.stderr}"
            )
        
        return ToolResult(
            success=True,
            tool_name="cancel_job",
            data={"job_id": job_id},
            message=f"✅ Job {job_id} has been cancelled."
        )
        
    except FileNotFoundError:
        return ToolResult(
            success=False,
            tool_name="cancel_job",
            error="SLURM not available",
            message="❌ SLURM is not available on this system."
        )
    except Exception as e:
        return ToolResult(
            success=False,
            tool_name="cancel_job",
            error=str(e),
            message=f"❌ Error cancelling job: {e}"
        )
