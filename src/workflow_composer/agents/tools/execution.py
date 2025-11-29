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
    r"(?:what(?:'s| is)|show|check)\s+(?:the\s+)?(?:job\s+)?(?:status|progress)",
    r"(?:how(?:'s| is))\s+(?:the\s+)?(?:job|workflow|pipeline)\s*(?:doing|going|running)?",
    r"(?:check|show)\s+(?:nextflow|workflow)\s+(?:status|progress)",
    r"(?:job|slurm)\s+status",
]


def get_job_status_impl(
    job_id: str = None,
    workflow_dir: str = None,
) -> ToolResult:
    """
    Get status of SLURM job(s) and/or Nextflow workflow.
    
    Uses WorkflowMonitor for rich Nextflow status when available,
    falls back to SLURM squeue for HPC job monitoring.
    
    Args:
        job_id: Specific SLURM job ID to check
        workflow_dir: Path to workflow directory (for Nextflow status)
        
    Returns:
        ToolResult with job status
    """
    results = {}
    messages = []
    
    # Try to use WorkflowMonitor for Nextflow status
    if workflow_dir or not job_id:
        try:
            from workflow_composer.monitor.workflow_monitor import WorkflowMonitor
            use_monitor = True
        except ImportError:
            use_monitor = False
            logger.debug("WorkflowMonitor not available")
        
        if use_monitor:
            # Find workflow directory
            if workflow_dir:
                wf_dir = Path(workflow_dir)
            else:
                # Try to find most recent workflow
                possible_dirs = [
                    Path.cwd() / "work",
                    Path.cwd() / "generated_workflows",
                    Path.home() / "BioPipelines" / "generated_workflows",
                ]
                
                wf_dir = None
                for parent in possible_dirs:
                    if parent.exists():
                        # Look for .nextflow.log
                        if (parent / ".nextflow.log").exists():
                            wf_dir = parent
                            break
                        # Or search subdirectories
                        for d in parent.iterdir():
                            if d.is_dir() and (d / ".nextflow.log").exists():
                                wf_dir = d
                                break
                        if wf_dir:
                            break
            
            if wf_dir and wf_dir.exists():
                try:
                    monitor = WorkflowMonitor()
                    execution = monitor.scan_workflow(str(wf_dir))
                    
                    if execution:
                        # Build progress bar
                        progress = execution.progress
                        bar_len = 20
                        filled = int(bar_len * progress / 100)
                        bar = "█" * filled + "░" * (bar_len - filled)
                        
                        # Count processes by status
                        counts = execution.process_counts
                        
                        status_emoji = {
                            "pending": "⏳",
                            "running": "🔄",
                            "completed": "✅", 
                            "failed": "❌",
                            "cached": "📦",
                        }
                        
                        status_lines = []
                        for status, count in counts.items():
                            if count > 0:
                                emoji = status_emoji.get(status, "•")
                                status_lines.append(f"  {emoji} {status.title()}: {count}")
                        
                        duration = ""
                        if execution.duration:
                            mins = int(execution.duration // 60)
                            secs = int(execution.duration % 60)
                            duration = f"\n**Duration:** {mins}m {secs}s"
                        
                        workflow_msg = f"""📊 **Nextflow Workflow Status**

**Workflow:** {execution.name}
**Status:** {execution.status.value.upper()}
**Progress:** [{bar}] {progress:.1f}%{duration}

**Process Summary:**
{chr(10).join(status_lines)}
"""
                        
                        if execution.error_message:
                            workflow_msg += f"\n**Error:**\n```\n{execution.error_message[:300]}...\n```"
                        
                        messages.append(workflow_msg)
                        results["nextflow"] = {
                            "workflow": execution.name,
                            "status": execution.status.value,
                            "progress": progress,
                            "processes": counts,
                        }
                        
                except Exception as e:
                    logger.debug(f"WorkflowMonitor scan failed: {e}")
    
    # Also check SLURM if available
    try:
        if job_id:
            cmd = ["squeue", "-j", str(job_id), "-o", "%.18i %.9P %.30j %.8u %.8T %.10M %.9l %.6D %R"]
        else:
            cmd = ["squeue", "--me", "-o", "%.18i %.9P %.30j %.8u %.8T %.10M %.9l %.6D %R"]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            
            if len(lines) <= 1:
                if job_id:
                    messages.append(f"✅ SLURM job {job_id} has completed or doesn't exist.")
                else:
                    messages.append("✅ No running SLURM jobs.")
            else:
                job_table = "\n".join(lines)
                messages.append(f"""📋 **SLURM Jobs**

```
{job_table}
```

**Legend:** PD=Pending, R=Running, CG=Completing
""")
            
            results["slurm"] = {"output": result.stdout}
            
    except FileNotFoundError:
        # SLURM not available - only show if no other results
        if not results:
            messages.append("ℹ️ SLURM is not available on this system.")
    except subprocess.TimeoutExpired:
        messages.append("⚠️ SLURM status check timed out.")
    except Exception as e:
        logger.debug(f"SLURM check failed: {e}")
    
    if not messages:
        return ToolResult(
            success=False,
            tool_name="get_job_status",
            error="No status information available",
            message="❌ Could not get job status. Specify a workflow directory or job ID."
        )
    
    return ToolResult(
        success=True,
        tool_name="get_job_status",
        data=results,
        message="\n\n".join(messages)
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
