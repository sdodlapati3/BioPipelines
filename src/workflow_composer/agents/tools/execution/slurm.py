"""
SLURM Job Management Tools
==========================

Tools for submitting, canceling, resubmitting, and listing SLURM jobs.

Functions:
    - submit_job_impl: Submit workflow to SLURM
    - get_job_status_impl: Check job status
    - cancel_job_impl: Cancel a job
    - resubmit_job_impl: Resubmit a failed job
    - list_jobs_impl: List user's jobs
"""

import logging
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..base import ToolResult

logger = logging.getLogger(__name__)


# =============================================================================
# PATTERNS
# =============================================================================

SUBMIT_JOB_PATTERNS = [
    r"(?:run|execute|submit|start)\s+(?:it|the workflow|this|pipeline)\s*(?:on|with|using)?\s*(slurm|local|docker)?",
    r"(?:run|execute|submit)\s+(?:workflow|pipeline)?\s*['\"]?([^'\"]+)['\"]?\s*(?:on|with)?\s*(slurm|local|docker)?",
]

GET_JOB_STATUS_PATTERNS = [
    r"(?:what(?:'s| is)|show|check)\s+(?:the\s+)?(?:job\s+)?(?:status|progress)",
    r"(?:how(?:'s| is))\s+(?:the\s+)?(?:job|workflow|pipeline)\s*(?:doing|going|running)?",
    r"(?:check|show)\s+(?:nextflow|workflow)\s+(?:status|progress)",
    r"(?:job|slurm)\s+status",
]

CANCEL_JOB_PATTERNS = [
    r"(?:cancel|stop|abort|kill)\s+(?:the\s+)?(?:job|workflow|pipeline)\s*(\d+)?",
]

RESUBMIT_JOB_PATTERNS = [
    r"resubmit\s+(?:the\s+)?(?:job|slurm)",
    r"resubmit\s+(?:job\s+)?(?:id\s*)?(\d+)",
    r"retry\s+(?:the\s+)?(?:failed\s+)?job",
    r"rerun\s+(?:the\s+)?(?:failed\s+)?job",
    r"(?:job\s+)?(\d+)\s+(?:failed|crashed).+(?:resubmit|retry)",
]

LIST_JOBS_PATTERNS = [
    r"list\s+(?:my\s+)?(?:slurm\s+)?jobs",
    r"show\s+(?:my\s+)?(?:slurm\s+)?jobs",
    r"(?:my|all)\s+jobs",
    r"what\s+jobs\s+(?:are|do\s+i\s+have)\s+running",
    r"squeue",
]


# =============================================================================
# SUBMIT_JOB
# =============================================================================

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
            monitor = WorkflowMonitor(workflow_dir or ".")
            status = monitor.get_workflow_status()
            
            if status:
                results["nextflow"] = status
                
                # Format Nextflow status
                nf_msg = f"""📊 **Nextflow Workflow Status**

**Run Name:** {status.get('run_name', 'N/A')}
**Status:** {status.get('status', 'Unknown')}
**Started:** {status.get('start_time', 'N/A')}
**Duration:** {status.get('duration', 'N/A')}

**Progress:** {status.get('completed', 0)}/{status.get('total', 0)} tasks completed
"""
                messages.append(nf_msg)
    
    # Check SLURM jobs
    try:
        cmd = ["squeue", "--format=%i|%j|%t|%M|%R", "--noheader"]
        if job_id:
            cmd.extend(["-j", str(job_id)])
        else:
            cmd.append("--me")
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0 and result.stdout.strip():
            jobs = []
            for line in result.stdout.strip().split("\n"):
                parts = line.split("|")
                if len(parts) >= 5:
                    jobs.append({
                        "id": parts[0],
                        "name": parts[1],
                        "state": parts[2],
                        "time": parts[3],
                        "reason": parts[4],
                    })
            
            results["slurm_jobs"] = jobs
            
            # Format SLURM status
            if jobs:
                job_list = "\n".join([
                    f"  • **{j['id']}**: {j['name']} | {j['state']} | {j['time']}"
                    for j in jobs
                ])
                slurm_msg = f"""🖥️ **SLURM Jobs**

{job_list}

Use `watch job <id>` for detailed info.
"""
                messages.append(slurm_msg)
        elif not job_id:
            messages.append("📋 No SLURM jobs currently running.")
    except FileNotFoundError:
        if job_id:
            messages.append("⚠️ SLURM not available on this system.")
    except Exception as e:
        logger.warning(f"Error checking SLURM: {e}")
    
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
# CANCEL_JOB
# =============================================================================

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


# =============================================================================
# RESUBMIT_JOB
# =============================================================================

def resubmit_job_impl(
    job_id: str = None,
    script_path: str = None,
    modify_resources: Dict[str, Any] = None,
) -> ToolResult:
    """
    Resubmit a failed SLURM job.
    
    Args:
        job_id: The original job ID to resubmit
        script_path: Path to submission script (auto-detected if job_id provided)
        modify_resources: Dict of resource modifications (e.g., {"mem": "32G", "time": "4:00:00"})
        
    Returns:
        ToolResult with new job ID
    """
    import asyncio
    
    try:
        from workflow_composer.agents.autonomous.recovery import RecoveryManager
        
        # Use RecoveryManager for smart resubmission
        recovery = RecoveryManager(require_confirmation=False)
        
        async def do_resubmit():
            # Build context
            context = {}
            if job_id:
                context["job_id"] = job_id
            if modify_resources:
                context["resources"] = modify_resources
            
            return await recovery._resubmit_job(context)
        
        # Execute async
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(asyncio.run, do_resubmit())
                    result = future.result(timeout=60)
            else:
                result = loop.run_until_complete(do_resubmit())
        except RuntimeError:
            result = asyncio.run(do_resubmit())
        
        if result.success:
            message = f"""✅ **Job Resubmitted**

{result.message}

**Original Job:** {job_id or 'N/A'}
**Action Taken:** {result.action.value if result.action else 'resubmit'}

Use `watch job` to monitor the new job.
"""
        else:
            message = f"""❌ **Resubmit Failed**

{result.message}

**Error:** {result.error or 'Unknown error'}

Try manually with: `sbatch <script.sh>`
"""
        
        return ToolResult(
            success=result.success,
            tool_name="resubmit_job",
            data=result.to_dict() if hasattr(result, 'to_dict') else {},
            message=message
        )
        
    except ImportError as e:
        # Fallback: manual resubmission
        logger.warning(f"RecoveryManager not available: {e}")
        
        if not script_path and job_id:
            # Try to find script from job info
            try:
                result = subprocess.run(
                    ["scontrol", "show", "job", str(job_id)],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                
                if result.returncode == 0:
                    for line in result.stdout.split("\n"):
                        if "Command=" in line:
                            script_path = line.split("Command=")[1].strip()
                            break
            except Exception:
                pass
        
        if script_path and Path(script_path).exists():
            try:
                result = subprocess.run(
                    ["sbatch", script_path],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                
                if result.returncode == 0:
                    # Parse job ID from output
                    import re
                    match = re.search(r"Submitted batch job (\d+)", result.stdout)
                    new_job_id = match.group(1) if match else "Unknown"
                    
                    return ToolResult(
                        success=True,
                        tool_name="resubmit_job",
                        data={"new_job_id": new_job_id, "script": script_path},
                        message=f"""✅ **Job Resubmitted**

**New Job ID:** {new_job_id}
**Script:** {script_path}
"""
                    )
                else:
                    return ToolResult(
                        success=False,
                        tool_name="resubmit_job",
                        error=result.stderr,
                        message=f"❌ Submission failed: {result.stderr}"
                    )
            except Exception as e2:
                return ToolResult(
                    success=False,
                    tool_name="resubmit_job",
                    error=str(e2),
                    message=f"❌ Submission error: {e2}"
                )
        else:
            return ToolResult(
                success=False,
                tool_name="resubmit_job",
                error="No script found",
                message=f"""❌ **Cannot Find Submission Script**

Provide the script path: `resubmit job script=/path/to/script.sh`

Or check job info: `scontrol show job {job_id}`
"""
            )
    
    except Exception as e:
        logger.error(f"resubmit_job failed: {e}")
        return ToolResult(
            success=False,
            tool_name="resubmit_job",
            error=str(e),
            message=f"❌ Resubmit failed: {e}"
        )


# =============================================================================
# LIST_JOBS
# =============================================================================

def list_jobs_impl(
    user: str = None,
    state: str = None,
    partition: str = None,
) -> ToolResult:
    """
    List SLURM jobs for the current user.
    
    Args:
        user: Filter by user (default: current user)
        state: Filter by state (pending, running, completed, failed)
        partition: Filter by partition
        
    Returns:
        ToolResult with job list
    """
    try:
        from workflow_composer.agents.autonomous.job_monitor import JobMonitor, JobState
        
        # Get running jobs
        jobs = JobMonitor.list_running_jobs(user=user)
        
        if not jobs:
            return ToolResult(
                success=True,
                tool_name="list_jobs",
                data={"jobs": []},
                message="""📋 **No Running Jobs**

You don't have any jobs currently running or pending.

To submit a workflow: `submit workflow`
To check recent completed jobs: `sacct --starttime=today`
"""
            )
        
        # Filter by state if specified
        if state:
            state_map = {
                "pending": ["PD"],
                "running": ["R"],
                "completing": ["CG"],
            }
            filter_states = state_map.get(state.lower(), [state.upper()])
            jobs = [j for j in jobs if j["state"] in filter_states]
        
        # Filter by partition if specified
        if partition:
            jobs = [j for j in jobs if j["partition"] == partition]
        
        if not jobs:
            return ToolResult(
                success=True,
                tool_name="list_jobs",
                data={"jobs": []},
                message=f"📋 No jobs matching filters (state={state}, partition={partition})"
            )
        
        # State emoji
        state_emoji = {
            "PD": "⏳",
            "R": "🏃",
            "CG": "🔄",
        }
        
        # Build table
        job_rows = []
        for j in jobs:
            emoji = state_emoji.get(j["state"], "❓")
            job_rows.append(
                f"| {j['job_id']} | {j['name'][:25]} | {emoji} {j['state']} | {j['time']} | {j['partition']} | {j['nodelist']} |"
            )
        
        table = "\n".join(job_rows)
        
        # Count by state
        pending = sum(1 for j in jobs if j["state"] == "PD")
        running = sum(1 for j in jobs if j["state"] == "R")
        
        message = f"""📋 **Your SLURM Jobs** ({len(jobs)} total)

| Job ID | Name | State | Time | Partition | Node(s) |
|--------|------|-------|------|-----------|---------|
{table}

**Summary:** ⏳ {pending} pending, 🏃 {running} running

**Commands:**
- `watch job <id>` - Detailed info on specific job
- `cancel job <id>` - Cancel a job
- `get logs <id>` - View job output
"""
        
        return ToolResult(
            success=True,
            tool_name="list_jobs",
            data={
                "jobs": jobs,
                "pending": pending,
                "running": running,
            },
            message=message
        )
        
    except ImportError:
        # Fallback to direct squeue
        try:
            cmd = ["squeue", "--format=%i|%j|%t|%M|%P|%N", "--noheader"]
            if user:
                cmd.extend(["-u", user])
            else:
                cmd.append("--me")
            
            if partition:
                cmd.extend(["-p", partition])
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
            if result.returncode != 0:
                return ToolResult(
                    success=False,
                    tool_name="list_jobs",
                    error=result.stderr,
                    message=f"❌ squeue failed: {result.stderr}"
                )
            
            if not result.stdout.strip():
                return ToolResult(
                    success=True,
                    tool_name="list_jobs",
                    data={"jobs": []},
                    message="📋 No jobs currently running or pending."
                )
            
            # Parse output
            jobs = []
            for line in result.stdout.strip().split("\n"):
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
            
            # Build simple output
            job_lines = [f"  • **{j['job_id']}**: {j['name']} ({j['state']}) - {j['time']}" for j in jobs]
            
            return ToolResult(
                success=True,
                tool_name="list_jobs",
                data={"jobs": jobs},
                message=f"""📋 **Your Jobs** ({len(jobs)} total)

{chr(10).join(job_lines)}
"""
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                tool_name="list_jobs",
                error=str(e),
                message=f"❌ Failed to list jobs: {e}"
            )
    
    except Exception as e:
        logger.error(f"list_jobs failed: {e}")
        return ToolResult(
            success=False,
            tool_name="list_jobs",
            error=str(e),
            message=f"❌ List jobs failed: {e}"
        )


__all__ = [
    # Patterns
    "SUBMIT_JOB_PATTERNS",
    "GET_JOB_STATUS_PATTERNS",
    "CANCEL_JOB_PATTERNS",
    "RESUBMIT_JOB_PATTERNS",
    "LIST_JOBS_PATTERNS",
    # Functions
    "submit_job_impl",
    "get_job_status_impl",
    "cancel_job_impl",
    "resubmit_job_impl",
    "list_jobs_impl",
]
