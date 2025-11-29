"""
Execution Tools
===============

Tools for submitting and managing SLURM jobs, vLLM management, and job monitoring.

This is a large module (~1450 lines) that handles:
- SLURM job management: submit_job, get_job_status, cancel_job, resubmit_job, list_jobs
- vLLM management: check_system_health, restart_vllm
- Job monitoring: get_logs, watch_job

TODO: Future refactoring could split this into:
- slurm.py: SLURM job operations
- vllm.py: vLLM server management
- monitoring.py: Log viewing and job watching

For now, kept as single file to maintain stability.
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


# =============================================================================
# CHECK_SYSTEM_HEALTH
# =============================================================================

CHECK_SYSTEM_HEALTH_PATTERNS = [
    r"(?:check|show|what\s+is)\s+(?:the\s+)?(?:system\s+)?health",
    r"(?:is\s+the\s+)?(?:system|vllm|gpu|disk)\s+(?:healthy|ok|running)",
    r"health\s+(?:check|status)",
]


def check_system_health_impl() -> ToolResult:
    """
    Check system health including vLLM, GPU, disk, and memory.
    
    Uses HealthChecker from the autonomous module for comprehensive
    system health monitoring.
    
    Returns:
        ToolResult with health status
    """
    try:
        # Try to use HealthChecker
        try:
            from workflow_composer.agents.autonomous.health_checker import (
                HealthChecker,
                HealthStatus,
            )
            use_checker = True
        except ImportError:
            use_checker = False
            logger.debug("HealthChecker not available")
        
        if use_checker:
            checker = HealthChecker()
            
            # Run sync health check
            import asyncio
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            health = loop.run_until_complete(checker.check_all())
            
            # Format results
            status_emoji = {
                HealthStatus.HEALTHY: "✅",
                HealthStatus.DEGRADED: "⚠️",
                HealthStatus.UNHEALTHY: "❌",
                HealthStatus.UNKNOWN: "❓",
            }
            
            overall_emoji = status_emoji.get(health.status, "❓")
            
            components_list = []
            for comp in health.components:
                emoji = status_emoji.get(comp.status, "❓")
                response = f" ({comp.response_time_ms:.0f}ms)" if comp.response_time_ms else ""
                components_list.append(f"  {emoji} **{comp.name}**: {comp.message}{response}")
            
            components_str = "\n".join(components_list)
            
            message = f"""🏥 **System Health Check**

**Overall Status:** {overall_emoji} {health.status.value.upper()}

**Components:**
{components_str}

_Checked at: {health.checked_at.strftime('%Y-%m-%d %H:%M:%S')}_
"""
            
            return ToolResult(
                success=health.status.is_ok,
                tool_name="check_system_health",
                data=health.to_dict(),
                message=message
            )
        
        else:
            # Fallback: basic checks
            import shutil
            import psutil
            
            results = []
            
            # Disk check
            disk = shutil.disk_usage("/")
            disk_percent = (disk.used / disk.total) * 100
            disk_free_gb = disk.free / (1024**3)
            if disk_percent > 90:
                results.append(f"  ❌ **Disk**: {disk_percent:.1f}% used ({disk_free_gb:.1f} GB free)")
            elif disk_percent > 80:
                results.append(f"  ⚠️ **Disk**: {disk_percent:.1f}% used ({disk_free_gb:.1f} GB free)")
            else:
                results.append(f"  ✅ **Disk**: {disk_percent:.1f}% used ({disk_free_gb:.1f} GB free)")
            
            # Memory check
            mem = psutil.virtual_memory()
            if mem.percent > 90:
                results.append(f"  ❌ **Memory**: {mem.percent:.1f}% used")
            elif mem.percent > 80:
                results.append(f"  ⚠️ **Memory**: {mem.percent:.1f}% used")
            else:
                results.append(f"  ✅ **Memory**: {mem.percent:.1f}% used")
            
            # GPU check (if available)
            try:
                gpu_result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5
                )
                if gpu_result.returncode == 0:
                    lines = gpu_result.stdout.strip().split("\n")
                    for i, line in enumerate(lines):
                        parts = line.split(",")
                        if len(parts) >= 3:
                            util = float(parts[0].strip())
                            mem_used = float(parts[1].strip())
                            mem_total = float(parts[2].strip())
                            mem_pct = (mem_used / mem_total) * 100
                            results.append(f"  ✅ **GPU {i}**: {util:.0f}% util, {mem_pct:.0f}% mem")
            except Exception:
                results.append(f"  ❓ **GPU**: not available")
            
            # SLURM check
            try:
                slurm_result = subprocess.run(
                    ["squeue", "--version"],
                    capture_output=True, text=True, timeout=5
                )
                if slurm_result.returncode == 0:
                    results.append(f"  ✅ **SLURM**: available")
                else:
                    results.append(f"  ❌ **SLURM**: error")
            except Exception:
                results.append(f"  ❓ **SLURM**: not available")
            
            message = f"""🏥 **System Health Check**

**Components:**
{chr(10).join(results)}
"""
            
            return ToolResult(
                success=True,
                tool_name="check_system_health",
                data={"components": results},
                message=message
            )
            
    except Exception as e:
        logger.exception("check_system_health failed")
        return ToolResult(
            success=False,
            tool_name="check_system_health",
            error=str(e),
            message=f"❌ Health check error: {e}"
        )


# =============================================================================
# RESTART_VLLM
# =============================================================================

RESTART_VLLM_PATTERNS = [
    r"(?:restart|reboot|reload)\s+(?:the\s+)?(?:vllm|llm|model)\s*(?:server)?",
    r"(?:vllm|llm)\s+(?:is\s+)?(?:down|not\s+working|crashed)",
    r"(?:start|stop|kill)\s+(?:the\s+)?(?:vllm|llm)\s*(?:server)?",
]


def restart_vllm_impl(
    force: bool = False,
    wait_healthy: bool = True,
    timeout: int = 60,
) -> ToolResult:
    """
    Restart the vLLM server.
    
    Args:
        force: Force kill existing processes
        wait_healthy: Wait for server to become healthy
        timeout: Timeout in seconds to wait for healthy status
        
    Returns:
        ToolResult with restart status
    """
    import asyncio
    import time
    
    try:
        from workflow_composer.agents.autonomous.recovery import RecoveryManager
        from workflow_composer.agents.autonomous.health_checker import HealthChecker
        
        recovery = RecoveryManager(require_confirmation=False)
        
        # Execute restart
        async def do_restart():
            return await recovery._restart_vllm(None)
        
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(asyncio.run, do_restart())
                    result = future.result(timeout=timeout + 30)
            else:
                result = loop.run_until_complete(do_restart())
        except RuntimeError:
            result = asyncio.run(do_restart())
        
        if result.success:
            message = f"""✅ **vLLM Server Restarted**

**Status:** Running
**Message:** {result.message}

The vLLM server has been restarted and is ready for requests.
"""
        else:
            message = f"""❌ **vLLM Restart Failed**

**Message:** {result.message}
**Error:** {result.error or 'Unknown error'}

**Manual restart steps:**
1. `pkill -f "vllm.entrypoints"` - Kill existing processes
2. `cd scripts && ./start_server.sh` - Start the server
3. Wait 30-60 seconds for model loading
"""
        
        return ToolResult(
            success=result.success,
            tool_name="restart_vllm",
            data=result.to_dict(),
            message=message
        )
        
    except ImportError as e:
        # Fallback: manual restart
        logger.warning(f"Recovery module not available: {e}")
        
        try:
            # Kill existing vLLM processes
            kill_result = subprocess.run(
                ["pkill", "-f", "vllm.entrypoints"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            
            # Wait a moment
            import time
            time.sleep(3)
            
            # Start server using script
            start_script = Path.cwd() / "scripts" / "start_server.sh"
            if start_script.exists():
                subprocess.Popen(
                    ["bash", str(start_script)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    cwd=str(Path.cwd()),
                )
                
                return ToolResult(
                    success=True,
                    tool_name="restart_vllm",
                    data={"method": "fallback"},
                    message="""✅ **vLLM Server Restart Initiated**

The server restart has been initiated. Please wait 30-60 seconds for the model to load.

Run `check system health` to verify the server is healthy.
"""
                )
            else:
                return ToolResult(
                    success=False,
                    tool_name="restart_vllm",
                    error="Start script not found",
                    message=f"""❌ **Start Script Not Found**

Expected: {start_script}

Please start the server manually or ensure the script exists.
"""
                )
                
        except Exception as e2:
            return ToolResult(
                success=False,
                tool_name="restart_vllm",
                error=str(e2),
                message=f"""❌ **Restart Failed**

Error: {e2}

Manual restart: `cd scripts && ./start_server.sh`
"""
            )
    
    except Exception as e:
        logger.error(f"vLLM restart failed: {e}")
        return ToolResult(
            success=False,
            tool_name="restart_vllm",
            error=str(e),
            message=f"""❌ **vLLM Restart Failed**

Error: {e}

Try manual restart:
1. `pkill -f "vllm.entrypoints"`
2. `cd scripts && ./start_server.sh`
"""
        )


# =============================================================================
# RESUBMIT_JOB
# =============================================================================

RESUBMIT_JOB_PATTERNS = [
    r"resubmit\s+(?:the\s+)?(?:job|slurm)",
    r"resubmit\s+(?:job\s+)?(?:id\s*)?(\d+)",
    r"retry\s+(?:the\s+)?(?:failed\s+)?job",
    r"rerun\s+(?:the\s+)?(?:failed\s+)?job",
    r"(?:job\s+)?(\d+)\s+(?:failed|crashed).+(?:resubmit|retry)",
]


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
        from workflow_composer.agents.autonomous.job_monitor import JobMonitor, JobInfo
        
        # Get job info if job_id provided
        job_info = None
        if job_id:
            monitor = JobMonitor()
            job_info = monitor.get_job_info(job_id)
            
            if job_info is None:
                return ToolResult(
                    success=False,
                    tool_name="resubmit_job",
                    error=f"Could not find job {job_id}",
                    message=f"""❌ **Job Not Found**

Could not retrieve information for job ID: {job_id}

This might happen if:
- The job is too old and not in SLURM's history
- The job ID is incorrect
- SLURM accounting is not enabled

**Alternative:** Provide the script path directly:
`resubmit job script=/path/to/script.sh`
"""
                )
        
        # Find script if not provided
        if not script_path and job_info and job_info.working_dir:
            for pattern in ["*.sh", "*.slurm", "submit_*.sh", "run_*.sh"]:
                scripts = list(job_info.working_dir.glob(pattern))
                if scripts:
                    script_path = str(scripts[0])
                    break
        
        if not script_path:
            return ToolResult(
                success=False,
                tool_name="resubmit_job",
                error="No submission script found",
                message="""❌ **No Submission Script Found**

Could not locate the job submission script.

Please provide the script path:
`resubmit job script=/path/to/your_job.sh`
"""
            )
        
        script_file = Path(script_path)
        if not script_file.exists():
            return ToolResult(
                success=False,
                tool_name="resubmit_job",
                error=f"Script not found: {script_path}",
                message=f"❌ Script not found: {script_path}"
            )
        
        # Modify resources if requested
        if modify_resources:
            # Read the script
            with open(script_file, 'r') as f:
                script_content = f.read()
            
            # Apply modifications
            import re
            for key, value in modify_resources.items():
                if key == "mem":
                    script_content = re.sub(
                        r'#SBATCH\s+--mem=\S+',
                        f'#SBATCH --mem={value}',
                        script_content
                    )
                elif key == "time":
                    script_content = re.sub(
                        r'#SBATCH\s+--time=\S+',
                        f'#SBATCH --time={value}',
                        script_content
                    )
                elif key == "cpus":
                    script_content = re.sub(
                        r'#SBATCH\s+--cpus-per-task=\d+',
                        f'#SBATCH --cpus-per-task={value}',
                        script_content
                    )
            
            # Write modified script
            modified_script = script_file.parent / f"{script_file.stem}_retry{script_file.suffix}"
            with open(modified_script, 'w') as f:
                f.write(script_content)
            script_path = str(modified_script)
        
        # Submit the job
        working_dir = job_info.working_dir if job_info else script_file.parent
        
        result = subprocess.run(
            ["sbatch", script_path],
            capture_output=True,
            text=True,
            cwd=str(working_dir),
            timeout=30,
        )
        
        if result.returncode == 0:
            # Parse new job ID
            import re
            match = re.search(r"Submitted batch job (\d+)", result.stdout)
            new_job_id = match.group(1) if match else None
            
            message = f"""✅ **Job Resubmitted Successfully**

**New Job ID:** {new_job_id}
**Script:** {script_path}
**Working Directory:** {working_dir}
"""
            if job_id:
                message += f"\n**Original Job ID:** {job_id}"
            if modify_resources:
                message += f"\n**Modified Resources:** {modify_resources}"
            
            message += f"""

**Monitor the job:**
- `get job status {new_job_id}`
- `squeue -j {new_job_id}`
"""
            
            return ToolResult(
                success=True,
                tool_name="resubmit_job",
                data={
                    "new_job_id": new_job_id,
                    "original_job_id": job_id,
                    "script": script_path,
                    "working_dir": str(working_dir),
                    "modified_resources": modify_resources,
                },
                message=message
            )
        else:
            return ToolResult(
                success=False,
                tool_name="resubmit_job",
                error=result.stderr,
                message=f"""❌ **Job Submission Failed**

**Error:** {result.stderr}

**Script:** {script_path}

Check that:
1. The script is valid
2. You have access to the partition
3. Resources are available
"""
            )
        
    except ImportError as e:
        # Fallback: direct sbatch
        logger.warning(f"Job monitor not available: {e}")
        
        if not script_path:
            return ToolResult(
                success=False,
                tool_name="resubmit_job",
                error="script_path required when job_monitor unavailable",
                message="""❌ **Script Path Required**

Please provide the script path:
`resubmit job script=/path/to/script.sh`
"""
            )
        
        try:
            result = subprocess.run(
                ["sbatch", script_path],
                capture_output=True,
                text=True,
                timeout=30,
            )
            
            if result.returncode == 0:
                import re
                match = re.search(r"Submitted batch job (\d+)", result.stdout)
                new_job_id = match.group(1) if match else "unknown"
                
                return ToolResult(
                    success=True,
                    tool_name="resubmit_job",
                    data={"new_job_id": new_job_id, "script": script_path},
                    message=f"""✅ **Job Submitted**

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
    
    except Exception as e:
        logger.error(f"resubmit_job failed: {e}")
        return ToolResult(
            success=False,
            tool_name="resubmit_job",
            error=str(e),
            message=f"❌ Resubmit failed: {e}"
        )


# =============================================================================
# WATCH_JOB - Rich job monitoring with detailed info
# =============================================================================

WATCH_JOB_PATTERNS = [
    r"watch\s+(?:the\s+)?job\s*(?:id\s*)?(\d+)?",
    r"monitor\s+(?:the\s+)?job\s*(?:id\s*)?(\d+)?",
    r"track\s+(?:the\s+)?job\s*(?:id\s*)?(\d+)?",
    r"(?:detailed|full)\s+(?:job\s+)?status\s+(?:for\s+)?(\d+)?",
]


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


# =============================================================================
# LIST_JOBS - Quick overview of all user's SLURM jobs
# =============================================================================

LIST_JOBS_PATTERNS = [
    r"list\s+(?:my\s+)?(?:slurm\s+)?jobs",
    r"show\s+(?:my\s+)?(?:slurm\s+)?jobs",
    r"(?:my|all)\s+jobs",
    r"what\s+jobs\s+(?:are|do\s+i\s+have)\s+running",
    r"squeue",
]


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
