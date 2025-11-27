"""
Enhanced Tool Wrappers for Autonomous Agent
=============================================

This module provides enhanced tool wrappers that:
- Use the executor layer for safe command execution
- Include retry logic with exponential backoff
- Provide detailed error context for diagnosis
- Support async execution for long-running operations
- Log all actions for audit trail

These tools are designed for use by the AutonomousAgent.
"""

import asyncio
import logging
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


# =============================================================================
# Tool Result Types
# =============================================================================

class ToolStatus(Enum):
    """Status of tool execution."""
    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"  # Some operations succeeded
    TIMEOUT = "timeout"
    RETRY_EXHAUSTED = "retry_exhausted"
    PERMISSION_DENIED = "permission_denied"


@dataclass
class EnhancedToolResult:
    """Enhanced result from tool execution with detailed context."""
    
    status: ToolStatus
    tool_name: str
    data: Any = None
    message: str = ""
    error: Optional[str] = None
    error_context: Optional[Dict[str, Any]] = None
    
    # Execution metadata
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_seconds: float = 0.0
    retry_count: int = 0
    
    # For UI updates
    ui_update: Optional[Dict[str, Any]] = None
    
    # Recovery suggestions
    recovery_suggestions: List[str] = field(default_factory=list)
    
    @property
    def success(self) -> bool:
        """Check if operation was successful."""
        return self.status == ToolStatus.SUCCESS
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "status": self.status.value,
            "tool_name": self.tool_name,
            "data": self.data,
            "message": self.message,
            "error": self.error,
            "error_context": self.error_context,
            "duration_seconds": self.duration_seconds,
            "retry_count": self.retry_count,
            "recovery_suggestions": self.recovery_suggestions,
        }


# =============================================================================
# Retry Decorator with Exponential Backoff
# =============================================================================

def with_retry(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    exponential: bool = True,
    retryable_exceptions: Tuple = (Exception,),
):
    """
    Decorator for retry logic with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay between retries (seconds)
        max_delay: Maximum delay between retries (seconds)
        exponential: Use exponential backoff
        retryable_exceptions: Exceptions that trigger a retry
    """
    def decorator(func: Callable):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            last_exception = None
            retry_count = 0
            
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e
                    retry_count = attempt + 1
                    
                    if attempt < max_retries:
                        delay = base_delay * (2 ** attempt if exponential else 1)
                        delay = min(delay, max_delay)
                        logger.warning(
                            f"{func.__name__} failed (attempt {attempt + 1}/{max_retries + 1}), "
                            f"retrying in {delay:.1f}s: {e}"
                        )
                        await asyncio.sleep(delay)
                    else:
                        logger.error(f"{func.__name__} failed after {max_retries + 1} attempts: {e}")
            
            raise last_exception
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            last_exception = None
            retry_count = 0
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e
                    retry_count = attempt + 1
                    
                    if attempt < max_retries:
                        delay = base_delay * (2 ** attempt if exponential else 1)
                        delay = min(delay, max_delay)
                        logger.warning(
                            f"{func.__name__} failed (attempt {attempt + 1}/{max_retries + 1}), "
                            f"retrying in {delay:.1f}s: {e}"
                        )
                        time.sleep(delay)
                    else:
                        logger.error(f"{func.__name__} failed after {max_retries + 1} attempts: {e}")
            
            raise last_exception
        
        # Return appropriate wrapper based on whether function is async
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator


# =============================================================================
# Enhanced Tool Registry
# =============================================================================

class EnhancedToolRegistry:
    """Registry of enhanced tools with metadata."""
    
    def __init__(self):
        self._tools: Dict[str, "EnhancedTool"] = {}
    
    def register(self, tool: "EnhancedTool"):
        """Register a tool."""
        self._tools[tool.name] = tool
        logger.debug(f"Registered tool: {tool.name}")
    
    def get(self, name: str) -> Optional["EnhancedTool"]:
        """Get a tool by name."""
        return self._tools.get(name)
    
    def list_tools(self) -> List[str]:
        """List all tool names."""
        return list(self._tools.keys())
    
    def get_tool_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all tools."""
        return {
            name: {
                "description": tool.description,
                "requires_permission": tool.requires_permission,
                "is_async": tool.is_async,
            }
            for name, tool in self._tools.items()
        }


# Global registry
_tool_registry = EnhancedToolRegistry()


def get_tool_registry() -> EnhancedToolRegistry:
    """Get the global tool registry."""
    return _tool_registry


# =============================================================================
# Base Enhanced Tool Class
# =============================================================================

@dataclass
class EnhancedTool:
    """Base class for enhanced tools."""
    
    name: str
    description: str
    requires_permission: bool = False
    is_async: bool = True
    timeout_seconds: float = 300.0  # 5 minutes default
    max_retries: int = 3
    
    async def execute(self, *args, **kwargs) -> EnhancedToolResult:
        """Execute the tool. Override in subclasses."""
        raise NotImplementedError
    
    def execute_sync(self, *args, **kwargs) -> EnhancedToolResult:
        """Synchronous execution wrapper."""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(self.execute(*args, **kwargs))


# =============================================================================
# SLURM Tools
# =============================================================================

class SLURMTool(EnhancedTool):
    """Base class for SLURM-related tools."""
    
    def __init__(self, name: str, description: str):
        super().__init__(
            name=name,
            description=description,
            requires_permission=True,
            is_async=True,
        )
        self._executor = None
    
    def _get_executor(self):
        """Get or create command executor."""
        if self._executor is None:
            try:
                from workflow_composer.agents.executor import CommandSandbox
                self._executor = CommandSandbox(
                    workspace_root=Path.home() / "BioPipelines"
                )
            except ImportError:
                logger.warning("CommandSandbox not available")
        return self._executor


class SLURMSubmitTool(SLURMTool):
    """Submit a job to SLURM."""
    
    def __init__(self):
        super().__init__(
            name="slurm_submit",
            description="Submit a workflow to SLURM cluster"
        )
        self.timeout_seconds = 60.0
    
    @with_retry(max_retries=2, base_delay=5.0)
    async def execute(
        self,
        script_path: str,
        partition: str = "h100quadflex",
        nodes: int = 1,
        time_limit: str = "24:00:00",
        **kwargs
    ) -> EnhancedToolResult:
        """
        Submit a job to SLURM.
        
        Args:
            script_path: Path to the submission script
            partition: SLURM partition
            nodes: Number of nodes
            time_limit: Job time limit
            
        Returns:
            EnhancedToolResult with job ID
        """
        start_time = datetime.now()
        
        script_path = Path(script_path)
        if not script_path.exists():
            return EnhancedToolResult(
                status=ToolStatus.FAILURE,
                tool_name=self.name,
                error=f"Script not found: {script_path}",
                message=f"❌ Script not found: `{script_path}`",
                start_time=start_time,
                end_time=datetime.now(),
                recovery_suggestions=[
                    f"Check if script exists: ls -la {script_path}",
                    "Regenerate the workflow script",
                ]
            )
        
        # Build sbatch command
        cmd = [
            "sbatch",
            f"--partition={partition}",
            f"--nodes={nodes}",
            f"--time={time_limit}",
            str(script_path)
        ]
        
        try:
            executor = self._get_executor()
            
            if executor:
                result = await asyncio.to_thread(
                    executor.run_command,
                    " ".join(cmd),
                    timeout=self.timeout_seconds,
                )
                
                if result.return_code != 0:
                    return EnhancedToolResult(
                        status=ToolStatus.FAILURE,
                        tool_name=self.name,
                        error=result.stderr,
                        error_context={"command": " ".join(cmd), "return_code": result.return_code},
                        message=f"❌ SLURM submission failed: {result.stderr}",
                        start_time=start_time,
                        end_time=datetime.now(),
                        recovery_suggestions=self._get_recovery_suggestions(result.stderr),
                    )
                
                # Parse job ID from output
                # "Submitted batch job 12345"
                output = result.stdout.strip()
                job_id = None
                if "Submitted batch job" in output:
                    parts = output.split()
                    job_id = parts[-1]
                
                return EnhancedToolResult(
                    status=ToolStatus.SUCCESS,
                    tool_name=self.name,
                    data={"job_id": job_id, "script": str(script_path)},
                    message=f"🚀 Submitted job **{job_id}** to {partition}",
                    start_time=start_time,
                    end_time=datetime.now(),
                    duration_seconds=(datetime.now() - start_time).total_seconds(),
                    ui_update={"active_job": job_id},
                )
            else:
                # Direct subprocess fallback
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=self.timeout_seconds
                )
                
                if proc.returncode != 0:
                    return EnhancedToolResult(
                        status=ToolStatus.FAILURE,
                        tool_name=self.name,
                        error=stderr.decode(),
                        message=f"❌ SLURM submission failed",
                        start_time=start_time,
                        end_time=datetime.now(),
                    )
                
                output = stdout.decode().strip()
                job_id = output.split()[-1] if "Submitted" in output else None
                
                return EnhancedToolResult(
                    status=ToolStatus.SUCCESS,
                    tool_name=self.name,
                    data={"job_id": job_id},
                    message=f"🚀 Submitted job **{job_id}**",
                    start_time=start_time,
                    end_time=datetime.now(),
                )
                
        except asyncio.TimeoutError:
            return EnhancedToolResult(
                status=ToolStatus.TIMEOUT,
                tool_name=self.name,
                error="SLURM submission timed out",
                message="⏱️ SLURM submission timed out",
                start_time=start_time,
                end_time=datetime.now(),
                recovery_suggestions=[
                    "Check SLURM controller status: sinfo",
                    "Try again with shorter timeout",
                ]
            )
        except Exception as e:
            return EnhancedToolResult(
                status=ToolStatus.FAILURE,
                tool_name=self.name,
                error=str(e),
                message=f"❌ Error submitting job: {e}",
                start_time=start_time,
                end_time=datetime.now(),
            )
    
    def _get_recovery_suggestions(self, error: str) -> List[str]:
        """Get recovery suggestions based on error message."""
        suggestions = []
        error_lower = error.lower()
        
        if "partition" in error_lower:
            suggestions.append("Check available partitions: sinfo")
            suggestions.append("Try partition 'compute' or 'gpu'")
        if "memory" in error_lower or "mem" in error_lower:
            suggestions.append("Reduce memory request")
            suggestions.append("Check available memory: scontrol show partition")
        if "time" in error_lower:
            suggestions.append("Reduce time limit")
        if "permission" in error_lower or "denied" in error_lower:
            suggestions.append("Check account/QOS permissions")
            suggestions.append("Contact system administrator")
        
        return suggestions


class SLURMStatusTool(SLURMTool):
    """Check SLURM job status."""
    
    def __init__(self):
        super().__init__(
            name="slurm_status",
            description="Check status of SLURM jobs"
        )
        self.timeout_seconds = 30.0
    
    async def execute(self, job_id: str = None) -> EnhancedToolResult:
        """
        Check job status.
        
        Args:
            job_id: Optional specific job ID
            
        Returns:
            EnhancedToolResult with job status
        """
        start_time = datetime.now()
        
        try:
            if job_id:
                cmd = ["squeue", "-j", str(job_id), "-o", "%i|%j|%T|%M|%R"]
            else:
                cmd = ["squeue", "--me", "-o", "%i|%j|%T|%M|%R"]
            
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=self.timeout_seconds
            )
            
            output = stdout.decode().strip()
            lines = output.split('\n')
            
            jobs = []
            if len(lines) > 1:  # Has header + data
                for line in lines[1:]:
                    parts = line.split('|')
                    if len(parts) >= 5:
                        jobs.append({
                            "job_id": parts[0],
                            "name": parts[1],
                            "state": parts[2],
                            "time": parts[3],
                            "reason": parts[4],
                        })
            
            if jobs:
                job_lines = "\n".join([
                    f"  - `{j['job_id']}`: {j['state']} ({j['time']})"
                    for j in jobs[:10]
                ])
                message = f"📊 **Jobs:**\n\n{job_lines}"
            else:
                message = "📋 No active jobs found"
            
            return EnhancedToolResult(
                status=ToolStatus.SUCCESS,
                tool_name=self.name,
                data={"jobs": jobs},
                message=message,
                start_time=start_time,
                end_time=datetime.now(),
            )
            
        except Exception as e:
            return EnhancedToolResult(
                status=ToolStatus.FAILURE,
                tool_name=self.name,
                error=str(e),
                message=f"❌ Failed to get job status: {e}",
                start_time=start_time,
                end_time=datetime.now(),
            )


class SLURMCancelTool(SLURMTool):
    """Cancel SLURM jobs."""
    
    def __init__(self):
        super().__init__(
            name="slurm_cancel",
            description="Cancel a SLURM job"
        )
        self.timeout_seconds = 30.0
    
    async def execute(self, job_id: str) -> EnhancedToolResult:
        """
        Cancel a SLURM job.
        
        Args:
            job_id: Job ID to cancel
            
        Returns:
            EnhancedToolResult confirming cancellation
        """
        start_time = datetime.now()
        
        if not job_id:
            return EnhancedToolResult(
                status=ToolStatus.FAILURE,
                tool_name=self.name,
                error="No job ID provided",
                message="❌ Please specify a job ID to cancel",
                start_time=start_time,
                end_time=datetime.now(),
            )
        
        try:
            proc = await asyncio.create_subprocess_exec(
                "scancel", str(job_id),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=self.timeout_seconds
            )
            
            if proc.returncode != 0:
                return EnhancedToolResult(
                    status=ToolStatus.FAILURE,
                    tool_name=self.name,
                    error=stderr.decode(),
                    message=f"❌ Failed to cancel job: {stderr.decode()}",
                    start_time=start_time,
                    end_time=datetime.now(),
                )
            
            return EnhancedToolResult(
                status=ToolStatus.SUCCESS,
                tool_name=self.name,
                data={"job_id": job_id, "cancelled": True},
                message=f"🛑 Cancelled job **{job_id}**",
                start_time=start_time,
                end_time=datetime.now(),
            )
            
        except Exception as e:
            return EnhancedToolResult(
                status=ToolStatus.FAILURE,
                tool_name=self.name,
                error=str(e),
                message=f"❌ Error cancelling job: {e}",
                start_time=start_time,
                end_time=datetime.now(),
            )


class SLURMLogsTool(SLURMTool):
    """Get SLURM job logs."""
    
    def __init__(self):
        super().__init__(
            name="slurm_logs",
            description="Get logs from a SLURM job"
        )
        self.timeout_seconds = 60.0
    
    async def execute(
        self,
        job_id: str,
        lines: int = 100,
        log_type: str = "both"  # "stdout", "stderr", "both"
    ) -> EnhancedToolResult:
        """
        Get job logs.
        
        Args:
            job_id: Job ID
            lines: Number of lines to retrieve
            log_type: Type of logs to get
            
        Returns:
            EnhancedToolResult with log content
        """
        start_time = datetime.now()
        
        # Common log locations
        log_patterns = [
            Path.home() / f"slurm-{job_id}.out",
            Path.home() / "BioPipelines" / "logs" / f"slurm-{job_id}.out",
            Path(f"/tmp/slurm-{job_id}.out"),
        ]
        
        log_file = None
        for pattern in log_patterns:
            if pattern.exists():
                log_file = pattern
                break
        
        if not log_file:
            # Try to find any matching log
            try:
                home_logs = list(Path.home().glob(f"*{job_id}*.out"))
                if home_logs:
                    log_file = home_logs[0]
            except Exception:
                pass
        
        if not log_file or not log_file.exists():
            return EnhancedToolResult(
                status=ToolStatus.FAILURE,
                tool_name=self.name,
                error=f"Log file not found for job {job_id}",
                message=f"❌ No log file found for job `{job_id}`",
                start_time=start_time,
                end_time=datetime.now(),
                recovery_suggestions=[
                    f"Job may still be starting",
                    f"Check for log files: ls -la ~/slurm-{job_id}*",
                    "Wait a few seconds and try again",
                ]
            )
        
        try:
            # Read last N lines
            proc = await asyncio.create_subprocess_exec(
                "tail", "-n", str(lines), str(log_file),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=self.timeout_seconds
            )
            
            log_content = stdout.decode()
            
            return EnhancedToolResult(
                status=ToolStatus.SUCCESS,
                tool_name=self.name,
                data={
                    "logs": log_content,
                    "log_file": str(log_file),
                    "lines": lines,
                },
                message=f"📄 **Logs** ({log_file.name}, last {lines} lines):\n\n```\n{log_content[:2000]}\n```",
                start_time=start_time,
                end_time=datetime.now(),
            )
            
        except Exception as e:
            return EnhancedToolResult(
                status=ToolStatus.FAILURE,
                tool_name=self.name,
                error=str(e),
                message=f"❌ Error reading logs: {e}",
                start_time=start_time,
                end_time=datetime.now(),
            )


# =============================================================================
# vLLM Tools
# =============================================================================

class VLLMTool(EnhancedTool):
    """Base class for vLLM-related tools."""
    
    def __init__(self, name: str, description: str):
        super().__init__(
            name=name,
            description=description,
            requires_permission=False,
            is_async=True,
        )
        self.vllm_url = "http://localhost:8000"
    
    async def _check_vllm_health(self) -> Tuple[bool, str]:
        """Check if vLLM server is healthy."""
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.vllm_url}/health",
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    if response.status == 200:
                        return True, "healthy"
                    return False, f"status {response.status}"
        except Exception as e:
            return False, str(e)


class VLLMQueryTool(VLLMTool):
    """Query the vLLM server."""
    
    def __init__(self):
        super().__init__(
            name="vllm_query",
            description="Query the local LLM server"
        )
        self.timeout_seconds = 120.0
        self.max_retries = 2
    
    @with_retry(max_retries=2, base_delay=2.0)
    async def execute(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        system_prompt: str = None,
    ) -> EnhancedToolResult:
        """
        Query the vLLM server.
        
        Args:
            prompt: The prompt to send
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            system_prompt: Optional system prompt
            
        Returns:
            EnhancedToolResult with LLM response
        """
        start_time = datetime.now()
        
        # Check health first
        healthy, status = await self._check_vllm_health()
        if not healthy:
            return EnhancedToolResult(
                status=ToolStatus.FAILURE,
                tool_name=self.name,
                error=f"vLLM server not available: {status}",
                message="❌ LLM server is not available",
                start_time=start_time,
                end_time=datetime.now(),
                recovery_suggestions=[
                    "Start vLLM server: ./scripts/start_vllm.sh",
                    "Check GPU availability: nvidia-smi",
                    "Check server logs",
                ]
            )
        
        try:
            import aiohttp
            
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            payload = {
                "model": "MiniMax-M2",  # Or detect from server
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.vllm_url}/v1/chat/completions",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.timeout_seconds)
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        return EnhancedToolResult(
                            status=ToolStatus.FAILURE,
                            tool_name=self.name,
                            error=f"HTTP {response.status}: {error_text}",
                            message="❌ LLM query failed",
                            start_time=start_time,
                            end_time=datetime.now(),
                        )
                    
                    result = await response.json()
                    
                    # Extract response text
                    response_text = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                    
                    return EnhancedToolResult(
                        status=ToolStatus.SUCCESS,
                        tool_name=self.name,
                        data={
                            "response": response_text,
                            "model": result.get("model"),
                            "usage": result.get("usage"),
                        },
                        message=response_text,
                        start_time=start_time,
                        end_time=datetime.now(),
                        duration_seconds=(datetime.now() - start_time).total_seconds(),
                    )
                    
        except asyncio.TimeoutError:
            return EnhancedToolResult(
                status=ToolStatus.TIMEOUT,
                tool_name=self.name,
                error="LLM query timed out",
                message="⏱️ LLM query timed out",
                start_time=start_time,
                end_time=datetime.now(),
                recovery_suggestions=[
                    "Reduce max_tokens",
                    "Simplify prompt",
                    "Check GPU memory usage",
                ]
            )
        except Exception as e:
            return EnhancedToolResult(
                status=ToolStatus.FAILURE,
                tool_name=self.name,
                error=str(e),
                message=f"❌ LLM query error: {e}",
                start_time=start_time,
                end_time=datetime.now(),
            )


class VLLMHealthTool(VLLMTool):
    """Check vLLM server health."""
    
    def __init__(self):
        super().__init__(
            name="vllm_health",
            description="Check vLLM server health and status"
        )
        self.timeout_seconds = 10.0
    
    async def execute(self) -> EnhancedToolResult:
        """
        Check vLLM health.
        
        Returns:
            EnhancedToolResult with health status
        """
        start_time = datetime.now()
        
        healthy, status = await self._check_vllm_health()
        
        if healthy:
            # Get more details
            try:
                import aiohttp
                async with aiohttp.ClientSession() as session:
                    # Get model info
                    async with session.get(
                        f"{self.vllm_url}/v1/models",
                        timeout=aiohttp.ClientTimeout(total=5)
                    ) as response:
                        if response.status == 200:
                            models = await response.json()
                            model_list = [m.get("id") for m in models.get("data", [])]
                        else:
                            model_list = []
            except Exception:
                model_list = []
            
            return EnhancedToolResult(
                status=ToolStatus.SUCCESS,
                tool_name=self.name,
                data={"healthy": True, "models": model_list},
                message=f"✅ vLLM server is healthy\n- Models: {', '.join(model_list) or 'unknown'}",
                start_time=start_time,
                end_time=datetime.now(),
            )
        else:
            return EnhancedToolResult(
                status=ToolStatus.FAILURE,
                tool_name=self.name,
                data={"healthy": False, "error": status},
                error=status,
                message=f"❌ vLLM server is not available: {status}",
                start_time=start_time,
                end_time=datetime.now(),
                recovery_suggestions=[
                    "Start vLLM server: sbatch scripts/start_vllm.sbatch",
                    "Check if GPU node is available: squeue --me",
                    "Check server logs for errors",
                ]
            )


# =============================================================================
# File Tools
# =============================================================================

class FileReadTool(EnhancedTool):
    """Read file contents safely."""
    
    def __init__(self):
        super().__init__(
            name="file_read",
            description="Read file contents safely",
            requires_permission=True,
        )
        self._file_ops = None
    
    def _get_file_ops(self):
        """Get file operations handler."""
        if self._file_ops is None:
            try:
                from workflow_composer.agents.executor import FileOperations
                self._file_ops = FileOperations(
                    workspace=Path.home() / "BioPipelines"
                )
            except ImportError:
                pass
        return self._file_ops
    
    async def execute(
        self,
        file_path: str,
        encoding: str = "utf-8",
        max_size: int = 1024 * 1024  # 1MB default
    ) -> EnhancedToolResult:
        """
        Read a file safely.
        
        Args:
            file_path: Path to file
            encoding: File encoding
            max_size: Maximum file size to read
            
        Returns:
            EnhancedToolResult with file contents
        """
        start_time = datetime.now()
        file_path = Path(file_path).expanduser().resolve()
        
        if not file_path.exists():
            return EnhancedToolResult(
                status=ToolStatus.FAILURE,
                tool_name=self.name,
                error=f"File not found: {file_path}",
                message=f"❌ File not found: `{file_path}`",
                start_time=start_time,
                end_time=datetime.now(),
            )
        
        file_ops = self._get_file_ops()
        if file_ops:
            result = file_ops.read_file(str(file_path), encoding=encoding)
            # FileContent doesn't have 'success', check if we got content
            if hasattr(result, 'text') and result.text is not None:
                return EnhancedToolResult(
                    status=ToolStatus.SUCCESS,
                    tool_name=self.name,
                    data={"content": result.text, "path": str(file_path)},
                    message=f"📄 Read `{file_path.name}` ({result.size_bytes} bytes)",
                    start_time=start_time,
                    end_time=datetime.now(),
                )
            elif hasattr(result, 'error'):
                return EnhancedToolResult(
                    status=ToolStatus.FAILURE,
                    tool_name=self.name,
                    error=result.error,
                    message=f"❌ Failed to read file: {result.error}",
                    start_time=start_time,
                    end_time=datetime.now(),
                )
            # Fall through to direct read
        
        # Direct read fallback
        try:
            if file_path.stat().st_size > max_size:
                return EnhancedToolResult(
                    status=ToolStatus.FAILURE,
                    tool_name=self.name,
                    error=f"File too large: {file_path.stat().st_size} bytes",
                    message=f"❌ File too large (max {max_size} bytes)",
                    start_time=start_time,
                    end_time=datetime.now(),
                )
            
            content = file_path.read_text(encoding=encoding)
            
            return EnhancedToolResult(
                status=ToolStatus.SUCCESS,
                tool_name=self.name,
                data={"content": content, "path": str(file_path)},
                message=f"📄 Read `{file_path.name}`",
                start_time=start_time,
                end_time=datetime.now(),
            )
            
        except Exception as e:
            return EnhancedToolResult(
                status=ToolStatus.FAILURE,
                tool_name=self.name,
                error=str(e),
                message=f"❌ Error reading file: {e}",
                start_time=start_time,
                end_time=datetime.now(),
            )


class FileWriteTool(EnhancedTool):
    """Write file contents safely."""
    
    def __init__(self):
        super().__init__(
            name="file_write",
            description="Write file contents safely with backup",
            requires_permission=True,
        )
        self._file_ops = None
    
    def _get_file_ops(self):
        """Get file operations handler."""
        if self._file_ops is None:
            try:
                from workflow_composer.agents.executor import FileOperations
                self._file_ops = FileOperations(
                    workspace=Path.home() / "BioPipelines"
                )
            except ImportError:
                pass
        return self._file_ops
    
    async def execute(
        self,
        file_path: str,
        content: str,
        create_backup: bool = True,
        encoding: str = "utf-8",
    ) -> EnhancedToolResult:
        """
        Write a file safely.
        
        Args:
            file_path: Path to file
            content: Content to write
            create_backup: Whether to backup existing file
            encoding: File encoding
            
        Returns:
            EnhancedToolResult confirming write
        """
        start_time = datetime.now()
        file_path = Path(file_path).expanduser().resolve()
        
        file_ops = self._get_file_ops()
        if file_ops:
            result = file_ops.write_file(
                str(file_path),
                content,
                create_backup=create_backup,
                encoding=encoding,
            )
            if result.success:
                return EnhancedToolResult(
                    status=ToolStatus.SUCCESS,
                    tool_name=self.name,
                    data={"path": str(file_path), "bytes": len(content)},
                    message=f"✅ Wrote `{file_path.name}` ({len(content)} bytes)",
                    start_time=start_time,
                    end_time=datetime.now(),
                )
            else:
                return EnhancedToolResult(
                    status=ToolStatus.FAILURE,
                    tool_name=self.name,
                    error=result.error,
                    message=f"❌ Failed to write file: {result.error}",
                    start_time=start_time,
                    end_time=datetime.now(),
                )
        
        # Direct write fallback
        try:
            if create_backup and file_path.exists():
                backup_path = file_path.with_suffix(f"{file_path.suffix}.bak")
                import shutil
                shutil.copy2(file_path, backup_path)
            
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content, encoding=encoding)
            
            return EnhancedToolResult(
                status=ToolStatus.SUCCESS,
                tool_name=self.name,
                data={"path": str(file_path), "bytes": len(content)},
                message=f"✅ Wrote `{file_path.name}`",
                start_time=start_time,
                end_time=datetime.now(),
            )
            
        except Exception as e:
            return EnhancedToolResult(
                status=ToolStatus.FAILURE,
                tool_name=self.name,
                error=str(e),
                message=f"❌ Error writing file: {e}",
                start_time=start_time,
                end_time=datetime.now(),
            )


# =============================================================================
# System Tools
# =============================================================================

class SystemHealthTool(EnhancedTool):
    """Check overall system health."""
    
    def __init__(self):
        super().__init__(
            name="system_health",
            description="Check system health (GPU, disk, memory, SLURM)",
            requires_permission=False,
        )
    
    async def execute(self) -> EnhancedToolResult:
        """
        Check system health.
        
        Returns:
            EnhancedToolResult with health status
        """
        start_time = datetime.now()
        health_status = {}
        issues = []
        
        # Check GPU
        try:
            proc = await asyncio.create_subprocess_exec(
                "nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5)
            if proc.returncode == 0:
                gpu_info = stdout.decode().strip()
                health_status["gpu"] = {"available": True, "info": gpu_info}
            else:
                health_status["gpu"] = {"available": False}
                issues.append("GPU not available")
        except Exception as e:
            health_status["gpu"] = {"available": False, "error": str(e)}
            issues.append(f"GPU check failed: {e}")
        
        # Check disk space
        try:
            import shutil
            total, used, free = shutil.disk_usage(Path.home())
            free_gb = free / (1024 ** 3)
            health_status["disk"] = {
                "free_gb": round(free_gb, 2),
                "healthy": free_gb > 10,
            }
            if free_gb < 10:
                issues.append(f"Low disk space: {free_gb:.1f}GB free")
        except Exception as e:
            health_status["disk"] = {"error": str(e)}
        
        # Check memory
        try:
            with open("/proc/meminfo") as f:
                meminfo = f.read()
            for line in meminfo.split("\n"):
                if "MemAvailable" in line:
                    available_kb = int(line.split()[1])
                    available_gb = available_kb / (1024 ** 2)
                    health_status["memory"] = {
                        "available_gb": round(available_gb, 2),
                        "healthy": available_gb > 4,
                    }
                    if available_gb < 4:
                        issues.append(f"Low memory: {available_gb:.1f}GB available")
                    break
        except Exception as e:
            health_status["memory"] = {"error": str(e)}
        
        # Check SLURM
        try:
            proc = await asyncio.create_subprocess_exec(
                "sinfo", "-h", "-o", "%P %a",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5)
            if proc.returncode == 0:
                health_status["slurm"] = {"available": True, "partitions": stdout.decode().strip()}
            else:
                health_status["slurm"] = {"available": False}
        except Exception:
            health_status["slurm"] = {"available": False}
        
        # Check vLLM
        vllm_tool = VLLMHealthTool()
        vllm_result = await vllm_tool.execute()
        health_status["vllm"] = vllm_result.data
        if not vllm_result.success:
            issues.append("vLLM server not running")
        
        # Build message
        status_emoji = "✅" if not issues else "⚠️"
        message_parts = [f"{status_emoji} **System Health Check**\n"]
        
        if health_status.get("gpu", {}).get("available"):
            message_parts.append(f"- 🎮 GPU: ✅ Available")
        else:
            message_parts.append(f"- 🎮 GPU: ❌ Not available")
        
        if health_status.get("disk", {}).get("healthy"):
            message_parts.append(f"- 💾 Disk: ✅ {health_status['disk']['free_gb']}GB free")
        else:
            message_parts.append(f"- 💾 Disk: ⚠️ Low space")
        
        if health_status.get("memory", {}).get("healthy"):
            message_parts.append(f"- 🧠 Memory: ✅ {health_status['memory']['available_gb']}GB available")
        else:
            message_parts.append(f"- 🧠 Memory: ⚠️ Low")
        
        if health_status.get("slurm", {}).get("available"):
            message_parts.append(f"- 📊 SLURM: ✅ Available")
        else:
            message_parts.append(f"- 📊 SLURM: ❌ Not available")
        
        if health_status.get("vllm", {}).get("healthy"):
            message_parts.append(f"- 🤖 vLLM: ✅ Running")
        else:
            message_parts.append(f"- 🤖 vLLM: ❌ Not running")
        
        if issues:
            message_parts.append(f"\n**Issues:** {', '.join(issues)}")
        
        return EnhancedToolResult(
            status=ToolStatus.SUCCESS if not issues else ToolStatus.PARTIAL,
            tool_name=self.name,
            data=health_status,
            message="\n".join(message_parts),
            start_time=start_time,
            end_time=datetime.now(),
            recovery_suggestions=[
                "Start vLLM: sbatch scripts/start_vllm.sbatch" if "vLLM" in str(issues) else None,
                "Free disk space: rm -rf ~/.cache/huggingface/hub/*" if "disk" in str(issues) else None,
            ]
        )


# =============================================================================
# Register All Tools
# =============================================================================

def register_all_tools():
    """Register all enhanced tools with the global registry."""
    registry = get_tool_registry()
    
    # SLURM tools
    registry.register(SLURMSubmitTool())
    registry.register(SLURMStatusTool())
    registry.register(SLURMCancelTool())
    registry.register(SLURMLogsTool())
    
    # vLLM tools
    registry.register(VLLMQueryTool())
    registry.register(VLLMHealthTool())
    
    # File tools
    registry.register(FileReadTool())
    registry.register(FileWriteTool())
    
    # System tools
    registry.register(SystemHealthTool())
    
    logger.info(f"Registered {len(registry.list_tools())} enhanced tools")
    return registry


# Auto-register on import
_registered = False
if not _registered:
    register_all_tools()
    _registered = True


# =============================================================================
# Convenience Functions
# =============================================================================

async def execute_tool(tool_name: str, *args, **kwargs) -> EnhancedToolResult:
    """
    Execute a tool by name.
    
    Args:
        tool_name: Name of the tool
        *args, **kwargs: Tool arguments
        
    Returns:
        EnhancedToolResult
    """
    registry = get_tool_registry()
    tool = registry.get(tool_name)
    
    if tool is None:
        return EnhancedToolResult(
            status=ToolStatus.FAILURE,
            tool_name=tool_name,
            error=f"Unknown tool: {tool_name}",
            message=f"❌ Unknown tool: {tool_name}",
        )
    
    return await tool.execute(*args, **kwargs)


def execute_tool_sync(tool_name: str, *args, **kwargs) -> EnhancedToolResult:
    """
    Execute a tool synchronously.
    
    Args:
        tool_name: Name of the tool
        *args, **kwargs: Tool arguments
        
    Returns:
        EnhancedToolResult
    """
    registry = get_tool_registry()
    tool = registry.get(tool_name)
    
    if tool is None:
        return EnhancedToolResult(
            status=ToolStatus.FAILURE,
            tool_name=tool_name,
            error=f"Unknown tool: {tool_name}",
            message=f"❌ Unknown tool: {tool_name}",
        )
    
    return tool.execute_sync(*args, **kwargs)
