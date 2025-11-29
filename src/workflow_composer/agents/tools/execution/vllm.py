"""
vLLM Server Management Tools
============================

Tools for checking system health and managing vLLM server.

Functions:
    - check_system_health_impl: Check vLLM, GPU, disk, memory status
    - restart_vllm_impl: Restart the vLLM server
"""

import logging
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

from ..base import ToolResult

logger = logging.getLogger(__name__)


# =============================================================================
# PATTERNS
# =============================================================================

CHECK_SYSTEM_HEALTH_PATTERNS = [
    r"(?:check|show|what\s+is)\s+(?:the\s+)?(?:system\s+)?health",
    r"(?:is\s+the\s+)?(?:system|vllm|gpu|disk)\s+(?:healthy|ok|running)",
    r"health\s+(?:check|status)",
]

RESTART_VLLM_PATTERNS = [
    r"(?:restart|reboot|reload)\s+(?:the\s+)?(?:vllm|llm|model)\s*(?:server)?",
    r"(?:vllm|llm)\s+(?:is\s+)?(?:down|not\s+working|crashed)",
    r"(?:start|stop|kill)\s+(?:the\s+)?(?:vllm|llm)\s*(?:server)?",
]


# =============================================================================
# CHECK_SYSTEM_HEALTH
# =============================================================================

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
            
            # Run async check
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as pool:
                        future = pool.submit(asyncio.run, checker.check_all())
                        status = future.result(timeout=30)
                else:
                    status = loop.run_until_complete(checker.check_all())
            except RuntimeError:
                status = asyncio.run(checker.check_all())
            
            # Format status message
            status_emoji = {
                HealthStatus.HEALTHY: "✅",
                HealthStatus.DEGRADED: "⚠️",
                HealthStatus.UNHEALTHY: "❌",
                HealthStatus.UNKNOWN: "❓",
            }
            
            components = []
            for name, check in status.items():
                emoji = status_emoji.get(check.status, "❓")
                components.append(f"| {name.title()} | {emoji} {check.status.value} | {check.message} |")
            
            component_table = "\n".join(components)
            
            # Overall status
            statuses = [c.status for c in status.values()]
            if all(s == HealthStatus.HEALTHY for s in statuses):
                overall = "✅ All Systems Healthy"
            elif any(s == HealthStatus.UNHEALTHY for s in statuses):
                overall = "❌ System Issues Detected"
            elif any(s == HealthStatus.DEGRADED for s in statuses):
                overall = "⚠️ System Degraded"
            else:
                overall = "❓ Status Unknown"
            
            message = f"""🏥 **System Health Check**

**Overall:** {overall}

| Component | Status | Details |
|-----------|--------|---------|
{component_table}

Last checked: {checker.last_check.isoformat() if checker.last_check else 'Never'}
"""
            
            return ToolResult(
                success=True,
                tool_name="check_system_health",
                data={
                    "overall": overall,
                    "components": {k: v.to_dict() for k, v in status.items()},
                },
                message=message
            )
        
        # Fallback: Basic checks
        checks = []
        
        # Check disk space
        try:
            result = subprocess.run(
                ["df", "-h", "."],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split("\n")
                if len(lines) >= 2:
                    parts = lines[1].split()
                    if len(parts) >= 5:
                        used_pct = int(parts[4].rstrip("%"))
                        if used_pct > 90:
                            checks.append("| Disk | ⚠️ Warning | {parts[4]} used |")
                        else:
                            checks.append(f"| Disk | ✅ OK | {parts[4]} used ({parts[3]} free) |")
        except Exception:
            checks.append("| Disk | ❓ Unknown | Could not check |")
        
        # Check GPU
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                parts = result.stdout.strip().split(", ")
                if len(parts) >= 3:
                    checks.append(f"| GPU | ✅ Available | {parts[0]}% util, {parts[1]}/{parts[2]} MB |")
            else:
                checks.append("| GPU | ❌ Not available | nvidia-smi failed |")
        except FileNotFoundError:
            checks.append("| GPU | ❌ Not available | nvidia-smi not found |")
        except Exception as e:
            checks.append(f"| GPU | ❓ Unknown | {e} |")
        
        # Check vLLM
        try:
            import requests
            resp = requests.get("http://localhost:8000/health", timeout=5)
            if resp.status_code == 200:
                checks.append("| vLLM | ✅ Healthy | Server responding |")
            else:
                checks.append(f"| vLLM | ⚠️ Degraded | Status {resp.status_code} |")
        except Exception:
            checks.append("| vLLM | ❌ Down | Not responding |")
        
        check_table = "\n".join(checks) if checks else "| All | ❓ Unknown | No checks performed |"
        
        message = f"""🏥 **System Health Check** (Basic)

| Component | Status | Details |
|-----------|--------|---------|
{check_table}

*For detailed health checks, install the autonomous module.*
"""
        
        return ToolResult(
            success=True,
            tool_name="check_system_health",
            data={"checks": checks},
            message=message
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return ToolResult(
            success=False,
            tool_name="check_system_health",
            error=str(e),
            message=f"❌ Health check error: {e}"
        )


# =============================================================================
# RESTART_VLLM
# =============================================================================

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


__all__ = [
    # Patterns
    "CHECK_SYSTEM_HEALTH_PATTERNS",
    "RESTART_VLLM_PATTERNS",
    # Functions
    "check_system_health_impl",
    "restart_vllm_impl",
]
