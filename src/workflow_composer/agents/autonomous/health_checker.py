"""
Health Checking System
======================

Monitors system health for autonomous operation.

Components:
- vLLM server health
- GPU status (via nvidia-smi)
- Disk space
- Memory usage
- SLURM availability
"""

import asyncio
import logging
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable
import json
import socket
import urllib.request
import urllib.error

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"
    
    @property
    def is_ok(self) -> bool:
        """Check if status is acceptable."""
        return self in {HealthStatus.HEALTHY, HealthStatus.DEGRADED}


@dataclass
class ComponentHealth:
    """Health status for a single component."""
    name: str
    status: HealthStatus
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    checked_at: datetime = field(default_factory=datetime.now)
    response_time_ms: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "details": self.details,
            "checked_at": self.checked_at.isoformat(),
            "response_time_ms": self.response_time_ms,
        }


@dataclass
class SystemHealth:
    """Overall system health."""
    status: HealthStatus
    components: List[ComponentHealth]
    checked_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "status": self.status.value,
            "components": [c.to_dict() for c in self.components],
            "checked_at": self.checked_at.isoformat(),
        }
    
    @property
    def unhealthy_components(self) -> List[ComponentHealth]:
        """Get list of unhealthy components."""
        return [c for c in self.components if c.status == HealthStatus.UNHEALTHY]
    
    @property
    def degraded_components(self) -> List[ComponentHealth]:
        """Get list of degraded components."""
        return [c for c in self.components if c.status == HealthStatus.DEGRADED]


class HealthChecker:
    """
    System health monitoring.
    
    Checks health of:
    - vLLM server (HTTP health endpoint)
    - GPU status (nvidia-smi)
    - Disk space
    - System memory
    - SLURM availability
    
    Example:
        checker = HealthChecker()
        health = await checker.check_all()
        
        if not health.status.is_ok:
            print("System unhealthy!")
            for c in health.unhealthy_components:
                print(f"  {c.name}: {c.message}")
    """
    
    def __init__(
        self,
        vllm_host: str = "localhost",
        vllm_port: int = 8000,
        min_disk_gb: float = 10.0,
        min_memory_gb: float = 4.0,
        min_gpu_memory_gb: float = 8.0,
    ):
        """
        Initialize health checker.
        
        Args:
            vllm_host: vLLM server hostname
            vllm_port: vLLM server port
            min_disk_gb: Minimum disk space threshold (GB)
            min_memory_gb: Minimum RAM threshold (GB)
            min_gpu_memory_gb: Minimum GPU memory threshold (GB)
        """
        self.vllm_host = vllm_host
        self.vllm_port = vllm_port
        self.min_disk_gb = min_disk_gb
        self.min_memory_gb = min_memory_gb
        self.min_gpu_memory_gb = min_gpu_memory_gb
        
        self._last_health: Optional[SystemHealth] = None
        self._check_history: List[SystemHealth] = []
    
    async def check_all(self) -> SystemHealth:
        """
        Check all system components.
        
        Returns:
            SystemHealth with status of all components
        """
        components = await asyncio.gather(
            self.check_vllm(),
            self.check_gpu(),
            self.check_disk(),
            self.check_memory(),
            self.check_slurm(),
            return_exceptions=True,
        )
        
        # Convert exceptions to unhealthy status
        checked = []
        for i, result in enumerate(components):
            if isinstance(result, Exception):
                checked.append(ComponentHealth(
                    name=["vllm", "gpu", "disk", "memory", "slurm"][i],
                    status=HealthStatus.UNKNOWN,
                    message=str(result),
                ))
            else:
                checked.append(result)
        
        # Determine overall status
        if any(c.status == HealthStatus.UNHEALTHY for c in checked):
            overall = HealthStatus.UNHEALTHY
        elif any(c.status == HealthStatus.DEGRADED for c in checked):
            overall = HealthStatus.DEGRADED
        elif any(c.status == HealthStatus.UNKNOWN for c in checked):
            overall = HealthStatus.DEGRADED
        else:
            overall = HealthStatus.HEALTHY
        
        health = SystemHealth(
            status=overall,
            components=checked,
        )
        
        self._last_health = health
        self._check_history.append(health)
        
        # Keep history bounded
        if len(self._check_history) > 100:
            self._check_history = self._check_history[-100:]
        
        return health
    
    async def check_vllm(self) -> ComponentHealth:
        """Check vLLM server health."""
        start = time.time()
        
        try:
            # Try /health endpoint
            url = f"http://{self.vllm_host}:{self.vllm_port}/health"
            
            # Use asyncio to avoid blocking
            loop = asyncio.get_event_loop()
            
            def do_request():
                try:
                    req = urllib.request.Request(url, method="GET")
                    with urllib.request.urlopen(req, timeout=5) as response:
                        return response.status, response.read().decode()
                except urllib.error.URLError as e:
                    return None, str(e)
                except Exception as e:
                    return None, str(e)
            
            status, body = await loop.run_in_executor(None, do_request)
            response_time = (time.time() - start) * 1000
            
            if status == 200:
                return ComponentHealth(
                    name="vllm",
                    status=HealthStatus.HEALTHY,
                    message="vLLM server responding",
                    response_time_ms=response_time,
                )
            else:
                return ComponentHealth(
                    name="vllm",
                    status=HealthStatus.UNHEALTHY,
                    message=f"vLLM returned status {status}",
                    details={"response": body},
                    response_time_ms=response_time,
                )
                
        except Exception as e:
            return ComponentHealth(
                name="vllm",
                status=HealthStatus.UNHEALTHY,
                message=f"Cannot reach vLLM: {e}",
            )
    
    async def check_gpu(self) -> ComponentHealth:
        """Check GPU health via nvidia-smi."""
        try:
            # Run nvidia-smi
            loop = asyncio.get_event_loop()
            
            def run_nvidia_smi():
                return subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=index,name,memory.total,memory.free,memory.used,utilization.gpu,temperature.gpu",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
            
            result = await loop.run_in_executor(None, run_nvidia_smi)
            
            if result.returncode != 0:
                return ComponentHealth(
                    name="gpu",
                    status=HealthStatus.UNHEALTHY,
                    message=f"nvidia-smi failed: {result.stderr}",
                )
            
            # Parse GPU info
            gpus = []
            low_memory_gpus = []
            high_temp_gpus = []
            
            for line in result.stdout.strip().split("\n"):
                if line:
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 7:
                        gpu = {
                            "index": int(parts[0]),
                            "name": parts[1],
                            "memory_total_mb": float(parts[2]),
                            "memory_free_mb": float(parts[3]),
                            "memory_used_mb": float(parts[4]),
                            "utilization_pct": float(parts[5]),
                            "temperature_c": float(parts[6]),
                        }
                        gpus.append(gpu)
                        
                        # Check memory threshold
                        free_gb = gpu["memory_free_mb"] / 1024
                        if free_gb < self.min_gpu_memory_gb:
                            low_memory_gpus.append(gpu["index"])
                        
                        # Check temperature (85C is typically throttling point)
                        if gpu["temperature_c"] > 80:
                            high_temp_gpus.append(gpu["index"])
            
            if not gpus:
                return ComponentHealth(
                    name="gpu",
                    status=HealthStatus.UNHEALTHY,
                    message="No GPUs detected",
                )
            
            # Determine status
            if high_temp_gpus:
                return ComponentHealth(
                    name="gpu",
                    status=HealthStatus.DEGRADED,
                    message=f"GPU(s) running hot: {high_temp_gpus}",
                    details={"gpus": gpus, "high_temp": high_temp_gpus},
                )
            
            if low_memory_gpus:
                return ComponentHealth(
                    name="gpu",
                    status=HealthStatus.DEGRADED,
                    message=f"Low GPU memory on: {low_memory_gpus}",
                    details={"gpus": gpus, "low_memory": low_memory_gpus},
                )
            
            return ComponentHealth(
                name="gpu",
                status=HealthStatus.HEALTHY,
                message=f"{len(gpus)} GPU(s) healthy",
                details={"gpus": gpus},
            )
            
        except FileNotFoundError:
            return ComponentHealth(
                name="gpu",
                status=HealthStatus.UNKNOWN,
                message="nvidia-smi not found",
            )
        except subprocess.TimeoutExpired:
            return ComponentHealth(
                name="gpu",
                status=HealthStatus.UNHEALTHY,
                message="nvidia-smi timed out",
            )
        except Exception as e:
            return ComponentHealth(
                name="gpu",
                status=HealthStatus.UNHEALTHY,
                message=f"GPU check failed: {e}",
            )
    
    async def check_disk(self, path: str = "/") -> ComponentHealth:
        """Check disk space."""
        try:
            loop = asyncio.get_event_loop()
            
            def get_disk_usage():
                import shutil
                return shutil.disk_usage(path)
            
            usage = await loop.run_in_executor(None, get_disk_usage)
            
            total_gb = usage.total / (1024**3)
            free_gb = usage.free / (1024**3)
            used_pct = (usage.used / usage.total) * 100
            
            details = {
                "path": path,
                "total_gb": round(total_gb, 2),
                "free_gb": round(free_gb, 2),
                "used_pct": round(used_pct, 1),
            }
            
            if free_gb < self.min_disk_gb:
                return ComponentHealth(
                    name="disk",
                    status=HealthStatus.UNHEALTHY,
                    message=f"Low disk space: {free_gb:.1f}GB free",
                    details=details,
                )
            
            if free_gb < self.min_disk_gb * 2:
                return ComponentHealth(
                    name="disk",
                    status=HealthStatus.DEGRADED,
                    message=f"Disk space getting low: {free_gb:.1f}GB free",
                    details=details,
                )
            
            return ComponentHealth(
                name="disk",
                status=HealthStatus.HEALTHY,
                message=f"{free_gb:.1f}GB free ({used_pct:.0f}% used)",
                details=details,
            )
            
        except Exception as e:
            return ComponentHealth(
                name="disk",
                status=HealthStatus.UNHEALTHY,
                message=f"Disk check failed: {e}",
            )
    
    async def check_memory(self) -> ComponentHealth:
        """Check system memory."""
        try:
            loop = asyncio.get_event_loop()
            
            def get_memory():
                # Read from /proc/meminfo
                with open("/proc/meminfo") as f:
                    meminfo = {}
                    for line in f:
                        parts = line.split(":")
                        if len(parts) == 2:
                            key = parts[0].strip()
                            value = parts[1].strip().split()[0]  # Get numeric part
                            meminfo[key] = int(value)  # Values are in KB
                    return meminfo
            
            meminfo = await loop.run_in_executor(None, get_memory)
            
            total_kb = meminfo.get("MemTotal", 0)
            free_kb = meminfo.get("MemAvailable", meminfo.get("MemFree", 0))
            
            total_gb = total_kb / (1024**2)
            free_gb = free_kb / (1024**2)
            used_pct = ((total_kb - free_kb) / total_kb) * 100 if total_kb > 0 else 0
            
            details = {
                "total_gb": round(total_gb, 2),
                "free_gb": round(free_gb, 2),
                "used_pct": round(used_pct, 1),
            }
            
            if free_gb < self.min_memory_gb:
                return ComponentHealth(
                    name="memory",
                    status=HealthStatus.UNHEALTHY,
                    message=f"Low memory: {free_gb:.1f}GB available",
                    details=details,
                )
            
            if free_gb < self.min_memory_gb * 2:
                return ComponentHealth(
                    name="memory",
                    status=HealthStatus.DEGRADED,
                    message=f"Memory getting low: {free_gb:.1f}GB available",
                    details=details,
                )
            
            return ComponentHealth(
                name="memory",
                status=HealthStatus.HEALTHY,
                message=f"{free_gb:.1f}GB available ({used_pct:.0f}% used)",
                details=details,
            )
            
        except Exception as e:
            return ComponentHealth(
                name="memory",
                status=HealthStatus.UNHEALTHY,
                message=f"Memory check failed: {e}",
            )
    
    async def check_slurm(self) -> ComponentHealth:
        """Check SLURM availability."""
        try:
            loop = asyncio.get_event_loop()
            start = time.time()
            
            def check_slurm_cmd():
                return subprocess.run(
                    ["sinfo", "--version"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
            
            result = await loop.run_in_executor(None, check_slurm_cmd)
            response_time = (time.time() - start) * 1000
            
            if result.returncode != 0:
                return ComponentHealth(
                    name="slurm",
                    status=HealthStatus.UNHEALTHY,
                    message=f"SLURM unavailable: {result.stderr}",
                    response_time_ms=response_time,
                )
            
            # Get partition info
            def get_partitions():
                return subprocess.run(
                    ["sinfo", "--format=%P|%a|%c|%D", "--noheader"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
            
            part_result = await loop.run_in_executor(None, get_partitions)
            
            partitions = []
            if part_result.returncode == 0:
                for line in part_result.stdout.strip().split("\n"):
                    if line:
                        parts = line.split("|")
                        if len(parts) >= 4:
                            partitions.append({
                                "name": parts[0].rstrip("*"),  # Remove default marker
                                "available": parts[1] == "up",
                                "cpus_per_node": parts[2],
                                "nodes": parts[3],
                            })
            
            available_partitions = [p for p in partitions if p["available"]]
            
            if not available_partitions:
                return ComponentHealth(
                    name="slurm",
                    status=HealthStatus.DEGRADED,
                    message="No available SLURM partitions",
                    details={"partitions": partitions},
                    response_time_ms=response_time,
                )
            
            return ComponentHealth(
                name="slurm",
                status=HealthStatus.HEALTHY,
                message=f"{len(available_partitions)} partition(s) available",
                details={"partitions": partitions},
                response_time_ms=response_time,
            )
            
        except FileNotFoundError:
            return ComponentHealth(
                name="slurm",
                status=HealthStatus.UNKNOWN,
                message="SLURM commands not found",
            )
        except subprocess.TimeoutExpired:
            return ComponentHealth(
                name="slurm",
                status=HealthStatus.UNHEALTHY,
                message="SLURM commands timed out",
            )
        except Exception as e:
            return ComponentHealth(
                name="slurm",
                status=HealthStatus.UNHEALTHY,
                message=f"SLURM check failed: {e}",
            )
    
    def get_last_health(self) -> Optional[SystemHealth]:
        """Get most recent health check result."""
        return self._last_health
    
    def get_health_history(self, limit: int = 10) -> List[SystemHealth]:
        """Get health check history."""
        return self._check_history[-limit:]


async def run_health_check() -> SystemHealth:
    """Quick health check (convenience function)."""
    checker = HealthChecker()
    return await checker.check_all()


def print_health_report(health: SystemHealth) -> None:
    """Print a formatted health report."""
    status_icons = {
        HealthStatus.HEALTHY: "✓",
        HealthStatus.DEGRADED: "⚠",
        HealthStatus.UNHEALTHY: "✗",
        HealthStatus.UNKNOWN: "?",
    }
    
    print(f"\n{'='*50}")
    print(f"System Health: {status_icons[health.status]} {health.status.value.upper()}")
    print(f"Checked at: {health.checked_at.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*50}")
    
    for comp in health.components:
        icon = status_icons[comp.status]
        timing = f" ({comp.response_time_ms:.0f}ms)" if comp.response_time_ms else ""
        print(f"{icon} {comp.name:10} {comp.status.value:10} {comp.message}{timing}")
    
    print()
