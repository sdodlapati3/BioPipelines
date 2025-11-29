"""
Execution Tools Package
=======================

Job submission, monitoring, and system management tools.

Split into submodules for maintainability:
- slurm: SLURM job management (submit, cancel, status, resubmit, list)
- vllm: vLLM server management (health check, restart)
- monitoring: Job monitoring and logs (get_logs, watch_job)

Usage:
    # Direct import (preferred)
    from workflow_composer.agents.tools.execution import submit_job_impl
    
    # Or via parent package
    from workflow_composer.agents.tools import submit_job_impl
"""

# Re-export everything for backward compatibility
from .slurm import (
    SUBMIT_JOB_PATTERNS,
    GET_JOB_STATUS_PATTERNS,
    CANCEL_JOB_PATTERNS,
    RESUBMIT_JOB_PATTERNS,
    LIST_JOBS_PATTERNS,
    submit_job_impl,
    get_job_status_impl,
    cancel_job_impl,
    resubmit_job_impl,
    list_jobs_impl,
)

from .vllm import (
    CHECK_SYSTEM_HEALTH_PATTERNS,
    RESTART_VLLM_PATTERNS,
    check_system_health_impl,
    restart_vllm_impl,
)

from .monitoring import (
    GET_LOGS_PATTERNS,
    WATCH_JOB_PATTERNS,
    get_logs_impl,
    watch_job_impl,
)

__all__ = [
    # SLURM
    "SUBMIT_JOB_PATTERNS",
    "GET_JOB_STATUS_PATTERNS",
    "CANCEL_JOB_PATTERNS",
    "RESUBMIT_JOB_PATTERNS",
    "LIST_JOBS_PATTERNS",
    "submit_job_impl",
    "get_job_status_impl",
    "cancel_job_impl",
    "resubmit_job_impl",
    "list_jobs_impl",
    # vLLM
    "CHECK_SYSTEM_HEALTH_PATTERNS",
    "RESTART_VLLM_PATTERNS",
    "check_system_health_impl",
    "restart_vllm_impl",
    # Monitoring
    "GET_LOGS_PATTERNS",
    "WATCH_JOB_PATTERNS",
    "get_logs_impl",
    "watch_job_impl",
]
