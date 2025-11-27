"""
Autonomous Operation System
=============================

Continuous monitoring and automated recovery:
- AutonomousAgent: Main agent with full capabilities
- JobMonitor: Watch SLURM jobs and trigger actions on state change
- HealthChecker: Monitor system health (vLLM, disk, GPU)
- RecoveryManager: Automated failure recovery

Example:
    from workflow_composer.agents.autonomous import (
        AutonomousAgent,
        create_agent,
        JobMonitor,
        HealthChecker,
        RecoveryManager,
    )
    
    # Quick start
    agent = create_agent(level="assisted")
    await agent.start_loop()
    
    # Watch a job
    agent.watch_job("12345")
    
    # Check health
    health = await agent.check_health()
    
    # Execute a task
    result = await agent.execute_task("Fix the import error")
"""

from .job_monitor import JobMonitor, JobWatch, JobEvent, JobState, JobInfo
from .health_checker import HealthChecker, HealthStatus, ComponentHealth, SystemHealth
from .recovery import RecoveryManager, RecoveryResult, RecoveryAction, RecoveryLoop
from .agent import AutonomousAgent, Task, TaskStatus, Action, ActionType, create_agent

__all__ = [
    # Main agent
    "AutonomousAgent",
    "create_agent",
    "Task",
    "TaskStatus",
    "Action",
    "ActionType",
    
    # Job monitoring
    "JobMonitor",
    "JobWatch",
    "JobEvent",
    "JobState",
    "JobInfo",
    
    # Health checking
    "HealthChecker",
    "HealthStatus",
    "ComponentHealth",
    "SystemHealth",
    
    # Recovery
    "RecoveryManager",
    "RecoveryResult",
    "RecoveryAction",
    "RecoveryLoop",
]
