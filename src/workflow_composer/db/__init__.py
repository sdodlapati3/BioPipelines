"""Database layer for BioPipelines.

This module provides persistent storage for users, jobs, and tool executions.
Uses SQLAlchemy for ORM and supports PostgreSQL (production) and SQLite (development).

Example:
    >>> from workflow_composer.db import get_db, User, Job
    >>> db = get_db()
    >>> user = db.query(User).filter_by(email="user@example.com").first()
"""

from .models import (
    APIKey,
    Base,
    Job,
    ToolExecution,
    User,
    WorkflowGeneration,
)
from .repositories import (
    JobRepository,
    ToolExecutionRepository,
    UserRepository,
    WorkflowRepository,
)
from .session import (
    DatabaseConfig,
    get_db,
    get_session,
    init_db,
)

__all__ = [
    # Models
    "Base",
    "User",
    "Job",
    "ToolExecution",
    "APIKey",
    "WorkflowGeneration",
    # Repositories
    "UserRepository",
    "JobRepository",
    "ToolExecutionRepository",
    "WorkflowRepository",
    # Session management
    "get_db",
    "get_session",
    "init_db",
    "DatabaseConfig",
]
